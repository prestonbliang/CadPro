"""Opt-in client for a separately deployed image-to-concept-mesh worker.

The worker contract is intentionally small: CadPro sends one bounded JPEG as
raw base64 in JSON to an administrator-configured ``POST /generate`` endpoint,
and the worker returns a binary glTF 2.0 (GLB) body.  This module does not turn
the returned mesh into STEP and never labels it as manufacturing CAD.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from io import BytesIO
import ipaddress
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_MAX_IMAGE_EDGE = 1_024
DEFAULT_MAX_IMAGE_BYTES = 1_000_000
DEFAULT_MAX_SOURCE_PIXELS = 25_000_000
DEFAULT_MAX_RESPONSE_BYTES = 128 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120.0
OUTPUT_FILENAME = "cadpro-ai-concept.glb"

_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_COMPONENT_BYTES = {
    5120: 1,  # BYTE
    5121: 1,  # UNSIGNED_BYTE
    5122: 2,  # SHORT
    5123: 2,  # UNSIGNED_SHORT
    5125: 4,  # UNSIGNED_INT
    5126: 4,  # FLOAT
}


class ConceptMeshError(RuntimeError):
    """Base class for concept-mesh errors safe to show to a user."""


class ConceptMeshServiceError(ConceptMeshError):
    """Raised when the remote worker cannot return a bounded response."""


class ConceptMeshValidationError(ConceptMeshError):
    """Raised when a response is not a usable, self-contained GLB."""


@dataclass(frozen=True)
class ConceptMeshConfig:
    """Server-owned settings for the optional concept-mesh worker.

    Both opt-in flags must be the literal value ``1`` in the environment.  The
    endpoint is server configuration, never a per-job URL, and must use the
    exact ``/generate`` path without credentials, query parameters, or a
    fragment. Remote endpoints require HTTPS; plaintext HTTP is allowed only
    for a loopback worker.
    """

    enabled: bool = False
    license_accepted: bool = False
    endpoint: str | None = field(default=None, repr=False)
    bearer_token: str | None = field(default=None, repr=False, compare=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_source_pixels: int = DEFAULT_MAX_SOURCE_PIXELS

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.license_accepted, bool):
            raise ValueError("Concept-mesh opt-in settings must be booleans")
        for name in (
            "max_response_bytes",
            "max_image_edge",
            "max_image_bytes",
            "max_source_pixels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_image_edge < 64 or self.max_image_edge > 4_096:
            raise ValueError("max_image_edge must be between 64 and 4096")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or not 0 < float(self.timeout_seconds) <= 600
        ):
            raise ValueError("timeout_seconds must be between 0 and 600 seconds")
        if self.endpoint is not None:
            _validate_endpoint(self.endpoint)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ConceptMeshConfig":
        values = os.environ if environ is None else environ
        endpoint = _clean_optional(values.get("CADPRO_ML_MESH_ENDPOINT"))
        token = _clean_optional(values.get("CADPRO_ML_MESH_TOKEN"))
        return cls(
            enabled=_literal_one(values.get("CADPRO_ML_MESH_ENABLED")),
            license_accepted=_literal_one(
                values.get("CADPRO_ML_MESH_LICENSE_ACCEPTED")
            ),
            endpoint=endpoint,
            bearer_token=token,
        )

    @property
    def available(self) -> bool:
        return self.enabled and self.license_accepted and self.endpoint is not None

    @property
    def disabled_reason(self) -> str | None:
        if not self.enabled:
            return "AI concept-mesh generation is disabled by the server."
        if not self.license_accepted:
            return "The concept-mesh model license has not been accepted by the server owner."
        if self.endpoint is None:
            return "The concept-mesh worker endpoint is not configured."
        return None


@dataclass(frozen=True)
class GlbSummary:
    mesh_count: int
    primitive_count: int
    position_accessor_count: int


@dataclass(frozen=True)
class ConceptMeshResult:
    status: str
    glb_path: Path | None
    metadata: Mapping[str, Any]
    warnings: tuple[str, ...] = ()

    @classmethod
    def disabled(cls, reason: str) -> "ConceptMeshResult":
        return cls(
            status="disabled",
            glb_path=None,
            metadata=_metadata(),
            warnings=(reason,),
        )


class _Response(Protocol):
    headers: Any

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


class _Opener(Protocol):
    def open(self, request: Request, timeout: float) -> _Response: ...


class _NoRedirectHandler(HTTPRedirectHandler):
    """Turn every redirect into a request failure instead of following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def generate_concept_mesh(
    representative_image: str | Path,
    output_dir: str | Path,
    *,
    config: ConceptMeshConfig | None = None,
    opener: _Opener | None = None,
) -> ConceptMeshResult:
    """Generate and atomically publish one optional concept GLB.

    Disabled configurations return before reading the image or opening a
    connection.  A caller should pass a representative still for photo sets or
    videos; video decoding deliberately does not live in this module.
    """

    settings = config or ConceptMeshConfig.from_env()
    if not settings.available:
        return ConceptMeshResult.disabled(
            settings.disabled_reason or "AI concept-mesh generation is unavailable."
        )

    encoded_image = _prepare_image(Path(representative_image), settings)
    request_body = json.dumps(
        {"image": base64.b64encode(encoded_image).decode("ascii")},
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {
        "Accept": "model/gltf-binary, application/octet-stream",
        "Content-Type": "application/json",
        "User-Agent": "CadPro-concept-mesh/1",
    }
    if settings.bearer_token:
        headers["Authorization"] = f"Bearer {settings.bearer_token}"
    request = Request(
        settings.endpoint,
        data=request_body,
        headers=headers,
        method="POST",
    )

    transport = opener or build_opener(_NoRedirectHandler())
    try:
        response = transport.open(request, timeout=float(settings.timeout_seconds))
    except Exception:
        raise ConceptMeshServiceError(
            "The concept-mesh worker request failed before a model was returned."
        ) from None
    try:
        payload = _read_bounded_response(response, settings.max_response_bytes)
    except ConceptMeshError:
        raise
    except Exception:
        raise ConceptMeshServiceError(
            "The concept-mesh worker response could not be read."
        ) from None
    finally:
        try:
            response.close()
        except Exception:
            pass

    summary = validate_self_contained_glb(payload)
    destination = _atomic_write(Path(output_dir), payload)
    return ConceptMeshResult(
        status="completed",
        glb_path=destination,
        metadata=_metadata(summary),
    )


async def generate_concept_mesh_async(
    representative_image: str | Path,
    output_dir: str | Path,
    *,
    config: ConceptMeshConfig | None = None,
    opener: _Opener | None = None,
) -> ConceptMeshResult:
    """Run :func:`generate_concept_mesh` without blocking an async caller."""

    return await asyncio.to_thread(
        generate_concept_mesh,
        representative_image,
        output_dir,
        config=config,
        opener=opener,
    )


def validate_self_contained_glb(payload: bytes) -> GlbSummary:
    """Validate the GLB container and its embedded POSITION geometry."""

    if not isinstance(payload, bytes) or len(payload) < 20:
        raise ConceptMeshValidationError("Concept-mesh output is not a complete GLB file.")
    magic, version, declared_length = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ConceptMeshValidationError("Concept-mesh output has an invalid GLB header.")

    chunks: dict[int, bytes] = {}
    offset = 12
    order: list[int] = []
    while offset < len(payload):
        if len(payload) - offset < 8:
            raise ConceptMeshValidationError("Concept-mesh output has a truncated GLB chunk.")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + chunk_length
        if chunk_length % 4 or end > len(payload):
            raise ConceptMeshValidationError("Concept-mesh output has an invalid GLB chunk.")
        if chunk_type not in {_JSON_CHUNK, _BIN_CHUNK} or chunk_type in chunks:
            raise ConceptMeshValidationError("Concept-mesh output has unsupported GLB chunks.")
        chunks[chunk_type] = payload[offset:end]
        order.append(chunk_type)
        offset = end
    if offset != len(payload) or not order or order[0] != _JSON_CHUNK:
        raise ConceptMeshValidationError("Concept-mesh output has an invalid GLB chunk order.")

    try:
        json_payload = chunks[_JSON_CHUNK].rstrip(b" \t\r\n\x00")
        document = json.loads(json_payload.decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        raise ConceptMeshValidationError(
            "Concept-mesh output has an invalid GLB JSON document."
        ) from None
    if not isinstance(document, dict):
        raise ConceptMeshValidationError("Concept-mesh GLB JSON must be an object.")
    asset = document.get("asset")
    if not isinstance(asset, dict) or asset.get("version") != "2.0":
        raise ConceptMeshValidationError("Concept-mesh output is not glTF 2.0.")
    _validate_embedded_uris(document)

    buffer_sizes = _validate_buffers(document, chunks.get(_BIN_CHUNK))
    views = _validate_buffer_views(document, buffer_sizes)
    accessors = document.get("accessors")
    if not isinstance(accessors, list):
        raise ConceptMeshValidationError("Concept-mesh output has no accessor table.")
    meshes = document.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise ConceptMeshValidationError("Concept-mesh output contains no meshes.")

    primitive_count = 0
    position_accessors: set[int] = set()
    for mesh in meshes:
        if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list):
            raise ConceptMeshValidationError("Concept-mesh output has an invalid mesh entry.")
        for primitive in mesh["primitives"]:
            if not isinstance(primitive, dict):
                raise ConceptMeshValidationError(
                    "Concept-mesh output has an invalid mesh primitive."
                )
            primitive_count += 1
            attributes = primitive.get("attributes")
            position_index = attributes.get("POSITION") if isinstance(attributes, dict) else None
            if isinstance(position_index, bool) or not isinstance(position_index, int):
                continue
            _validate_position_accessor(position_index, accessors, views)
            position_accessors.add(position_index)
    if primitive_count == 0 or not position_accessors:
        raise ConceptMeshValidationError(
            "Concept-mesh output has no mesh primitive with POSITION geometry."
        )
    return GlbSummary(
        mesh_count=len(meshes),
        primitive_count=primitive_count,
        position_accessor_count=len(position_accessors),
    )


def _prepare_image(path: Path, config: ConceptMeshConfig) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                width, height = (int(value) for value in source.size)
                if (
                    width <= 0
                    or height <= 0
                    or width * height > config.max_source_pixels
                ):
                    raise ValueError("image bounds")
                source.seek(0)
                image = ImageOps.exif_transpose(source)
                image.load()
    except (
        FileNotFoundError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
    ):
        raise ConceptMeshError(
            "The representative concept-mesh image could not be decoded within safety limits."
        ) from None

    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, "white")
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        image = flattened
    else:
        image = image.convert("RGB")
    image.thumbnail(
        (config.max_image_edge, config.max_image_edge),
        Image.Resampling.LANCZOS,
    )

    for _ in range(18):
        for quality in (88, 78, 68, 58, 48, 38):
            output = BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=False,
            )
            encoded = output.getvalue()
            if len(encoded) <= config.max_image_bytes:
                return encoded
        if min(image.size) <= 64:
            break
        next_size = (
            max(64, int(image.width * 0.8)),
            max(64, int(image.height * 0.8)),
        )
        image = image.resize(next_size, Image.Resampling.LANCZOS)
    raise ConceptMeshError(
        "The representative concept-mesh image could not be compressed within safety limits."
    )


def _read_bounded_response(response: _Response, maximum: int) -> bytes:
    status = getattr(response, "status", None)
    if status is None:
        try:
            status = response.getcode()  # type: ignore[attr-defined]
        except Exception:
            status = 200
    if status != 200:
        raise ConceptMeshServiceError(
            "The concept-mesh worker returned an unsuccessful status."
        )
    content_encoding = _header(response, "Content-Encoding")
    if content_encoding and content_encoding.lower() != "identity":
        raise ConceptMeshServiceError(
            "The concept-mesh worker returned an unsupported encoded response."
        )
    declared = _header(response, "Content-Length")
    if declared:
        try:
            declared_length = int(declared)
        except (TypeError, ValueError):
            raise ConceptMeshServiceError(
                "The concept-mesh worker returned an invalid response length."
            ) from None
        if declared_length < 0 or declared_length > maximum:
            raise ConceptMeshServiceError(
                "The concept-mesh worker response exceeded the configured size limit."
            )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(64 * 1024, maximum - total + 1))
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise ConceptMeshServiceError(
                "The concept-mesh worker returned a non-binary response."
            )
        total += len(chunk)
        if total > maximum:
            raise ConceptMeshServiceError(
                "The concept-mesh worker response exceeded the configured size limit."
            )
        chunks.append(chunk)
    if declared and total != declared_length:
        raise ConceptMeshServiceError(
            "The concept-mesh worker response ended before its declared length."
        )
    if not chunks:
        raise ConceptMeshServiceError("The concept-mesh worker returned an empty response.")
    return b"".join(chunks)


def _header(response: _Response, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    return str(value) if value is not None else None


def _validate_buffers(document: Mapping[str, Any], bin_chunk: bytes | None) -> list[int]:
    buffers = document.get("buffers")
    if not isinstance(buffers, list) or not buffers:
        raise ConceptMeshValidationError("Concept-mesh output has no embedded geometry buffer.")
    sizes: list[int] = []
    for index, buffer in enumerate(buffers):
        if not isinstance(buffer, dict):
            raise ConceptMeshValidationError("Concept-mesh output has an invalid buffer entry.")
        declared = buffer.get("byteLength")
        if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
            raise ConceptMeshValidationError("Concept-mesh output has an invalid buffer length.")
        uri = buffer.get("uri")
        if uri is None:
            if index != 0 or bin_chunk is None:
                raise ConceptMeshValidationError(
                    "Concept-mesh output references missing embedded buffer data."
                )
            actual = len(bin_chunk)
        else:
            actual = len(_decode_data_uri(uri))
        if actual < declared or actual > declared + 3:
            raise ConceptMeshValidationError(
                "Concept-mesh output buffer length does not match its embedded data."
            )
        sizes.append(declared)
    return sizes


def _validate_buffer_views(
    document: Mapping[str, Any], buffer_sizes: list[int]
) -> list[tuple[int, int, int, int | None]]:
    raw_views = document.get("bufferViews")
    if not isinstance(raw_views, list) or not raw_views:
        raise ConceptMeshValidationError("Concept-mesh output has no buffer views.")
    views: list[tuple[int, int, int, int | None]] = []
    for view in raw_views:
        if not isinstance(view, dict):
            raise ConceptMeshValidationError("Concept-mesh output has an invalid buffer view.")
        buffer_index = view.get("buffer")
        offset = view.get("byteOffset", 0)
        length = view.get("byteLength")
        stride = view.get("byteStride")
        if (
            isinstance(buffer_index, bool)
            or not isinstance(buffer_index, int)
            or not 0 <= buffer_index < len(buffer_sizes)
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            or offset + length > buffer_sizes[buffer_index]
            or (
                stride is not None
                and (
                    isinstance(stride, bool)
                    or not isinstance(stride, int)
                    or not 4 <= stride <= 252
                )
            )
        ):
            raise ConceptMeshValidationError("Concept-mesh output has an invalid buffer view.")
        views.append((buffer_index, offset, length, stride))
    return views


def _validate_position_accessor(
    index: int,
    accessors: list[Any],
    views: list[tuple[int, int, int, int | None]],
) -> None:
    if not 0 <= index < len(accessors) or not isinstance(accessors[index], dict):
        raise ConceptMeshValidationError("Concept-mesh POSITION accessor is invalid.")
    accessor = accessors[index]
    view_index = accessor.get("bufferView")
    offset = accessor.get("byteOffset", 0)
    component_type = accessor.get("componentType")
    count = accessor.get("count")
    if (
        accessor.get("type") != "VEC3"
        or isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or not 0 <= view_index < len(views)
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or component_type not in _COMPONENT_BYTES
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ConceptMeshValidationError("Concept-mesh POSITION accessor is invalid.")
    _buffer_index, _view_offset, view_length, stride = views[view_index]
    element_bytes = 3 * _COMPONENT_BYTES[component_type]
    step = stride or element_bytes
    if step < element_bytes or offset + (count - 1) * step + element_bytes > view_length:
        raise ConceptMeshValidationError("Concept-mesh POSITION data exceeds its buffer view.")


def _validate_embedded_uris(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "uri":
                if not isinstance(item, str) or not item.startswith("data:"):
                    raise ConceptMeshValidationError(
                        "Concept-mesh output references an external resource."
                    )
                _decode_data_uri(item)
            else:
                _validate_embedded_uris(item)
    elif isinstance(value, list):
        for item in value:
            _validate_embedded_uris(item)


def _decode_data_uri(value: Any) -> bytes:
    if not isinstance(value, str) or not value.startswith("data:"):
        raise ConceptMeshValidationError(
            "Concept-mesh output references an external resource."
        )
    header, separator, encoded = value.partition(",")
    if not separator or not header.lower().endswith(";base64"):
        raise ConceptMeshValidationError("Concept-mesh output has an invalid embedded resource.")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        raise ConceptMeshValidationError(
            "Concept-mesh output has an invalid embedded resource."
        ) from None


def _atomic_write(output_dir: Path, payload: bytes) -> Path:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".cadpro-ai-concept-",
            suffix=".tmp",
            dir=output_dir,
        )
    except OSError:
        raise ConceptMeshError("The concept-mesh output directory is not writable.") from None
    temporary = Path(temporary_name)
    destination = output_dir / OUTPUT_FILENAME
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError:
        raise ConceptMeshError("The concept-mesh artifact could not be published.") from None
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def _metadata(summary: GlbSummary | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "hunyuan-compatible-worker",
        "artifact_kind": "ai_concept_mesh",
        "format": "glb",
        "metric_scale": False,
        "manufacturing_cad": False,
        "derived_from_step": False,
    }
    if summary is not None:
        metadata.update(
            {
                "mesh_count": summary.mesh_count,
                "primitive_count": summary.primitive_count,
                "position_accessor_count": summary.position_accessor_count,
            }
        )
    return metadata


def _validate_endpoint(value: str) -> None:
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme.lower() in {"http", "https"}
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and parts.path == "/generate"
            and not parts.query
            and not parts.fragment
        )
        _ = parts.port
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(
            "CADPRO_ML_MESH_ENDPOINT must be an HTTP(S) URL with the exact /generate path"
        )
    if parts.scheme.lower() == "http" and not _is_loopback_host(parts.hostname or ""):
        raise ValueError(
            "A remote concept-mesh endpoint must use HTTPS; plaintext HTTP is loopback-only"
        )


def _is_loopback_host(hostname: str) -> bool:
    normalized = hostname.lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _literal_one(value: str | None) -> bool:
    return value is not None and value.strip() == "1"


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None

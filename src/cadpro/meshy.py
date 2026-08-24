"""Secure, opt-in client for Meshy's official hosted generation APIs.

The module deliberately treats Meshy output as a non-metric visual asset.  It
does not convert the returned triangle mesh to STEP and never labels it as
manufacturing CAD.  API keys remain server-owned, signed asset URLs are
downloaded immediately, and neither values nor provider response bodies are
included in user-facing errors or provenance reports.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
import time
from typing import Any, Callable, ClassVar, Literal, Mapping, Protocol, Sequence
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from cad_diff.html_report import render_html
from cadpro.ml_mesh import validate_self_contained_glb


API_ORIGIN = "https://api.meshy.ai"
TEXT_ENDPOINT = "/openapi/v2/text-to-3d"
IMAGE_ENDPOINT = "/openapi/v1/image-to-3d"
MULTI_IMAGE_ENDPOINT = "/openapi/v1/multi-image-to-3d"
RIGGING_ENDPOINT = "/openapi/v1/rigging"

DEFAULT_TIMEOUT_SECONDS = 15 * 60.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_MAX_JSON_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_ASSET_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_IMAGE_EDGE = 2_048
DEFAULT_MAX_IMAGE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_TOTAL_IMAGE_BYTES = 12 * 1024 * 1024
DEFAULT_MAX_SOURCE_PIXELS = 25_000_000
MAX_PROMPT_CHARS = 600

_SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PENDING_STATUSES = {"PENDING", "IN_PROGRESS"}
_FAILED_STATUSES = {"FAILED", "CANCELED", "CANCELLED", "EXPIRED"}
_JSON_CHUNK = 0x4E4F534A


class MeshyError(RuntimeError):
    """A Meshy failure whose message is safe to expose to a user."""


@dataclass(frozen=True)
class MeshyConfig:
    """Server-owned Meshy credentials and bounded transport settings."""

    enabled: bool = False
    api_key: str | None = field(default=None, repr=False, compare=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES
    max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES
    max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_total_image_bytes: int = DEFAULT_MAX_TOTAL_IMAGE_BYTES
    max_source_pixels: int = DEFAULT_MAX_SOURCE_PIXELS

    api_origin: ClassVar[str] = API_ORIGIN

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("api_key must be non-empty text or None")
        for name in ("timeout_seconds", "http_timeout_seconds", "poll_interval_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if float(self.timeout_seconds) > 24 * 60 * 60:
            raise ValueError("timeout_seconds may not exceed 24 hours")
        if float(self.http_timeout_seconds) > 10 * 60:
            raise ValueError("http_timeout_seconds may not exceed 10 minutes")
        if float(self.poll_interval_seconds) > 60:
            raise ValueError("poll_interval_seconds may not exceed 60 seconds")
        for name in (
            "max_json_bytes",
            "max_asset_bytes",
            "max_image_edge",
            "max_image_bytes",
            "max_total_image_bytes",
            "max_source_pixels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 64 <= self.max_image_edge <= 8_192:
            raise ValueError("max_image_edge must be between 64 and 8192")
        if self.max_total_image_bytes < self.max_image_bytes:
            raise ValueError("max_total_image_bytes must be at least max_image_bytes")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "MeshyConfig":
        values = os.environ if environ is None else environ
        key = values.get("MESHY_API_KEY")
        clean_key = key.strip() if isinstance(key, str) and key.strip() else None
        return cls(
            enabled=values.get("CADPRO_MESHY_ENABLED", "").strip() == "1",
            api_key=clean_key,
        )

    @property
    def available(self) -> bool:
        return self.enabled and self.api_key is not None


@dataclass(frozen=True)
class MeshyOptions:
    """Generation, remeshing, texturing, and optional humanoid-rig settings."""

    texture: bool = True
    pbr: bool = False
    target_faces: int = 30_000
    topology: Literal["triangle", "quad"] = "triangle"
    texture_resolution: Literal["2k", "4k"] = "2k"
    rig_humanoid: bool = False
    height_meters: float | None = None

    def __post_init__(self) -> None:
        for name in ("texture", "pbr", "rig_humanoid"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if (
            isinstance(self.target_faces, bool)
            or not isinstance(self.target_faces, int)
            or not 100 <= self.target_faces <= 300_000
        ):
            raise ValueError("target_faces must be between 100 and 300000")
        if self.topology not in {"triangle", "quad"}:
            raise ValueError("topology must be 'triangle' or 'quad'")
        if self.texture_resolution not in {"2k", "4k"}:
            raise ValueError("texture_resolution must be '2k' or '4k'")
        if self.pbr and not self.texture:
            raise ValueError("PBR maps require texture generation")
        if self.rig_humanoid and not self.texture:
            raise ValueError("Meshy humanoid rigging requires a textured mesh")
        if self.height_meters is not None:
            value = self.height_meters
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 < float(value) <= 100
            ):
                raise ValueError("height_meters must be greater than 0 and at most 100")
            if not self.rig_humanoid:
                raise ValueError("height_meters is only used when rig_humanoid is enabled")

    @property
    def height(self) -> float | None:
        """Convenience alias for the official ``height_meters`` field."""

        return self.height_meters


@dataclass(frozen=True)
class MeshyResult:
    glb_path: Path
    stl_path: Path | None
    preview_path: Path
    report_path: Path
    rigged_glb_path: Path | None
    metadata: Mapping[str, Any]
    warnings: tuple[str, ...] = ()


class _Response(Protocol):
    headers: Any
    status: int

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


class _Opener(Protocol):
    def open(self, request: Request, timeout: float) -> _Response: ...


class _NoRedirectHandler(HTTPRedirectHandler):
    """Do not allow an API or signed-asset response to redirect elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


@dataclass(frozen=True)
class _GlbFeatures:
    textured: bool
    pbr_maps: bool
    rigged: bool


def generate_meshy_asset(
    output_dir: str | Path,
    *,
    prompt: str | None = None,
    image_paths: Sequence[str | Path] = (),
    config: MeshyConfig | None = None,
    options: MeshyOptions | None = None,
    opener: _Opener | None = None,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    stem: str = "cadpro-ai-asset",
) -> MeshyResult:
    """Generate and locally preserve one Meshy asset with provenance.

    Input may be a text prompt, one image, or two to four ordered images.  An
    image request may also carry a bounded prompt; when texturing is enabled it
    is sent using Meshy's official ``texture_prompt`` guidance field.  Successful
    signed assets are downloaded during this call and all public files are
    staged before publication.
    """

    settings = config or MeshyConfig.from_env()
    if not settings.available:
        raise MeshyError(
            "Meshy generation is disabled or MESHY_API_KEY is not configured."
        )
    if not _SAFE_STEM.fullmatch(stem):
        raise ValueError(
            "stem must be 1-128 safe filename characters and start with a letter or digit"
        )
    selected_options = options or MeshyOptions()
    clean_prompt, paths, input_kind = _validate_inputs(prompt, image_paths)
    prepared_images = _prepare_images(paths, settings)

    transport = opener or build_opener(_NoRedirectHandler())
    sleep_fn = sleep or time.sleep
    clock = monotonic or time.monotonic
    if not callable(sleep_fn) or not callable(clock):
        raise ValueError("sleep and monotonic must be callable")
    deadline = float(clock()) + float(settings.timeout_seconds)

    task_objects: list[Mapping[str, Any]] = []
    task_ids: dict[str, str] = {}
    final_endpoint: str
    final_task_id: str

    if input_kind == "text":
        preview_payload: dict[str, Any] = {
            "mode": "preview",
            "prompt": clean_prompt,
            "should_remesh": True,
            "topology": selected_options.topology,
            "target_polycount": selected_options.target_faces,
            "target_formats": ["glb"] if selected_options.texture else ["glb", "stl"],
        }
        preview_id, preview_task = _create_and_poll(
            TEXT_ENDPOINT,
            preview_payload,
            settings=settings,
            transport=transport,
            sleep=sleep_fn,
            clock=clock,
            deadline=deadline,
        )
        task_ids["preview"] = preview_id
        task_objects.append(preview_task)
        if selected_options.texture:
            refine_payload = {
                "mode": "refine",
                "preview_task_id": preview_id,
                "enable_pbr": selected_options.pbr,
                "texture_resolution": selected_options.texture_resolution,
                "target_formats": ["glb", "stl"],
            }
            refine_id, refine_task = _create_and_poll(
                TEXT_ENDPOINT,
                refine_payload,
                settings=settings,
                transport=transport,
                sleep=sleep_fn,
                clock=clock,
                deadline=deadline,
            )
            task_ids["refine"] = refine_id
            task_objects.append(refine_task)
            final_task_id, final_task = refine_id, refine_task
        else:
            final_task_id, final_task = preview_id, preview_task
        final_endpoint = TEXT_ENDPOINT
    else:
        generation_payload: dict[str, Any] = {
            "should_texture": selected_options.texture,
            "should_remesh": True,
            "topology": selected_options.topology,
            "target_polycount": selected_options.target_faces,
            "target_formats": ["glb", "stl"],
        }
        if selected_options.texture:
            generation_payload.update(
                {
                    "enable_pbr": selected_options.pbr,
                    "texture_resolution": selected_options.texture_resolution,
                }
            )
            if clean_prompt is not None:
                generation_payload["texture_prompt"] = clean_prompt
        if input_kind == "image":
            final_endpoint = IMAGE_ENDPOINT
            generation_payload["image_url"] = prepared_images[0]
        else:
            final_endpoint = MULTI_IMAGE_ENDPOINT
            generation_payload["image_urls"] = list(prepared_images)
        final_task_id, final_task = _create_and_poll(
            final_endpoint,
            generation_payload,
            settings=settings,
            transport=transport,
            sleep=sleep_fn,
            clock=clock,
            deadline=deadline,
        )
        task_ids["generation"] = final_task_id
        task_objects.append(final_task)

    glb_url = _task_model_url(final_task, "glb", required=True)
    glb_payload = _download_asset(
        glb_url,
        settings=settings,
        transport=transport,
        accept="model/gltf-binary, application/octet-stream",
    )
    _validate_glb(glb_payload)
    features = _inspect_glb(glb_payload)

    warning_messages = [
        (
            "Meshy output is a non-metric AI-generated mesh, not manufacturing CAD or "
            "a STEP-derived engineering model."
        )
    ]
    if selected_options.texture and not features.textured:
        warning_messages.append(
            "Texture generation was requested, but embedded texture usage was not verified in the GLB."
        )
    if selected_options.pbr and not features.pbr_maps:
        warning_messages.append(
            "PBR maps were requested, but normal and metallic-roughness maps were not both verified."
        )

    stl_payload: bytes | None = None
    stl_url = _task_model_url(final_task, "stl", required=False)
    if stl_url is not None:
        stl_payload = _download_asset(
            stl_url,
            settings=settings,
            transport=transport,
            accept="model/stl, application/sla, application/octet-stream",
        )
        _validate_stl(stl_payload)
        warning_messages.append(
            "The Meshy STL is non-metric and has not been verified as manifold or print-ready."
        )
    else:
        warning_messages.append("Meshy did not return the requested optional STL asset.")

    rigged_payload: bytes | None = None
    rig_features: _GlbFeatures | None = None
    if selected_options.rig_humanoid:
        rig_payload: dict[str, Any] = {"input_task_id": final_task_id}
        if selected_options.height_meters is not None:
            rig_payload["height_meters"] = float(selected_options.height_meters)
        rig_id, rig_task = _create_and_poll(
            RIGGING_ENDPOINT,
            rig_payload,
            settings=settings,
            transport=transport,
            sleep=sleep_fn,
            clock=clock,
            deadline=deadline,
        )
        task_ids["rigging"] = rig_id
        task_objects.append(rig_task)
        rigged_url = _rigged_glb_url(rig_task)
        rigged_payload = _download_asset(
            rigged_url,
            settings=settings,
            transport=transport,
            accept="model/gltf-binary, application/octet-stream",
        )
        _validate_glb(rigged_payload)
        rig_features = _inspect_glb(rigged_payload)
        if not rig_features.rigged:
            raise MeshyError(
                "Meshy rigging output did not contain a verifiable skin and joint-weight binding."
            )

    credits = _credits(task_objects)
    metadata: dict[str, Any] = {
        "provider": "meshy",
        "artifact_kind": "ai_generated_mesh",
        "input_kind": input_kind,
        "input_image_count": len(paths),
        "metric_scale": False,
        "manufacturing_cad": False,
        "derived_from_step": False,
        "textured": features.textured,
        "pbr": features.pbr_maps,
        "rigged": bool(rig_features and rig_features.rigged),
        "requested_topology": selected_options.topology,
        "requested_target_faces": selected_options.target_faces,
        "requested_texture_resolution": (
            selected_options.texture_resolution if selected_options.texture else None
        ),
        "provider_task_ids": dict(task_ids),
        "consumed_credits": credits,
    }
    if selected_options.height_meters is not None:
        metadata["rig_height_meters"] = float(selected_options.height_meters)

    destination = Path(output_dir)
    names = {
        "glb": f"{stem}.glb",
        "stl": f"{stem}.stl",
        "preview": f"{stem}.preview.html",
        "report": f"{stem}.report.json",
        "rigged": f"{stem}.rigged.glb",
    }
    preview_text = render_html(glb_payload, title=f"CadPro Meshy preview - {stem}")
    artifact_sizes: dict[str, dict[str, Any]] = {
        "glb": {"file": names["glb"], "bytes": len(glb_payload)},
        "preview": {
            "file": names["preview"],
            "bytes": len(preview_text.encode("utf-8")),
        },
    }
    if stl_payload is not None:
        artifact_sizes["stl"] = {"file": names["stl"], "bytes": len(stl_payload)}
    if rigged_payload is not None:
        artifact_sizes["rigged_glb"] = {
            "file": names["rigged"],
            "bytes": len(rigged_payload),
        }
    report = {
        "schema_version": 1,
        "provider": "meshy",
        "provenance": {
            "input_kind": input_kind,
            "input_image_count": len(paths),
            "prompt": clean_prompt,
            "options": {
                "texture": selected_options.texture,
                "pbr": selected_options.pbr,
                "target_faces": selected_options.target_faces,
                "topology": selected_options.topology,
                "texture_resolution": selected_options.texture_resolution,
                "rig_humanoid": selected_options.rig_humanoid,
                "height_meters": selected_options.height_meters,
            },
            "provider_task_ids": dict(task_ids),
            "consumed_credits": credits,
        },
        "classification": dict(metadata),
        "artifacts": artifact_sizes,
        "warnings": list(warning_messages),
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"

    publications: dict[str, bytes] = {
        names["glb"]: glb_payload,
        names["preview"]: preview_text.encode("utf-8"),
        names["report"]: report_text.encode("utf-8"),
    }
    if stl_payload is not None:
        publications[names["stl"]] = stl_payload
    if rigged_payload is not None:
        publications[names["rigged"]] = rigged_payload
    _publish_staged(destination, publications, stem)

    return MeshyResult(
        glb_path=destination / names["glb"],
        stl_path=(destination / names["stl"] if stl_payload is not None else None),
        preview_path=destination / names["preview"],
        report_path=destination / names["report"],
        rigged_glb_path=(
            destination / names["rigged"] if rigged_payload is not None else None
        ),
        metadata=metadata,
        warnings=tuple(warning_messages),
    )


def _validate_inputs(
    prompt: str | None,
    image_paths: Sequence[str | Path],
) -> tuple[str | None, tuple[Path, ...], Literal["text", "image", "multi_image"]]:
    if isinstance(image_paths, (str, Path)):
        raise ValueError("image_paths must be a sequence of one to four paths")
    try:
        paths = tuple(Path(path) for path in image_paths)
    except (TypeError, ValueError):
        raise ValueError("image_paths must contain filesystem paths") from None

    clean_prompt: str | None = None
    if prompt is not None:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be text or None")
        clean_prompt = " ".join(prompt.split())
        if not clean_prompt or len(clean_prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt must contain 1 to {MAX_PROMPT_CHARS} characters")
        if any(not character.isprintable() for character in clean_prompt):
            raise ValueError("prompt may not contain control characters")
    if clean_prompt is not None and not paths:
        return clean_prompt, (), "text"
    if len(paths) == 1:
        return clean_prompt, paths, "image"
    if 2 <= len(paths) <= 4:
        return clean_prompt, paths, "multi_image"
    raise ValueError("Provide one text prompt or one to four object images")


def _prepare_images(paths: Sequence[Path], config: MeshyConfig) -> tuple[str, ...]:
    encoded: list[str] = []
    total = 0
    for path in paths:
        payload = _prepare_image(path, config)
        total += len(payload)
        if total > config.max_total_image_bytes:
            raise MeshyError("The prepared Meshy images exceeded the configured upload limit.")
        encoded.append("data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"))
    return tuple(encoded)


def _prepare_image(path: Path, config: MeshyConfig) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                width, height = (int(value) for value in source.size)
                if width <= 0 or height <= 0 or width * height > config.max_source_pixels:
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
        raise MeshyError("A Meshy reference image could not be decoded within safety limits.") from None

    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, "white")
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        image = flattened
    else:
        image = image.convert("RGB")
    image.thumbnail((config.max_image_edge, config.max_image_edge), Image.Resampling.LANCZOS)

    for _ in range(20):
        for quality in (90, 82, 72, 62, 52, 42, 32):
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True, progressive=False)
            payload = output.getvalue()
            if len(payload) <= config.max_image_bytes:
                return payload
        if min(image.size) <= 64:
            break
        image = image.resize(
            (max(64, int(image.width * 0.8)), max(64, int(image.height * 0.8))),
            Image.Resampling.LANCZOS,
        )
    raise MeshyError("A Meshy reference image could not fit the configured upload limit.")


def _create_and_poll(
    endpoint: str,
    payload: Mapping[str, Any],
    *,
    settings: MeshyConfig,
    transport: _Opener,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    deadline: float,
) -> tuple[str, Mapping[str, Any]]:
    created = _api_json(
        "POST",
        endpoint,
        payload,
        settings=settings,
        transport=transport,
    )
    task_id = _task_id(created.get("result"))
    task = _poll_task(
        endpoint,
        task_id,
        settings=settings,
        transport=transport,
        sleep=sleep,
        clock=clock,
        deadline=deadline,
    )
    return task_id, task


def _poll_task(
    endpoint: str,
    task_id: str,
    *,
    settings: MeshyConfig,
    transport: _Opener,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    deadline: float,
) -> Mapping[str, Any]:
    maximum_attempts = max(
        2,
        int(math.ceil(settings.timeout_seconds / settings.poll_interval_seconds)) + 2,
    )
    task_path = f"{endpoint}/{quote(task_id, safe='')}"
    for _ in range(maximum_attempts):
        if float(clock()) >= deadline:
            raise MeshyError("Meshy generation timed out before the asset was ready.")
        task = _api_json(
            "GET",
            task_path,
            None,
            settings=settings,
            transport=transport,
        )
        status = task.get("status")
        if status == "SUCCEEDED":
            return task
        if status in _FAILED_STATUSES:
            raise MeshyError("Meshy could not generate the requested asset.")
        if status not in _PENDING_STATUSES:
            raise MeshyError("Meshy returned an unrecognized task status.")
        remaining = deadline - float(clock())
        if remaining <= 0:
            raise MeshyError("Meshy generation timed out before the asset was ready.")
        sleep(min(float(settings.poll_interval_seconds), remaining))
    raise MeshyError("Meshy generation timed out before the asset was ready.")


def _api_json(
    method: Literal["GET", "POST"],
    path: str,
    payload: Mapping[str, Any] | None,
    *,
    settings: MeshyConfig,
    transport: _Opener,
) -> Mapping[str, Any]:
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.api_key}",
        "User-Agent": "CadPro-Meshy/1",
    }
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(API_ORIGIN + path, data=body, headers=headers, method=method)
    response = _open_safe(transport, request, settings.http_timeout_seconds)
    try:
        raw = _read_bounded(response, settings.max_json_bytes)
    finally:
        _close(response)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise MeshyError("Meshy returned an invalid API response.") from None
    if not isinstance(document, Mapping):
        raise MeshyError("Meshy returned an invalid API response.")
    return document


def _download_asset(
    url: str,
    *,
    settings: MeshyConfig,
    transport: _Opener,
    accept: str,
) -> bytes:
    _validate_asset_url(url)
    request = Request(
        url,
        headers={"Accept": accept, "User-Agent": "CadPro-Meshy/1"},
        method="GET",
    )
    response = _open_safe(transport, request, settings.http_timeout_seconds)
    try:
        return _read_bounded(response, settings.max_asset_bytes)
    finally:
        _close(response)


def _open_safe(transport: _Opener, request: Request, timeout: float) -> _Response:
    try:
        return transport.open(request, timeout=float(timeout))
    except Exception:
        raise MeshyError("A Meshy network request failed.") from None


def _read_bounded(response: _Response, maximum: int) -> bytes:
    status = getattr(response, "status", None)
    if status is None:
        try:
            status = response.getcode()  # type: ignore[attr-defined]
        except Exception:
            status = 200
    if not isinstance(status, int) or not 200 <= status < 300:
        raise MeshyError("Meshy rejected a request or asset download.")
    encoding = _header(response, "Content-Encoding")
    if encoding and encoding.lower() != "identity":
        raise MeshyError("Meshy returned an unsupported encoded response.")
    declared = _header(response, "Content-Length")
    declared_length: int | None = None
    if declared is not None:
        try:
            declared_length = int(declared)
        except (TypeError, ValueError):
            raise MeshyError("Meshy returned an invalid response length.") from None
        if declared_length < 0 or declared_length > maximum:
            raise MeshyError("A Meshy response exceeded the configured size limit.")

    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = response.read(min(64 * 1024, maximum - total + 1))
        except Exception:
            raise MeshyError("A Meshy response could not be read.") from None
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise MeshyError("Meshy returned a non-binary response.")
        total += len(chunk)
        if total > maximum:
            raise MeshyError("A Meshy response exceeded the configured size limit.")
        chunks.append(chunk)
    if declared_length is not None and total != declared_length:
        raise MeshyError("A Meshy response ended before its declared length.")
    if not chunks:
        raise MeshyError("Meshy returned an empty response.")
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


def _close(response: _Response) -> None:
    try:
        response.close()
    except Exception:
        pass


def _task_id(value: Any) -> str:
    if not isinstance(value, str):
        raise MeshyError("Meshy did not return a task identifier.")
    task_id = value.strip()
    if not task_id or len(task_id) > 200 or any(not char.isprintable() for char in task_id):
        raise MeshyError("Meshy returned an invalid task identifier.")
    return task_id


def _task_model_url(task: Mapping[str, Any], format_name: str, *, required: bool) -> str | None:
    model_urls = task.get("model_urls")
    value = model_urls.get(format_name) if isinstance(model_urls, Mapping) else None
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise MeshyError(f"Meshy did not return the requested {format_name.upper()} asset.")
    return value


def _rigged_glb_url(task: Mapping[str, Any]) -> str:
    result = task.get("result")
    value = result.get("rigged_character_glb_url") if isinstance(result, Mapping) else None
    if not isinstance(value, str) or not value:
        raise MeshyError("Meshy did not return the requested rigged GLB asset.")
    return value


def _validate_asset_url(value: str) -> None:
    try:
        parts = urlsplit(value)
        hostname = (parts.hostname or "").lower().rstrip(".")
        valid = (
            parts.scheme.lower() == "https"
            and bool(hostname)
            and (hostname == "meshy.ai" or hostname.endswith(".meshy.ai"))
            and parts.username is None
            and parts.password is None
            and parts.port in (None, 443)
            and bool(parts.path)
            and not parts.fragment
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise MeshyError("Meshy returned an untrusted asset download location.")


def _validate_glb(payload: bytes) -> None:
    try:
        validate_self_contained_glb(payload)
    except Exception:
        raise MeshyError("Meshy returned an invalid or incomplete GLB asset.") from None


def _inspect_glb(payload: bytes) -> _GlbFeatures:
    try:
        offset = 12
        document: Mapping[str, Any] | None = None
        while offset < len(payload):
            chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
            offset += 8
            data = payload[offset : offset + chunk_length]
            offset += chunk_length
            if chunk_type == _JSON_CHUNK:
                parsed = json.loads(data.rstrip(b" \t\r\n\x00").decode("utf-8"))
                if isinstance(parsed, Mapping):
                    document = parsed
                break
        if document is None:
            raise ValueError("missing JSON")
    except Exception:
        raise MeshyError("Meshy returned an invalid or incomplete GLB asset.") from None

    images = document.get("images")
    textures = document.get("textures")
    materials = document.get("materials")
    image_count = len(images) if isinstance(images, list) else 0
    texture_sources: set[int] = set()
    if isinstance(textures, list):
        for index, texture in enumerate(textures):
            source = texture.get("source") if isinstance(texture, Mapping) else None
            if (
                isinstance(source, int)
                and not isinstance(source, bool)
                and 0 <= source < image_count
            ):
                texture_sources.add(index)

    base_color_materials: set[int] = set()
    normal_materials: set[int] = set()
    metallic_roughness_materials: set[int] = set()
    if isinstance(materials, list):
        for material_index, material in enumerate(materials):
            if not isinstance(material, Mapping):
                continue
            pbr = material.get("pbrMetallicRoughness")
            if isinstance(pbr, Mapping) and _texture_reference_valid(
                pbr.get("baseColorTexture"), texture_sources
            ):
                base_color_materials.add(material_index)
            if isinstance(pbr, Mapping) and _texture_reference_valid(
                pbr.get("metallicRoughnessTexture"), texture_sources
            ):
                metallic_roughness_materials.add(material_index)
            if _texture_reference_valid(material.get("normalTexture"), texture_sources):
                normal_materials.add(material_index)

    skins = document.get("skins")
    has_skin = isinstance(skins, list) and bool(skins)
    has_joints = False
    has_weights = False
    textured_material_used = False
    normal_material_used = False
    metallic_roughness_material_used = False
    meshes = document.get("meshes")
    if isinstance(meshes, list):
        for mesh in meshes:
            primitives = mesh.get("primitives") if isinstance(mesh, Mapping) else None
            if not isinstance(primitives, list):
                continue
            for primitive in primitives:
                attributes = primitive.get("attributes") if isinstance(primitive, Mapping) else None
                if isinstance(attributes, Mapping):
                    has_joints = has_joints or _accessor_has_type(
                        document, attributes.get("JOINTS_0"), "VEC4"
                    )
                    has_weights = has_weights or _accessor_has_type(
                        document, attributes.get("WEIGHTS_0"), "VEC4"
                    )
                    if _accessor_has_type(document, attributes.get("TEXCOORD_0"), "VEC2"):
                        material_index = primitive.get("material")
                        if isinstance(material_index, int) and not isinstance(material_index, bool):
                            textured_material_used = (
                                textured_material_used or material_index in base_color_materials
                            )
                            normal_material_used = (
                                normal_material_used or material_index in normal_materials
                            )
                            metallic_roughness_material_used = (
                                metallic_roughness_material_used
                                or material_index in metallic_roughness_materials
                            )
    return _GlbFeatures(
        textured=textured_material_used,
        pbr_maps=normal_material_used and metallic_roughness_material_used,
        rigged=has_skin and has_joints and has_weights,
    )


def _texture_reference_valid(value: Any, valid_texture_indices: set[int]) -> bool:
    if not isinstance(value, Mapping):
        return False
    index = value.get("index")
    return (
        isinstance(index, int)
        and not isinstance(index, bool)
        and index in valid_texture_indices
    )


def _accessor_has_type(document: Mapping[str, Any], value: Any, expected: str) -> bool:
    accessors = document.get("accessors")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not isinstance(accessors, list)
        or not 0 <= value < len(accessors)
        or not isinstance(accessors[value], Mapping)
    ):
        return False
    accessor = accessors[value]
    count = accessor.get("count")
    return (
        accessor.get("type") == expected
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
    )


def _validate_stl(payload: bytes) -> None:
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if triangle_count > 0 and len(payload) == 84 + triangle_count * 50:
            return
    stripped = payload.strip()
    lowered = stripped.lower()
    if lowered.startswith(b"solid") and b"facet" in lowered and lowered.endswith(b"endsolid"):
        return
    raise MeshyError("Meshy returned an invalid or empty STL asset.")


def _credits(tasks: Sequence[Mapping[str, Any]]) -> int | float | None:
    values: list[float] = []
    all_integers = True
    for task in tasks:
        raw = task.get("consumed_credits")
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or raw < 0
        ):
            continue
        values.append(float(raw))
        all_integers = all_integers and isinstance(raw, int)
    if not values:
        return None
    total = sum(values)
    return int(total) if all_integers else total


def _publish_staged(destination: Path, publications: Mapping[str, bytes], stem: str) -> None:
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{stem}-", dir=destination) as temporary_name:
            staging = Path(temporary_name)
            staged: list[tuple[Path, Path]] = []
            for filename, payload in publications.items():
                path = staging / filename
                with path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                staged.append((path, destination / filename))
            for source, target in staged:
                os.replace(source, target)
    except OSError:
        raise MeshyError("Meshy artifacts could not be published locally.") from None

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
from pathlib import Path
import struct

from PIL import Image
import pytest

from cadpro.ml_mesh import (
    ConceptMeshConfig,
    ConceptMeshServiceError,
    ConceptMeshValidationError,
    OUTPUT_FILENAME,
    _NoRedirectHandler,
    generate_concept_mesh,
    generate_concept_mesh_async,
    validate_self_contained_glb,
)


ENDPOINT = "http://127.0.0.1:8099/generate"
TOKEN = "worker-secret-token"


def _document() -> dict:
    return {
        "asset": {"version": "2.0", "generator": "test"},
        "buffers": [{"byteLength": 36}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
            }
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }


def _glb(document: dict | None = None, binary: bytes | None = None) -> bytes:
    source = _document() if document is None else document
    data = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0) if binary is None else binary
    json_chunk = json.dumps(source, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = data + b"\x00" * (-len(data) % 4)
    length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, length),
            struct.pack("<II", len(json_chunk), 0x4E4F534A),
            json_chunk,
            struct.pack("<II", len(bin_chunk), 0x004E4942),
            bin_chunk,
        )
    )


def _image(path: Path) -> Path:
    Image.new("RGBA", (2_000, 1_000), (50, 120, 180, 180)).save(path)
    return path


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200, headers: dict | None = None):
        self.payload = payload
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}
        self.offset = 0
        self.read_calls = 0
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        self.read_calls += 1
        if amount < 0:
            amount = len(self.payload) - self.offset
        result = self.payload[self.offset : self.offset + amount]
        self.offset += len(result)
        return result

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, response: FakeResponse | Exception):
        self.response = response
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _enabled(**changes) -> ConceptMeshConfig:
    values = {
        "enabled": True,
        "license_accepted": True,
        "endpoint": ENDPOINT,
        "bearer_token": TOKEN,
        "timeout_seconds": 7.5,
        "max_image_edge": 128,
        "max_image_bytes": 20_000,
    }
    values.update(changes)
    return ConceptMeshConfig(**values)


def test_default_and_license_gate_return_before_image_or_network(tmp_path):
    opener = FakeOpener(AssertionError("network must not run"))

    default = generate_concept_mesh(
        tmp_path / "missing.png", tmp_path / "out", opener=opener
    )
    license_missing = generate_concept_mesh(
        tmp_path / "missing.png",
        tmp_path / "out",
        config=ConceptMeshConfig(enabled=True, endpoint=ENDPOINT),
        opener=opener,
    )

    assert default.status == license_missing.status == "disabled"
    assert default.glb_path is license_missing.glb_path is None
    assert default.metadata["metric_scale"] is False
    assert default.metadata["manufacturing_cad"] is False
    assert default.metadata["derived_from_step"] is False
    assert "license" in license_missing.warnings[0].lower()
    assert opener.calls == []


def test_environment_requires_literal_opt_ins_and_fixed_generate_endpoint():
    config = ConceptMeshConfig.from_env(
        {
            "CADPRO_ML_MESH_ENABLED": "1",
            "CADPRO_ML_MESH_LICENSE_ACCEPTED": "true",
            "CADPRO_ML_MESH_ENDPOINT": ENDPOINT,
            "CADPRO_ML_MESH_TOKEN": TOKEN,
        }
    )
    assert config.enabled is True
    assert config.license_accepted is False
    assert config.available is False
    assert ENDPOINT not in repr(config)
    assert TOKEN not in repr(config)

    for endpoint in (
        "http://127.0.0.1:8099/other",
        "https://user:pass@example.test/generate",
        "https://example.test/generate?job=1",
    ):
        with pytest.raises(ValueError) as error:
            ConceptMeshConfig(endpoint=endpoint)
        assert endpoint not in str(error.value)

    with pytest.raises(ValueError, match="must use HTTPS"):
        ConceptMeshConfig(endpoint="http://worker.example/generate")
    with pytest.raises(ValueError, match="must use HTTPS"):
        ConceptMeshConfig(
            endpoint="http://worker.example/generate",
            bearer_token=TOKEN,
        )
    assert ConceptMeshConfig(endpoint=ENDPOINT, bearer_token=TOKEN).endpoint == ENDPOINT
    assert (
        ConceptMeshConfig(
            endpoint="https://worker.example/generate",
            bearer_token=TOKEN,
        ).endpoint
        == "https://worker.example/generate"
    )


def test_request_is_posted_as_bounded_raw_base64_and_result_is_atomic(tmp_path):
    source = _image(tmp_path / "source.png")
    payload = _glb()
    response = FakeResponse(payload)
    opener = FakeOpener(response)

    result = generate_concept_mesh(
        source,
        tmp_path / "artifacts",
        config=_enabled(),
        opener=opener,
    )

    assert result.status == "completed"
    assert result.glb_path == tmp_path / "artifacts" / OUTPUT_FILENAME
    assert result.glb_path.read_bytes() == payload
    assert list((tmp_path / "artifacts").glob(".cadpro-ai-concept-*.tmp")) == []
    assert result.metadata == {
        "provider": "hunyuan-compatible-worker",
        "artifact_kind": "ai_concept_mesh",
        "format": "glb",
        "metric_scale": False,
        "manufacturing_cad": False,
        "derived_from_step": False,
        "mesh_count": 1,
        "primitive_count": 1,
        "position_accessor_count": 1,
    }

    request, timeout = opener.calls[0]
    assert request.full_url == ENDPOINT
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 7.5
    posted = json.loads(request.data)
    assert set(posted) == {"image"}
    assert not posted["image"].startswith("data:")
    encoded = base64.b64decode(posted["image"], validate=True)
    assert len(encoded) <= 20_000
    with Image.open(BytesIO(encoded)) as prepared:
        assert prepared.format == "JPEG"
        assert max(prepared.size) <= 128
        assert prepared.mode == "RGB"
    assert response.closed is True


def test_async_interface_uses_the_same_worker_contract(tmp_path):
    source = _image(tmp_path / "source.png")
    opener = FakeOpener(FakeResponse(_glb()))

    result = asyncio.run(
        generate_concept_mesh_async(
            source,
            tmp_path / "async-output",
            config=_enabled(),
            opener=opener,
        )
    )

    assert result.status == "completed"
    assert len(opener.calls) == 1


def test_response_cap_is_checked_before_body_read_and_preserves_old_file(tmp_path):
    source = _image(tmp_path / "source.png")
    output = tmp_path / "artifacts"
    output.mkdir()
    existing = output / OUTPUT_FILENAME
    existing.write_bytes(b"previous-result")
    response = FakeResponse(
        b"private response body",
        headers={"Content-Length": "1000000"},
    )

    with pytest.raises(ConceptMeshServiceError) as error:
        generate_concept_mesh(
            source,
            output,
            config=_enabled(max_response_bytes=512),
            opener=FakeOpener(response),
        )

    assert response.read_calls == 0
    assert response.closed is True
    assert "private response body" not in str(error.value)
    assert existing.read_bytes() == b"previous-result"


def test_transport_errors_do_not_leak_endpoint_token_or_request_body(tmp_path):
    source = _image(tmp_path / "source.png")

    class ExplodingOpener:
        def open(self, request, timeout):
            raise RuntimeError(
                f"endpoint={request.full_url} auth={request.get_header('Authorization')} "
                f"body={request.data!r}"
            )

    with pytest.raises(ConceptMeshServiceError) as captured:
        generate_concept_mesh(
            source,
            tmp_path / "output",
            config=_enabled(),
            opener=ExplodingOpener(),
        )

    message = str(captured.value)
    assert ENDPOINT not in message
    assert TOKEN not in message
    assert "body=" not in message
    assert captured.value.__cause__ is None


def test_redirects_are_not_followed_and_non_success_is_safe(tmp_path):
    assert _NoRedirectHandler().redirect_request(
        None, None, 302, "Found", {}, "https://untrusted.test/next"
    ) is None
    source = _image(tmp_path / "source.png")
    response = FakeResponse(b"redirect body secret", status=302)

    with pytest.raises(ConceptMeshServiceError) as captured:
        generate_concept_mesh(
            source,
            tmp_path / "output",
            config=_enabled(),
            opener=FakeOpener(response),
        )

    assert "redirect body secret" not in str(captured.value)
    assert response.read_calls == 0


def test_glb_validator_requires_embedded_position_geometry():
    summary = validate_self_contained_glb(_glb())
    assert summary.mesh_count == summary.primitive_count == 1

    external = _document()
    external["buffers"][0]["uri"] = "https://untrusted.test/model.bin"
    no_position = _document()
    no_position["meshes"][0]["primitives"][0]["attributes"] = {"NORMAL": 0}
    position_overflow = _document()
    position_overflow["accessors"][0]["count"] = 4

    invalid_payloads = [
        b"not a glb",
        _glb(external),
        _glb(no_position),
        _glb(position_overflow),
    ]
    for payload in invalid_payloads:
        with pytest.raises(ConceptMeshValidationError):
            validate_self_contained_glb(payload)


def test_invalid_glb_never_replaces_existing_artifact(tmp_path):
    source = _image(tmp_path / "source.png")
    output = tmp_path / "artifacts"
    output.mkdir()
    existing = output / OUTPUT_FILENAME
    existing.write_bytes(b"known-good-old-artifact")

    with pytest.raises(ConceptMeshValidationError):
        generate_concept_mesh(
            source,
            output,
            config=_enabled(),
            opener=FakeOpener(FakeResponse(b"not a glb")),
        )

    assert existing.read_bytes() == b"known-good-old-artifact"

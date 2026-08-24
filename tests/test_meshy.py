from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import struct

from PIL import Image
import pytest

from cadpro.meshy import (
    API_ORIGIN,
    IMAGE_ENDPOINT,
    MULTI_IMAGE_ENDPOINT,
    RIGGING_ENDPOINT,
    TEXT_ENDPOINT,
    MeshyConfig,
    MeshyError,
    MeshyOptions,
    MeshyResult,
    generate_meshy_asset,
)


KEY = "meshy-test-secret"
ASSET_ORIGIN = "https://assets.meshy.ai/account/tasks/task/output"
GLB_URL = f"{ASSET_ORIGIN}/model.glb?Expires=123&Signature=private"
STL_URL = f"{ASSET_ORIGIN}/model.stl?Expires=123&Signature=private"
RIGGED_URL = f"{ASSET_ORIGIN}/rigged.glb?Expires=123&Signature=private"


def _document(*, textured: bool = False, pbr: bool = False, rigged: bool = False) -> dict:
    attributes: dict[str, int] = {"POSITION": 0}
    primitive: dict[str, object] = {"attributes": attributes}
    document: dict[str, object] = {
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
            },
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC2",
            },
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5121,
                "count": 3,
                "type": "VEC4",
            },
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC4",
            },
        ],
        "meshes": [{"primitives": [primitive]}],
    }
    if textured:
        attributes["TEXCOORD_0"] = 1
        primitive["material"] = 0
        material: dict[str, object] = {
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}
        }
        if pbr:
            material["normalTexture"] = {"index": 0}
            material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 0}
        document.update(
            {
                "images": [
                    {
                        "uri": "data:image/png;base64,"
                        + base64.b64encode(b"bounded-test-image").decode("ascii")
                    }
                ],
                "textures": [{"source": 0}],
                "materials": [material],
            }
        )
    if rigged:
        attributes["JOINTS_0"] = 2
        attributes["WEIGHTS_0"] = 3
        document["skins"] = [{"joints": [0]}]
    return document


def _glb(*, textured: bool = False, pbr: bool = False, rigged: bool = False) -> bytes:
    document = _document(textured=textured, pbr=pbr, rigged=rigged)
    binary = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, length),
            struct.pack("<II", len(json_chunk), 0x4E4F534A),
            json_chunk,
            struct.pack("<II", len(binary), 0x004E4942),
            binary,
        )
    )


def _stl() -> bytes:
    return b"test".ljust(80, b"\x00") + struct.pack("<I", 1) + b"\x00" * 50


def _image(path: Path, color: tuple[int, int, int, int] = (40, 100, 180, 160)) -> Path:
    Image.new("RGBA", (1_600, 900), color).save(path)
    return path


def _json_response(value: object, *, status: int = 200) -> "FakeResponse":
    return FakeResponse(
        json.dumps(value, separators=(",", ":")).encode("utf-8"),
        status=status,
        headers={"Content-Type": "application/json"},
    )


def _generation_task(
    task_id: str,
    *,
    glb_url: str = GLB_URL,
    stl_url: str | None = STL_URL,
    credits: int = 20,
) -> dict:
    urls = {"glb": glb_url}
    if stl_url is not None:
        urls["stl"] = stl_url
    return {
        "id": task_id,
        "status": "SUCCEEDED",
        "model_urls": urls,
        "consumed_credits": credits,
    }


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status = status
        self.headers = {"Content-Length": str(len(payload)), **(headers or {})}
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


class ScriptedOpener:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, float]] = []

    def open(self, request, timeout: float):
        self.calls.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected network call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _enabled(**changes) -> MeshyConfig:
    values = {
        "enabled": True,
        "api_key": KEY,
        "timeout_seconds": 8.0,
        "http_timeout_seconds": 4.0,
        "poll_interval_seconds": 1.0,
        "max_image_edge": 128,
        "max_image_bytes": 30_000,
        "max_total_image_bytes": 120_000,
        "max_source_pixels": 4_000_000,
        "max_asset_bytes": 2_000_000,
    }
    values.update(changes)
    return MeshyConfig(**values)


def _body(call: tuple[object, float]) -> dict:
    request = call[0]
    return json.loads(request.data.decode("utf-8"))


def test_environment_requires_literal_opt_in_and_never_represents_secret():
    truthy_but_not_literal = MeshyConfig.from_env(
        {"CADPRO_MESHY_ENABLED": "true", "MESHY_API_KEY": KEY}
    )
    missing_key = MeshyConfig.from_env({"CADPRO_MESHY_ENABLED": "1"})
    enabled = MeshyConfig.from_env(
        {"CADPRO_MESHY_ENABLED": " 1 ", "MESHY_API_KEY": f" {KEY} "}
    )

    assert truthy_but_not_literal.enabled is False
    assert missing_key.available is False
    assert enabled.available is True
    assert enabled.api_key == KEY
    assert KEY not in repr(enabled)
    assert MeshyConfig.api_origin == "https://api.meshy.ai"
    assert "endpoint" not in MeshyConfig.__dataclass_fields__


def test_disabled_gate_runs_before_image_or_network(tmp_path):
    opener = ScriptedOpener([AssertionError("network must not run")])

    with pytest.raises(MeshyError, match="disabled"):
        generate_meshy_asset(
            tmp_path / "output",
            image_paths=[tmp_path / "missing.png"],
            config=MeshyConfig(enabled=False, api_key=KEY),
            opener=opener,
        )

    assert opener.calls == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"target_faces": 99}, "target_faces"),
        ({"target_faces": 300_001}, "target_faces"),
        ({"topology": "ngon"}, "topology"),
        ({"texture_resolution": "8k"}, "texture_resolution"),
        ({"texture": False, "pbr": True}, "PBR"),
        ({"texture": False, "rig_humanoid": True}, "rigging"),
        ({"height_meters": 1.8}, "height_meters"),
        ({"rig_humanoid": True, "height_meters": 0}, "height_meters"),
    ],
)
def test_options_enforce_official_bounds(changes, message):
    with pytest.raises(ValueError, match=message):
        MeshyOptions(**changes)

    valid = MeshyOptions(
        texture=True,
        pbr=True,
        target_faces=300_000,
        topology="quad",
        texture_resolution="4k",
        rig_humanoid=True,
        height_meters=1.8,
    )
    assert valid.height == 1.8


@pytest.mark.parametrize(
    "prompt, paths",
    [
        (None, []),
        (None, [f"{index}.png" for index in range(5)]),
        ("x" * 601, []),
    ],
)
def test_input_modes_are_mutually_exclusive_and_bounded(tmp_path, prompt, paths):
    with pytest.raises(ValueError):
        generate_meshy_asset(
            tmp_path / "output",
            prompt=prompt,
            image_paths=paths,
            config=_enabled(),
            opener=ScriptedOpener([]),
        )


def test_single_image_flow_prepares_jpeg_and_publishes_complete_provenance(tmp_path):
    source = _image(tmp_path / "source.png")
    glb = _glb(textured=True, pbr=True)
    opener = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(_generation_task("image-task", credits=30)),
            FakeResponse(glb),
            FakeResponse(_stl()),
        ]
    )
    clock = FakeClock()

    result = generate_meshy_asset(
        tmp_path / "artifacts",
        prompt="  brushed   blue metal  ",
        image_paths=[source],
        config=_enabled(),
        options=MeshyOptions(
            texture=True,
            pbr=True,
            target_faces=55_000,
            topology="quad",
            texture_resolution="4k",
        ),
        opener=opener,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert isinstance(result, MeshyResult)
    assert result.glb_path.read_bytes() == glb
    assert result.stl_path is not None and result.stl_path.read_bytes() == _stl()
    assert result.preview_path.is_file() and result.report_path.is_file()
    assert result.rigged_glb_path is None
    assert result.metadata["metric_scale"] is False
    assert result.metadata["manufacturing_cad"] is False
    assert result.metadata["derived_from_step"] is False
    assert result.metadata["textured"] is True
    assert result.metadata["pbr"] is True
    assert result.metadata["rigged"] is False
    assert result.metadata["provider_task_ids"] == {"generation": "image-task"}
    assert result.metadata["consumed_credits"] == 30

    create_request, timeout = opener.calls[0]
    assert create_request.full_url == API_ORIGIN + IMAGE_ENDPOINT
    assert create_request.get_method() == "POST"
    assert create_request.get_header("Authorization") == f"Bearer {KEY}"
    assert timeout == 4.0
    posted = _body(opener.calls[0])
    assert set(posted) == {
        "image_url",
        "should_texture",
        "enable_pbr",
        "texture_resolution",
        "texture_prompt",
        "should_remesh",
        "topology",
        "target_polycount",
        "target_formats",
    }
    assert posted["should_texture"] is posted["enable_pbr"] is True
    assert posted["should_remesh"] is True
    assert posted["topology"] == "quad"
    assert posted["target_polycount"] == 55_000
    assert posted["texture_resolution"] == "4k"
    assert posted["texture_prompt"] == "brushed blue metal"
    assert posted["target_formats"] == ["glb", "stl"]
    assert posted["image_url"].startswith("data:image/jpeg;base64,")
    prepared = base64.b64decode(posted["image_url"].partition(",")[2], validate=True)
    assert len(prepared) <= 30_000
    with Image.open(BytesIO(prepared)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert max(image.size) <= 128

    assert opener.calls[1][0].full_url == API_ORIGIN + IMAGE_ENDPOINT + "/image-task"
    assert all(call[0].get_header("Authorization") is None for call in opener.calls[2:])
    preview = result.preview_path.read_text(encoding="utf-8")
    assert "window.__CAD_DIFF_GLB_BASE64__" in preview
    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["provenance"]["provider_task_ids"] == {"generation": "image-task"}
    assert report["provenance"]["prompt"] == "brushed blue metal"
    assert report["classification"]["textured"] is True
    assert KEY not in report_text
    assert ASSET_ORIGIN not in report_text
    assert not list(result.glb_path.parent.glob(".cadpro-ai-asset-*"))


def test_two_to_four_images_use_official_multi_image_flow_in_order(tmp_path):
    sources = [
        _image(tmp_path / f"view-{index}.png", (20 * index, 80, 140, 255))
        for index in range(1, 5)
    ]
    glb = _glb()
    opener = ScriptedOpener(
        [
            _json_response({"result": "multi-task"}),
            _json_response(_generation_task("multi-task", stl_url=None)),
            FakeResponse(glb),
        ]
    )

    result = generate_meshy_asset(
        tmp_path / "artifacts",
        prompt="this hint is omitted because texturing is disabled",
        image_paths=sources,
        config=_enabled(),
        options=MeshyOptions(texture=False, topology="triangle", target_faces=12_345),
        opener=opener,
    )

    assert opener.calls[0][0].full_url == API_ORIGIN + MULTI_IMAGE_ENDPOINT
    posted = _body(opener.calls[0])
    assert len(posted["image_urls"]) == 4
    assert all(value.startswith("data:image/jpeg;base64,") for value in posted["image_urls"])
    assert posted == {
        "image_urls": posted["image_urls"],
        "should_texture": False,
        "should_remesh": True,
        "topology": "triangle",
        "target_polycount": 12_345,
        "target_formats": ["glb", "stl"],
    }
    assert "texture_prompt" not in posted
    assert result.metadata["input_kind"] == "multi_image"
    assert result.metadata["input_image_count"] == 4
    assert result.metadata["textured"] is False
    assert result.stl_path is None
    assert any("did not return" in warning for warning in result.warnings)


def test_text_without_texture_uses_only_v2_preview_task(tmp_path):
    opener = ScriptedOpener(
        [
            _json_response({"result": "preview-task"}),
            _json_response(_generation_task("preview-task", credits=10)),
            FakeResponse(_glb()),
            FakeResponse(_stl()),
        ]
    )

    result = generate_meshy_asset(
        tmp_path / "artifacts",
        prompt="  a   precise desk organizer  ",
        config=_enabled(),
        options=MeshyOptions(texture=False, topology="quad", target_faces=8_000),
        opener=opener,
    )

    assert len(opener.calls) == 4
    assert opener.calls[0][0].full_url == API_ORIGIN + TEXT_ENDPOINT
    assert _body(opener.calls[0]) == {
        "mode": "preview",
        "prompt": "a precise desk organizer",
        "should_remesh": True,
        "topology": "quad",
        "target_polycount": 8_000,
        "target_formats": ["glb", "stl"],
    }
    assert result.metadata["provider_task_ids"] == {"preview": "preview-task"}
    assert result.metadata["textured"] is False
    assert json.loads(result.report_path.read_text())["provenance"]["prompt"] == (
        "a precise desk organizer"
    )


def test_text_with_texture_uses_v2_preview_then_refine(tmp_path):
    opener = ScriptedOpener(
        [
            _json_response({"result": "preview-task"}),
            _json_response(
                {"id": "preview-task", "status": "SUCCEEDED", "consumed_credits": 10}
            ),
            _json_response({"result": "refine-task"}),
            _json_response(_generation_task("refine-task", credits=20)),
            FakeResponse(_glb(textured=True, pbr=True)),
            FakeResponse(_stl()),
        ]
    )

    result = generate_meshy_asset(
        tmp_path / "artifacts",
        prompt="a brass robot owl",
        config=_enabled(),
        options=MeshyOptions(texture=True, pbr=True, texture_resolution="4k"),
        opener=opener,
    )

    assert _body(opener.calls[0])["target_formats"] == ["glb"]
    assert opener.calls[2][0].full_url == API_ORIGIN + TEXT_ENDPOINT
    assert _body(opener.calls[2]) == {
        "mode": "refine",
        "preview_task_id": "preview-task",
        "enable_pbr": True,
        "texture_resolution": "4k",
        "target_formats": ["glb", "stl"],
    }
    assert result.metadata["provider_task_ids"] == {
        "preview": "preview-task",
        "refine": "refine-task",
    }
    assert result.metadata["consumed_credits"] == 30
    assert result.metadata["textured"] is result.metadata["pbr"] is True


def test_optional_rigging_uses_completed_task_and_verifies_skin_binding(tmp_path):
    opener = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(_generation_task("image-task", credits=30)),
            FakeResponse(_glb(textured=True)),
            FakeResponse(_stl()),
            _json_response({"result": "rig-task"}),
            _json_response(
                {
                    "id": "rig-task",
                    "status": "SUCCEEDED",
                    "consumed_credits": 5,
                    "result": {"rigged_character_glb_url": RIGGED_URL},
                }
            ),
            FakeResponse(_glb(textured=True, rigged=True)),
        ]
    )

    result = generate_meshy_asset(
        tmp_path / "artifacts",
        image_paths=[_image(tmp_path / "humanoid.png")],
        config=_enabled(),
        options=MeshyOptions(rig_humanoid=True, height_meters=1.82),
        opener=opener,
    )

    assert opener.calls[4][0].full_url == API_ORIGIN + RIGGING_ENDPOINT
    assert _body(opener.calls[4]) == {
        "input_task_id": "image-task",
        "height_meters": 1.82,
    }
    assert result.rigged_glb_path is not None
    assert result.rigged_glb_path.read_bytes() == _glb(textured=True, rigged=True)
    assert result.metadata["rigged"] is True
    assert result.metadata["provider_task_ids"]["rigging"] == "rig-task"
    assert result.metadata["consumed_credits"] == 35
    assert result.metadata["rig_height_meters"] == 1.82


def test_requested_features_are_not_claimed_without_glb_linkage(tmp_path):
    opener = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(_generation_task("image-task")),
            FakeResponse(_glb()),
            FakeResponse(_stl()),
        ]
    )

    result = generate_meshy_asset(
        tmp_path / "artifacts",
        image_paths=[_image(tmp_path / "object.png")],
        config=_enabled(),
        options=MeshyOptions(texture=True, pbr=True),
        opener=opener,
    )

    assert result.metadata["textured"] is False
    assert result.metadata["pbr"] is False
    assert any("not verified" in warning for warning in result.warnings)


def test_polling_is_bounded_by_injected_time_and_preserves_existing_outputs(tmp_path):
    output = tmp_path / "artifacts"
    output.mkdir()
    old = output / "cadpro-ai-asset.glb"
    old.write_bytes(b"known-good")
    pending = {"id": "image-task", "status": "IN_PROGRESS", "progress": 20}
    opener = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(pending),
            _json_response(pending),
        ]
    )
    clock = FakeClock()

    with pytest.raises(MeshyError, match="timed out"):
        generate_meshy_asset(
            output,
            image_paths=[_image(tmp_path / "source.png")],
            config=_enabled(timeout_seconds=2.0, poll_interval_seconds=1.0),
            opener=opener,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert clock.sleeps == [1.0, 1.0]
    assert old.read_bytes() == b"known-good"
    assert not (output / "cadpro-ai-asset.report.json").exists()


def test_oversized_asset_is_rejected_before_read_and_outputs_are_atomic(tmp_path):
    output = tmp_path / "artifacts"
    output.mkdir()
    old = output / "cadpro-ai-asset.glb"
    old.write_bytes(b"known-good")
    oversized = FakeResponse(
        b"private signed response body",
        headers={"Content-Length": "999999"},
    )
    opener = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(_generation_task("image-task")),
            oversized,
        ]
    )

    with pytest.raises(MeshyError, match="size limit") as captured:
        generate_meshy_asset(
            output,
            image_paths=[_image(tmp_path / "source.png")],
            config=_enabled(max_asset_bytes=512),
            opener=opener,
        )

    assert oversized.read_calls == 0
    assert oversized.closed is True
    assert "private signed response body" not in str(captured.value)
    assert old.read_bytes() == b"known-good"


def test_oversized_api_json_is_rejected_before_read(tmp_path):
    oversized = FakeResponse(
        b'{"secret":"provider response"}',
        headers={"Content-Length": "99999"},
    )

    with pytest.raises(MeshyError, match="size limit"):
        generate_meshy_asset(
            tmp_path / "output",
            prompt="a cup",
            config=_enabled(max_json_bytes=128),
            opener=ScriptedOpener([oversized]),
        )

    assert oversized.read_calls == 0
    assert oversized.closed is True


def test_untrusted_signed_download_host_is_rejected_without_requesting_it(tmp_path):
    malicious = "https://assets.meshy.ai.attacker.example/model.glb?token=private"
    opener = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(_generation_task("image-task", glb_url=malicious)),
        ]
    )

    with pytest.raises(MeshyError, match="untrusted") as captured:
        generate_meshy_asset(
            tmp_path / "output",
            image_paths=[_image(tmp_path / "source.png")],
            config=_enabled(),
            opener=opener,
        )

    assert len(opener.calls) == 2
    assert malicious not in str(captured.value)


def test_transport_and_provider_failures_never_leak_key_url_or_body(tmp_path):
    source = _image(tmp_path / "source.png")

    class ExplodingOpener:
        def open(self, request, timeout):
            raise RuntimeError(
                f"url={request.full_url} auth={request.get_header('Authorization')} "
                f"body={request.data!r}"
            )

    with pytest.raises(MeshyError) as captured:
        generate_meshy_asset(
            tmp_path / "output",
            image_paths=[source],
            config=_enabled(),
            opener=ExplodingOpener(),
        )

    message = str(captured.value)
    assert KEY not in message
    assert API_ORIGIN not in message
    assert "data:image" not in message
    assert "body=" not in message
    assert captured.value.__cause__ is None

    rejected = FakeResponse(b"private provider failure", status=401)
    with pytest.raises(MeshyError) as rejected_error:
        generate_meshy_asset(
            tmp_path / "output-2",
            prompt="a chair",
            config=_enabled(),
            opener=ScriptedOpener([rejected]),
        )
    assert rejected.read_calls == 0
    assert "private provider failure" not in str(rejected_error.value)


def test_failed_task_and_invalid_glb_leave_previous_artifacts_unchanged(tmp_path):
    output = tmp_path / "artifacts"
    output.mkdir()
    existing = {
        "cadpro-ai-asset.glb": b"old-glb",
        "cadpro-ai-asset.preview.html": b"old-preview",
        "cadpro-ai-asset.report.json": b"old-report",
    }
    for name, payload in existing.items():
        (output / name).write_bytes(payload)

    failed = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(
                {
                    "id": "image-task",
                    "status": "FAILED",
                    "task_error": {"message": "private failure details"},
                }
            ),
        ]
    )
    with pytest.raises(MeshyError) as failed_error:
        generate_meshy_asset(
            output,
            image_paths=[_image(tmp_path / "first.png")],
            config=_enabled(),
            opener=failed,
        )
    assert "private failure details" not in str(failed_error.value)

    invalid = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(_generation_task("image-task")),
            FakeResponse(b"not a glb"),
        ]
    )
    with pytest.raises(MeshyError, match="invalid"):
        generate_meshy_asset(
            output,
            image_paths=[_image(tmp_path / "second.png")],
            config=_enabled(),
            opener=invalid,
        )

    assert {name: (output / name).read_bytes() for name in existing} == existing


def test_invalid_rigging_output_is_not_claimed_or_published(tmp_path):
    output = tmp_path / "artifacts"
    opener = ScriptedOpener(
        [
            _json_response({"result": "image-task"}),
            _json_response(_generation_task("image-task")),
            FakeResponse(_glb(textured=True)),
            FakeResponse(_stl()),
            _json_response({"result": "rig-task"}),
            _json_response(
                {
                    "id": "rig-task",
                    "status": "SUCCEEDED",
                    "result": {"rigged_character_glb_url": RIGGED_URL},
                }
            ),
            FakeResponse(_glb(textured=True, rigged=False)),
        ]
    )

    with pytest.raises(MeshyError, match="verifiable skin"):
        generate_meshy_asset(
            output,
            image_paths=[_image(tmp_path / "humanoid.png")],
            config=_enabled(),
            options=MeshyOptions(rig_humanoid=True),
            opener=opener,
        )

    assert not output.exists()

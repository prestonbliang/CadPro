from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from cad_diff.step_io import load_step
from cadpro.web import _select_meshy_reference_images, create_app


@dataclass(frozen=True)
class _FakeMeshyResult:
    glb_path: Path
    stl_path: Path | None
    preview_path: Path
    report_path: Path
    rigged_glb_path: Path | None
    metadata: dict[str, object]
    warnings: tuple[str, ...] = ()


def _image_payload() -> bytes:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (32, 20), (128, 100), (10, 10, 10), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _fake_result(output_dir: str | Path, stem: str, *, rigged: bool) -> _FakeMeshyResult:
    output = Path(output_dir)
    glb = output / f"{stem}.glb"
    stl = output / f"{stem}.stl"
    preview = output / f"{stem}.preview.html"
    report = output / f"{stem}.report.json"
    rigged_path = output / f"{stem}.rigged.glb" if rigged else None
    glb.write_bytes(b"glTF-test-visual-asset")
    stl.write_bytes(b"solid visual\nendsolid visual\n")
    preview.write_text("<html><body>visual asset</body></html>", encoding="utf-8")
    report.write_text(
        json.dumps({"classification": "non_metric_ai_visual_mesh"}),
        encoding="utf-8",
    )
    if rigged_path is not None:
        rigged_path.write_bytes(b"glTF-test-rigged-asset")
    return _FakeMeshyResult(
        glb_path=glb,
        stl_path=stl,
        preview_path=preview,
        report_path=report,
        rigged_glb_path=rigged_path,
        metadata={
            "provider": "meshy",
            "artifact_kind": "ai_visual_mesh",
            "metric_scale": False,
            "manufacturing_cad": False,
            "derived_from_step": False,
            "texture_requested": True,
            "rig_requested": rigged,
        },
    )


def _wait(client: TestClient, status_url: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        payload = client.get(status_url).json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("Meshy-backed web job did not finish")


def _enable_meshy(monkeypatch) -> None:
    monkeypatch.setenv("CADPRO_MESHY_ENABLED", "1")
    monkeypatch.setenv("MESHY_API_KEY", "server-only-meshy-secret")


def test_text_job_generates_non_metric_visual_artifacts_without_step(
    tmp_path,
    monkeypatch,
):
    _enable_meshy(monkeypatch)
    calls = []

    def fake_generate(output_dir, *, prompt, image_paths=(), config, options, stem):
        calls.append((prompt, tuple(image_paths), config, options, stem))
        return _fake_result(output_dir, stem, rigged=options.rig_humanoid)

    monkeypatch.setattr("cadpro.meshy.generate_meshy_asset", fake_generate)
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        capabilities = health.json()["generative_mesh"]
        assert capabilities["available"] is True
        assert capabilities["text_to_3d"] is True
        assert capabilities["creates_step"] is False
        assert "server-only-meshy-secret" not in health.text

        accepted = client.post(
            "/api/jobs/text",
            data={
                "prompt": "  a compact orange robot with articulated arms  ",
                "mesh_texture": "true",
                "mesh_pbr": "true",
                "mesh_topology": "quad",
                "mesh_target_faces": "24000",
                "mesh_rig": "true",
                "mesh_height_m": "1.6",
            },
        )
        assert accepted.status_code == 202, accepted.text
        completed = _wait(client, accepted.json()["status_url"])
        assert completed["status"] == "completed", completed.get("error")
        assert completed["kind"] == "text"
        assert completed["input_count"] == 0
        assert completed["parameters"]["prompt"] == (
            "a compact orange robot with articulated arms"
        )
        result = completed["result"]
        filenames = {item["filename"] for item in result["artifacts"]}
        assert "cadpro-ai-asset.glb" in filenames
        assert "cadpro-ai-asset.stl" in filenames
        assert "cadpro-ai-asset.preview.html" in filenames
        assert "cadpro-ai-asset.report.json" in filenames
        assert "cadpro-ai-asset.rigged.glb" in filenames
        assert not any(name.endswith((".step", ".stp")) for name in filenames)
        assert result["metrics"] == {}
        assert result["concept_mesh"]["metric_scale"] is False
        assert result["concept_mesh"]["manufacturing_cad"] is False
        assert result["concept_mesh"]["input_strategy"] == "text_prompt"

    assert len(calls) == 1
    prompt, images, config, options, stem = calls[0]
    assert prompt == "a compact orange robot with articulated arms"
    assert images == ()
    assert config.available is True
    assert options.texture is options.pbr is options.rig_humanoid is True
    assert options.topology == "quad"
    assert options.target_faces == 24_000
    assert options.height_meters == 1.6
    assert stem == "cadpro-ai-asset"


def test_meshy_image_companion_keeps_valid_measured_step(tmp_path, monkeypatch):
    _enable_meshy(monkeypatch)
    calls = []

    def fake_generate(output_dir, *, prompt, image_paths=(), config, options, stem):
        calls.append((prompt, tuple(Path(item).name for item in image_paths), stem))
        return _fake_result(output_dir, stem, rigged=False)

    monkeypatch.setattr("cadpro.meshy.generate_meshy_asset", fake_generate)
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/image",
            files={"file": ("part.png", _image_payload(), "image/png")},
            data={
                "width_mm": "50",
                "depth_mm": "8",
                "concept_mesh": "true",
                "object_hint": "orange electronics enclosure",
                "mesh_target_faces": "18000",
            },
        )
        assert accepted.status_code == 202, accepted.text
        completed = _wait(client, accepted.json()["status_url"])
        assert completed["status"] == "completed", completed.get("error")
        result = completed["result"]
        filenames = {item["filename"] for item in result["artifacts"]}
        assert "cadpro-model.step" in filenames
        assert "cadpro-ai-concept.glb" in filenames
        assert "cadpro-ai-concept.preview.html" in filenames
        assert result["concept_mesh"]["provider"] == "meshy"
        assert result["concept_mesh"]["input_strategy"] == "single_reference"

        step = next(
            item for item in result["artifacts"] if item["filename"] == "cadpro-model.step"
        )
        roundtrip = tmp_path / "meshy-companion.step"
        roundtrip.write_bytes(client.get(step["download_url"]).content)
        assert len(load_step(roundtrip)) == 1

    assert calls == [("orange electronics enclosure", ("image.png",), "cadpro-ai-concept")]


def test_text_generation_is_gated_and_options_are_validated_before_staging(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CADPRO_MESHY_ENABLED", raising=False)
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    app = create_app(storage_parent=tmp_path / "disabled", job_retention_seconds=60)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post("/api/jobs/text", data={"prompt": "a gear"})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "generative_mesh_unavailable"

    _enable_meshy(monkeypatch)
    enabled = create_app(storage_parent=tmp_path / "enabled", job_retention_seconds=60)
    with TestClient(enabled, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/api/jobs/text",
            data={
                "prompt": "a humanoid",
                "mesh_texture": "false",
                "mesh_rig": "true",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_mesh_options"


def test_meshy_reference_selection_uses_four_evenly_spaced_views():
    paths = tuple(Path(f"view-{index:02d}.png") for index in range(20))
    assert _select_meshy_reference_images(paths) == (
        paths[0],
        paths[5],
        paths[10],
        paths[15],
    )
    assert _select_meshy_reference_images(paths[:3]) == paths[:3]

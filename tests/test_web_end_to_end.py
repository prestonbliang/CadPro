import json
import time
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cad_diff.step_io import load_step
from cadpro.enrichment import EnrichmentReport, EnrichmentServiceError
from cadpro.web import create_app


class _DocumentCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.elements = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)
            self.elements[element_id] = (tag, attributes)


def _image_payload() -> bytes:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 20), (125, 100), (10, 10, 10), -1)
    cv2.circle(image, (80, 60), 12, (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _orbit_payload() -> bytes:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 20), (120, 100), (10, 10, 10), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _turntable_video_payload(path: Path) -> bytes:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12,
        (160, 120),
    )
    assert writer.isOpened()
    for index in range(40):
        frame = np.full((120, 160, 3), 255, dtype=np.uint8)
        width = 80 - round(18 * abs(np.sin(2 * np.pi * index / 40)))
        cv2.rectangle(
            frame,
            (80 - width // 2, 20),
            (80 + width // 2, 100),
            (10, 10, 10),
            -1,
        )
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def _wait_for_terminal(client: TestClient, status_url: str) -> dict:
    deadline = time.monotonic() + 45
    snapshot = {}
    while time.monotonic() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200
        snapshot = response.json()
        if snapshot["status"] in {"completed", "failed"}:
            return snapshot
        time.sleep(0.05)
    raise AssertionError(f"reconstruction job did not finish: {snapshot}")


def test_frontend_supports_three_capture_modes_and_has_v2_social_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("CADPRO_PUBLIC_ORIGIN", "https://cadpro.example")
    monkeypatch.delenv("CADPRO_AI_ENRICHMENT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(storage_parent=tmp_path)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.get("/")
        health = client.get("/api/health")
        script = client.get("/static/app.js")
        social_card = client.get("/static/og-capture-to-cad.png")

    assert response.status_code == 200
    assert health.status_code == 200
    assert health.json()["version"] == "2.0.0"
    assert health.json()["capture_limits"]["images"] == {"minimum": 1, "maximum": 1}
    assert health.json()["capture_limits"]["photos"] == {"minimum": 20, "maximum": 50}
    assert health.json()["capture_limits"]["video_views"] == {
        "minimum": 20,
        "maximum": 50,
    }
    assert health.json()["capture_limits"]["image_file"]["maximum_pixels"] == 12_500_000
    assert health.json()["intelligence"] == {
        "available": False,
        "provider": "openai",
        "model": None,
        "vision": False,
        "web_search": False,
        "geometry_mutation": False,
    }
    assert "https://cadpro.example/static/og-capture-to-cad.png" in response.text
    assert "__CADPRO_ORIGIN__" not in response.text
    assert 'data-mode="image"' in response.text
    assert 'data-mode="photos"' in response.text
    assert 'data-mode="video"' in response.text
    assert 'fetch(`/api/jobs/${state.mode}`' in script.text

    collector = _DocumentCollector()
    collector.feed(response.text)
    assert len(collector.ids) == len(set(collector.ids))
    assert {
        "file-input",
        "width-mm",
        "depth-mm",
        "view-count",
        "ai-enhance",
        "build-button",
        "result-section",
    } <= set(collector.ids)
    tag, file_input = collector.elements["file-input"]
    assert tag == "input"
    assert file_input["type"] == "file"
    assert "multiple" not in file_input
    assert "video" not in file_input["accept"]

    assert script.status_code == 200
    assert social_card.status_code == 200
    assert social_card.headers["content-type"].startswith("image/png")
    card = cv2.imdecode(np.frombuffer(social_card.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert card is not None
    assert card.shape[0] >= 630
    assert card.shape[1] >= 1200


def test_frontend_enforces_bounded_inputs_and_sequential_photo_preview_memory():
    assets = Path(__file__).parents[1] / "src" / "cadpro" / "web_assets"
    script = (assets / "app.js").read_text(encoding="utf-8")
    document = (assets / "index.html").read_text(encoding="utf-8")
    styles = (assets / "styles.css").read_text(encoding="utf-8")

    assert "MAX_IMAGE_BYTES = 25 * MEBIBYTE" in script
    assert "MAX_PHOTO_SET_BYTES = 500 * MEBIBYTE" in script
    assert "MAX_VIDEO_BYTES = 2 * GIBIBYTE" in script
    assert "THUMBNAIL_MAX_EDGE = 240" in script
    assert "MAX_IMAGE_EDGE = 8_192" in script
    assert "MAX_IMAGE_PIXELS = 12_500_000" in script
    assert 'mode === "image" && incoming.length !== 1' in script
    assert "photos && incoming.length > 50" in script
    assert "photos ? count >= 20 && count <= 50" in script
    assert "video && incoming.length !== 1" in script
    assert "file.size > MAX_IMAGE_BYTES" in script
    assert "totalBytes > MAX_PHOTO_SET_BYTES" in script
    assert "incoming[0].size > MAX_VIDEO_BYTES" in script
    assert "state.files = [...incoming]" in script
    assert "syntheticProgress" not in script
    assert "new AbortController()" in script
    assert "for (let index = 0; index < files.length; index += 1)" in script
    assert "await inspectImage(files[index], signal)" in script
    assert "state.thumbnails.set(files[index], result.thumbnail)" in script
    assert 'canvas.toDataURL("image/jpeg", 0.74)' in script
    assert "Promise.all(state.files.map" not in script
    assert 'fileInput.multiple = mode === "photos"' in script
    assert 'form.append("files", file, file.name)' in script
    assert 'form.append("file", state.files[0], state.files[0].name)' in script
    assert 'fetch(`/api/jobs/${state.mode}`' in script
    assert 'data-mode="photos"' in document
    assert 'data-mode="video"' in document
    assert 'id="depth-mm"' in document
    assert 'id="view-count" type="range" min="20" max="50"' in document
    assert 'aria-controls="capture-panel"' in document
    assert 'id="capture-panel" role="tabpanel"' in document
    assert 'id="optional-warning" role="status"' in document
    assert "image/tiff" not in document
    assert "image/tiff" not in script
    assert '["ArrowLeft", "ArrowRight", "Home", "End"]' in script
    assert 'result.concept_mesh?.status === "failed"' in script
    assert 'research?.status === "failed"' in script
    assert ".drop-zone:focus-within" in styles
    assert ".mode-button:focus-visible" in styles
    assert ".mode-switch.three-modes { grid-template-columns: 1fr; }" in styles


def test_real_single_image_job_exports_reloadable_measured_solid(tmp_path):
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/image",
            files={"file": ("bracket.png", _image_payload(), "image/png")},
            data={"width_mm": "45", "depth_mm": "7.5"},
        )
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["kind"] == "image"
        assert accepted.json()["input_count"] == 1
        assert accepted.json()["parameters"] == {"width_mm": 45.0, "depth_mm": 7.5}

        snapshot = _wait_for_terminal(client, accepted.json()["status_url"])
        assert snapshot["status"] == "completed", snapshot.get("error")
        assert snapshot["stage"] == "complete"
        assert snapshot["progress"] == 100
        assert snapshot["input_count"] == 1

        result = snapshot["result"]
        metrics = result["metrics"]
        assert metrics["is_valid"] is True
        assert metrics["solid_count"] == 1
        assert metrics["dimensions_mm"] == pytest.approx([45.0, 40.0, 7.5], abs=0.02)
        assert metrics["volume_mm3"] < 45.0 * 40.0 * 7.5
        diagnostic = result["input_diagnostics"][0]
        assert len(result["input_diagnostics"]) == 1
        assert diagnostic["order"] == 0
        assert diagnostic["source_name"] == "image.png"
        assert diagnostic["source_size"] == [160, 120]
        assert diagnostic["frame_index"] is None
        assert 4 <= diagnostic["outline_points"] <= 256
        assert diagnostic["hole_count"] == 1
        assert 0 < diagnostic["foreground_fraction"] < 1

        artifacts = result["artifacts"]
        assert {artifact["filename"] for artifact in artifacts} == {
            "cadpro-model.step",
            "cadpro-model.stl",
            "cadpro-model.glb",
            "cadpro-model.preview.html",
            "cadpro-model.report.json",
        }
        for artifact in artifacts:
            download = client.get(artifact["download_url"])
            assert download.status_code == 200
            assert len(download.content) == artifact["size_bytes"] > 0
            disposition = download.headers["content-disposition"]
            if artifact["filename"].endswith(".html"):
                assert disposition.startswith("inline;")
            else:
                assert disposition.startswith("attachment;")

            if artifact["filename"].endswith(".step"):
                step_path = tmp_path / "roundtrip.step"
                step_path.write_bytes(download.content)
                assert len(load_step(step_path)) == 1
            if artifact["filename"].endswith(".report.json"):
                report = json.loads(download.content)
                assert report["reconstruction"] == {"mode": "image", "input_count": 1}
                assert report["geometry"]["dimensions_mm"] == {
                    "x": pytest.approx(45.0, abs=0.02),
                    "y": pytest.approx(40.0, abs=0.02),
                    "z": pytest.approx(7.5, abs=0.02),
                }


def test_real_image_job_publishes_mocked_enrichment_with_valid_step(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CADPRO_AI_ENRICHMENT", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")
    monkeypatch.setenv("CADPRO_AI_MODEL", "gpt-test")
    provider_calls = []

    def fake_enrich(image_paths, query, *, config):
        provider_calls.append((tuple(image_paths), query, config))
        return EnrichmentReport(
            status="completed",
            provider="openai",
            model=config.model,
            query=query,
            object_identity={
                "common_name": "mounting bracket",
                "manufacturer": "Example Fabrication",
                "model_number": "MB-42",
                "confidence": 0.82,
                "evidence": "The visible profile matches the cited product drawing.",
                "source_urls": ["https://manufacturer.example/mb-42"],
            },
            candidate_dimensions=(
                {
                    "name": "catalog width",
                    "value": 45.0,
                    "unit": "mm",
                    "basis": "published_reference",
                    "confidence": 0.75,
                    "source_url": "https://manufacturer.example/mb-42",
                    "caveat": "Verify the photographed product revision before use.",
                },
            ),
            specification_facts=(),
            cad_feature_observations=(
                {
                    "name": "center opening",
                    "description": "One enclosed opening is visible in the supplied profile.",
                    "evidence_basis": "visible_image",
                    "confidence": 0.9,
                    "source_url": None,
                },
            ),
            uncertainties=("The backside is not visible in a single image.",),
            source_urls=("https://manufacturer.example/mb-42",),
            warnings=("Advisory research did not modify the CAD geometry.",),
        )

    monkeypatch.setattr("cadpro.enrichment.enrich_references", fake_enrich)
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/image",
            files={"file": ("bracket.png", _image_payload(), "image/png")},
            data={
                "width_mm": "45",
                "depth_mm": "7.5",
                "ai_enhance": "true",
                "object_hint": "  mounting\n bracket   MB-42  ",
            },
        )
        assert accepted.status_code == 202, accepted.text
        snapshot = _wait_for_terminal(client, accepted.json()["status_url"])

        assert snapshot["status"] == "completed", snapshot.get("error")
        assert snapshot["result"]["metrics"]["is_valid"] is True
        enrichment = snapshot["result"]["enrichment"]
        assert enrichment["status"] == "completed"
        assert enrichment["query"] == "mounting bracket MB-42"
        assert enrichment["object_identity"]["model_number"] == "MB-42"
        assert enrichment["source_urls"] == [
            "https://manufacturer.example/mb-42"
        ]

        artifacts = snapshot["result"]["artifacts"]
        step_artifact = next(
            artifact for artifact in artifacts if artifact["filename"].endswith(".step")
        )
        step_path = tmp_path / "enriched-roundtrip.step"
        step_path.write_bytes(client.get(step_artifact["download_url"]).content)
        assert len(load_step(step_path)) == 1

        report_artifact = next(
            artifact
            for artifact in artifacts
            if artifact["filename"].endswith(".report.json")
        )
        report = client.get(report_artifact["download_url"]).json()
        assert report["enrichment"] == enrichment
        assert report["geometry"]["solid_count"] == 1
        assert report["geometry"]["is_valid"] is True

    assert len(provider_calls) == 1
    paths, query, config = provider_calls[0]
    assert [path.name for path in paths] == ["image.png"]
    assert query == "mounting bracket MB-42"
    assert config.available is True
    assert config.model == "gpt-test"


def test_real_image_job_survives_mocked_enrichment_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CADPRO_AI_ENRICHMENT", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")
    monkeypatch.setenv("CADPRO_AI_MODEL", "gpt-test")
    provider_calls = []

    def failing_enrich(image_paths, query, *, config):
        provider_calls.append((tuple(image_paths), query, config))
        raise EnrichmentServiceError("The mocked reference provider is unavailable.")

    monkeypatch.setattr("cadpro.enrichment.enrich_references", failing_enrich)
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/image",
            files={"file": ("bracket.png", _image_payload(), "image/png")},
            data={
                "width_mm": "45",
                "depth_mm": "7.5",
                "ai_enhance": "true",
                "object_hint": "mounting bracket",
            },
        )
        assert accepted.status_code == 202, accepted.text
        snapshot = _wait_for_terminal(client, accepted.json()["status_url"])

        assert snapshot["status"] == "completed", snapshot.get("error")
        assert snapshot["error"] is None
        assert snapshot["result"]["metrics"]["is_valid"] is True
        enrichment = snapshot["result"]["enrichment"]
        assert enrichment["status"] == "failed"
        assert enrichment["query"] == "mounting bracket"
        assert enrichment["warnings"] == [
            "The mocked reference provider is unavailable."
        ]

        artifacts = snapshot["result"]["artifacts"]
        assert {artifact["filename"] for artifact in artifacts} == {
            "cadpro-model.step",
            "cadpro-model.stl",
            "cadpro-model.glb",
            "cadpro-model.preview.html",
            "cadpro-model.report.json",
        }
        step_artifact = next(
            artifact for artifact in artifacts if artifact["filename"].endswith(".step")
        )
        step_path = tmp_path / "failed-enrichment-roundtrip.step"
        step_path.write_bytes(client.get(step_artifact["download_url"]).content)
        assert len(load_step(step_path)) == 1

        report_artifact = next(
            artifact
            for artifact in artifacts
            if artifact["filename"].endswith(".report.json")
        )
        report = client.get(report_artifact["download_url"]).json()
        assert report["enrichment"] == enrichment
        assert report["geometry"]["solid_count"] == 1
        assert report["geometry"]["is_valid"] is True

    assert len(provider_calls) == 1


def test_real_twenty_photo_job_exports_reloadable_measured_visual_hull(tmp_path):
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)
    payload = _orbit_payload()
    uploads = [
        ("files", (f"view-{index:02d}.png", payload, "image/png"))
        for index in range(20)
    ]

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/photos",
            files=uploads,
            data={"width_mm": "40", "clockwise": "false"},
        )
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["kind"] == "photos"
        assert accepted.json()["input_count"] == 20
        assert accepted.json()["parameters"] == {"width_mm": 40.0, "clockwise": False}

        snapshot = _wait_for_terminal(client, accepted.json()["status_url"])
        assert snapshot["status"] == "completed", snapshot.get("error")
        assert snapshot["stage"] == "complete"
        assert snapshot["progress"] == 100
        assert len(snapshot["result"]["input_diagnostics"]) == 20
        assert snapshot["result"]["metrics"]["is_valid"] is True
        assert snapshot["result"]["metrics"]["solid_count"] == 1

        artifacts = snapshot["result"]["artifacts"]
        assert {artifact["filename"] for artifact in artifacts} == {
            "cadpro-model.step",
            "cadpro-model.stl",
            "cadpro-model.glb",
            "cadpro-model.preview.html",
            "cadpro-model.report.json",
        }
        step_artifact = next(
            artifact for artifact in artifacts if artifact["filename"].endswith(".step")
        )
        download = client.get(step_artifact["download_url"])
        assert download.status_code == 200
        step_path = tmp_path / "photo-orbit-roundtrip.step"
        step_path.write_bytes(download.content)
        assert len(load_step(step_path)) == 1


def test_real_turntable_video_job_exports_reloadable_measured_visual_hull(tmp_path):
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)
    payload = _turntable_video_payload(tmp_path / "orbit.avi")

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/video",
            files={"file": ("orbit.avi", payload, "video/x-msvideo")},
            data={"width_mm": "55", "views": "20", "clockwise": "false"},
        )
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["kind"] == "video"
        assert accepted.json()["input_count"] == 1
        assert accepted.json()["parameters"] == {
            "width_mm": 55.0,
            "views": 20,
            "start_frame": 0,
            "end_frame": None,
            "clockwise": False,
        }

        snapshot = _wait_for_terminal(client, accepted.json()["status_url"])
        assert snapshot["status"] == "completed", snapshot.get("error")
        assert snapshot["stage"] == "complete"
        assert snapshot["progress"] == 100
        assert len(snapshot["result"]["input_diagnostics"]) == 20
        assert snapshot["result"]["metrics"]["is_valid"] is True
        assert snapshot["result"]["metrics"]["solid_count"] == 1

        step_artifact = next(
            artifact
            for artifact in snapshot["result"]["artifacts"]
            if artifact["filename"].endswith(".step")
        )
        download = client.get(step_artifact["download_url"])
        assert download.status_code == 200
        step_path = tmp_path / "video-roundtrip.step"
        step_path.write_bytes(download.content)
        assert len(load_step(step_path)) == 1

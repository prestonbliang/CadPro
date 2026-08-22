import json
import time
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cad_diff.step_io import load_step
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


def _wait_for_terminal(client: TestClient, status_url: str) -> dict:
    deadline = time.monotonic() + 20
    snapshot = {}
    while time.monotonic() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200
        snapshot = response.json()
        if snapshot["status"] in {"completed", "failed"}:
            return snapshot
        time.sleep(0.05)
    raise AssertionError(f"single-image job did not finish: {snapshot}")


def test_frontend_is_single_image_only_and_has_complete_social_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("CADPRO_PUBLIC_ORIGIN", "https://cadpro.example")
    app = create_app(storage_parent=tmp_path)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.get("/")
        health = client.get("/api/health")
        script = client.get("/static/app.js")
        social_card = client.get("/static/og-single-image.png")
        retired_card = client.get("/static/og.png")

    assert response.status_code == 200
    assert health.status_code == 200
    assert health.json()["version"] == "1.1.0"
    assert health.json()["capture_limits"]["images"] == {"minimum": 1, "maximum": 1}
    assert health.json()["capture_limits"]["image_file"]["maximum_pixels"] == 12_500_000
    assert "https://cadpro.example/static/og-single-image.png" in response.text
    assert "__CADPRO_ORIGIN__" not in response.text
    assert "/api/jobs/photos" not in response.text + script.text
    assert "/api/jobs/video" not in response.text + script.text

    collector = _DocumentCollector()
    collector.feed(response.text)
    assert len(collector.ids) == len(set(collector.ids))
    assert {"file-input", "width-mm", "depth-mm", "build-button", "result-section"} <= set(
        collector.ids
    )
    tag, file_input = collector.elements["file-input"]
    assert tag == "input"
    assert file_input["type"] == "file"
    assert "multiple" not in file_input
    assert "video" not in file_input["accept"]

    assert script.status_code == 200
    assert social_card.status_code == 200
    assert retired_card.status_code == 404
    assert social_card.headers["content-type"].startswith("image/png")
    card = cv2.imdecode(np.frombuffer(social_card.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert card is not None
    assert card.shape[:2] == (1024, 1536)


def test_frontend_enforces_one_bounded_image_and_one_bounded_thumbnail():
    assets = Path(__file__).parents[1] / "src" / "cadpro" / "web_assets"
    script = (assets / "app.js").read_text(encoding="utf-8")
    document = (assets / "index.html").read_text(encoding="utf-8")

    assert "MAX_IMAGE_BYTES = 25 * MEBIBYTE" in script
    assert "THUMBNAIL_MAX_EDGE = 320" in script
    assert "MAX_IMAGE_EDGE = 8_192" in script
    assert "MAX_IMAGE_PIXELS = 12_500_000" in script
    assert "incoming.length !== 1" in script
    assert "file.size > MAX_IMAGE_BYTES" in script
    assert "state.file = file" in script
    assert "state.files" not in script
    assert "syntheticProgress" not in script
    assert "new AbortController()" in script
    assert 'canvas.toDataURL("image/jpeg", 0.78)' in script
    assert 'form.append("file", state.file, state.file.name)' in script
    assert 'fetch("/api/jobs/image"' in script
    assert "multiple" not in document
    assert 'id="depth-mm"' in document


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

import time
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from cad_diff.step_io import load_step
from cadpro.web import create_app


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])


def _photo_payload() -> bytes:
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
        cv2.rectangle(frame, (80 - width // 2, 20), (80 + width // 2, 100), (10, 10, 10), -1)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_frontend_has_unique_interaction_targets_and_social_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("CADPRO_PUBLIC_ORIGIN", "https://cadpro.example")
    app = create_app(storage_parent=tmp_path)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.get("/")
        health = client.get("/api/health")

    assert response.status_code == 200
    assert health.json()["version"] == "1.0.0"
    assert "https://cadpro.example/static/og.png" in response.text
    assert "__CADPRO_ORIGIN__" not in response.text
    collector = _IdCollector()
    collector.feed(response.text)
    assert len(collector.ids) == len(set(collector.ids))
    assert {"file-input", "build-button", "progress-panel", "result-section"} <= set(collector.ids)


def test_frontend_bounds_uploads_and_sequential_photo_preview_memory():
    script = (
        Path(__file__).parents[1] / "src" / "cadpro" / "web_assets" / "app.js"
    ).read_text(encoding="utf-8")

    assert "MAX_IMAGE_BYTES = 25 * MEBIBYTE" in script
    assert "MAX_PHOTO_SET_BYTES = 500 * MEBIBYTE" in script
    assert "MAX_VIDEO_BYTES = 2 * GIBIBYTE" in script
    assert "THUMBNAIL_MAX_EDGE = 192" in script
    assert "file.size > MAX_IMAGE_BYTES" in script
    assert "totalBytes > MAX_PHOTO_SET_BYTES" in script
    assert "incoming[0].size > MAX_VIDEO_BYTES" in script
    assert "selectionGeneration" in script
    assert "new AbortController()" in script
    assert "state.files = [...incoming]" in script
    assert "for (let index = 0; index < files.length; index += 1)" in script
    assert "await inspectPhoto(files[index], signal)" in script
    assert "state.thumbnails.set(files[index], result.thumbnail)" in script
    assert 'canvas.toDataURL("image/jpeg", 0.72)' in script
    assert "Promise.all(state.files.map" not in script
    assert "state.previewUrls" not in script


def test_real_twenty_photo_job_exports_every_format(tmp_path):
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)
    payload = _photo_payload()
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
        status_url = accepted.json()["status_url"]

        deadline = time.monotonic() + 20
        snapshot = accepted.json()
        while snapshot["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.05)
            snapshot = client.get(status_url).json()

        assert snapshot["status"] == "completed", snapshot.get("error")
        assert snapshot["progress"] == 100
        assert snapshot["result"]["metrics"]["is_valid"] is True
        artifacts = snapshot["result"]["artifacts"]
        assert {artifact["filename"].split(".")[-1] for artifact in artifacts} >= {
            "step",
            "stl",
            "glb",
            "html",
            "json",
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


def test_real_turntable_video_job_exports_reloadable_step(tmp_path):
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)
    payload = _turntable_video_payload(tmp_path / "orbit.avi")

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/video",
            files={"file": ("orbit.avi", payload, "video/x-msvideo")},
            data={"width_mm": "55", "views": "20", "clockwise": "false"},
        )
        assert accepted.status_code == 202, accepted.text

        deadline = time.monotonic() + 20
        snapshot = accepted.json()
        while snapshot["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.05)
            snapshot = client.get(snapshot["status_url"]).json()

        assert snapshot["status"] == "completed", snapshot.get("error")
        assert snapshot["input_count"] == 1
        assert len(snapshot["result"]["input_diagnostics"]) == 20
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

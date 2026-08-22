from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from cad_diff.step_io import load_step
from cadpro.ml_mesh import ConceptMeshResult
from cadpro.web import RESEARCH_FRAME_MAX_EDGE, _representative_image_paths, create_app


def _image_payload() -> bytes:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 20), (125, 100), (10, 10, 10), -1)
    cv2.circle(image, (80, 60), 12, (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _wait(client: TestClient, status_url: str) -> dict:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        payload = client.get(status_url).json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("optional concept-mesh job did not finish")


def test_concept_mesh_is_supplemental_and_step_remains_valid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CADPRO_ML_MESH_ENABLED", "1")
    monkeypatch.setenv("CADPRO_ML_MESH_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("CADPRO_ML_MESH_ENDPOINT", "http://127.0.0.1:8099/generate")
    seen = []

    def fake_generate(representative_image, output_dir, *, config):
        seen.append((Path(representative_image), config))
        output = Path(output_dir) / "cadpro-ai-concept.glb"
        output.write_bytes((Path(output_dir) / "cadpro-model.glb").read_bytes())
        return ConceptMeshResult(
            status="completed",
            glb_path=output,
            metadata={
                "provider": "fake-hunyuan-worker",
                "artifact_kind": "ai_concept_mesh",
                "format": "glb",
                "metric_scale": False,
                "manufacturing_cad": False,
                "derived_from_step": False,
            },
        )

    monkeypatch.setattr("cadpro.ml_mesh.generate_concept_mesh", fake_generate)
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/image",
            files={"file": ("bracket.png", _image_payload(), "image/png")},
            data={
                "width_mm": "45",
                "depth_mm": "7.5",
                "concept_mesh": "true",
            },
        )
        assert accepted.status_code == 202, accepted.text
        completed = _wait(client, accepted.json()["status_url"])
        assert completed["status"] == "completed", completed.get("error")

        result = completed["result"]
        filenames = {artifact["filename"] for artifact in result["artifacts"]}
        assert "cadpro-model.step" in filenames
        assert "cadpro-model.glb" in filenames
        assert "cadpro-ai-concept.glb" in filenames
        concept = result["concept_mesh"]
        assert concept["status"] == "completed"
        assert concept["metric_scale"] is False
        assert concept["manufacturing_cad"] is False
        assert concept["derived_from_step"] is False
        assert concept["input_strategy"] == "representative_view"

        step_artifact = next(
            item for item in result["artifacts"] if item["filename"] == "cadpro-model.step"
        )
        step_path = tmp_path / "roundtrip.step"
        step_path.write_bytes(client.get(step_artifact["download_url"]).content)
        assert len(load_step(step_path)) == 1

        report_artifact = next(
            item
            for item in result["artifacts"]
            if item["filename"] == "cadpro-model.report.json"
        )
        report = json.loads(client.get(report_artifact["download_url"]).content)
        assert report["concept_mesh"]["manufacturing_cad"] is False
        assert report["artifacts"]["concept_mesh"]["file"] == "cadpro-ai-concept.glb"

    assert len(seen) == 1
    assert seen[0][0].name == "image.png"
    assert seen[0][1].available is True


def test_video_research_frames_are_checked_and_downscaled_before_writing(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "turntable.mp4"
    source.write_bytes(b"placeholder")
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    class FakeCapture:
        released = False
        position = 0

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 40
            if property_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 1_600
            if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 800
            return 0

        def set(self, property_id, value):
            self.position = int(value)
            return True

        def read(self):
            frame = np.full((800, 1_600, 3), self.position % 255, dtype=np.uint8)
            return True, frame

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    job = SimpleNamespace(
        kind="video",
        input_paths=(source,),
        input_dir=input_dir,
        parameters={"start_frame": 0, "end_frame": None},
    )

    paths = _representative_image_paths(job)

    assert len(paths) == 6
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert frame is not None
        assert max(frame.shape[:2]) == RESEARCH_FRAME_MAX_EDGE
    assert capture.released

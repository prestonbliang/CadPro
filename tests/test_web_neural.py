from __future__ import annotations

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from cad_diff.step_io import load_step
from cadpro.neural import HIDDEN_ONE, HIDDEN_TWO, INPUT_FEATURES, NeuralDepthModel
from cadpro.web import create_app


def _image_payload() -> bytes:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 22), (125, 98), (10, 10, 10), -1)
    cv2.circle(image, (80, 60), 12, (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _checkpoint(path: Path, ratio: float = 0.2) -> Path:
    normalized = (math.log(ratio) - math.log(0.01)) / (math.log(4.0) - math.log(0.01))
    logit = math.log(normalized / (1.0 - normalized))
    model = NeuralDepthModel(
        w1=np.zeros((INPUT_FEATURES, HIDDEN_ONE), dtype=np.float32),
        b1=np.zeros(HIDDEN_ONE, dtype=np.float32),
        w2=np.zeros((HIDDEN_ONE, HIDDEN_TWO), dtype=np.float32),
        b2=np.zeros(HIDDEN_TWO, dtype=np.float32),
        w3=np.zeros((HIDDEN_TWO, 1), dtype=np.float32),
        b3=np.asarray([logit], dtype=np.float32),
        feature_mean=np.zeros(INPUT_FEATURES, dtype=np.float32),
        feature_scale=np.ones(INPUT_FEATURES, dtype=np.float32),
        trained_examples=40,
        validation_relative_mae=0.12,
    )
    return model.save(path)


def _wait(client: TestClient, status_url: str) -> dict:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        payload = client.get(status_url).json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("neural reconstruction did not finish")


def test_neural_prediction_changes_depth_and_is_recorded_in_result_and_report(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint(tmp_path / "depth-model.npz", ratio=0.2)
    monkeypatch.setenv("CADPRO_NEURAL_ENABLED", "1")
    monkeypatch.setenv("CADPRO_NEURAL_CHECKPOINT", str(checkpoint))
    app = create_app(storage_parent=tmp_path / "jobs", job_retention_seconds=60)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        health = client.get("/api/health")
        accepted = client.post(
            "/api/jobs/image",
            files={"file": ("bracket.png", _image_payload(), "image/png")},
            data={"width_mm": "50", "neural_predict": "true"},
        )
        assert accepted.status_code == 202, accepted.text
        completed = _wait(client, accepted.json()["status_url"])
        assert completed["status"] == "completed", completed.get("error")

        prediction = completed["result"]["neural_prediction"]
        assert prediction["status"] == "completed"
        assert prediction["predicted_depth_mm"] == 10
        assert prediction["measured_width_mm"] == 50
        assert prediction["manufacturing_verified"] is False
        assert completed["parameters"] == {"width_mm": 50.0, "neural_predict": True}

        step_artifact = next(
            item for item in completed["result"]["artifacts"] if item["filename"].endswith(".step")
        )
        downloaded = tmp_path / "download.step"
        downloaded.write_bytes(client.get(step_artifact["download_url"]).content)
        assert len(load_step(downloaded)) == 1

        report_artifact = next(
            item
            for item in completed["result"]["artifacts"]
            if item["filename"].endswith(".report.json")
        )
        report = json.loads(client.get(report_artifact["download_url"]).content)
        assert report["neural_prediction"] == prediction
        assert report["geometry"]["dimensions_mm"]["z"] == 10

    capability = health.json()["neural_prediction"]
    assert capability == {
        "available": True,
        "enabled": True,
        "checkpoint_valid": True,
        "model_type": "numpy_mlp_depth_regressor",
        "predicts": "depth_to_width_ratio",
        "changes_geometry": True,
        "requires_measured_width": True,
        "trained_examples": 40,
        "validation_examples": 0,
    }
    assert str(checkpoint) not in health.text


def test_unconfigured_neural_request_and_missing_manual_depth_are_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CADPRO_NEURAL_ENABLED", raising=False)
    monkeypatch.delenv("CADPRO_NEURAL_CHECKPOINT", raising=False)
    app = create_app(storage_parent=tmp_path / "jobs")

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        neural = client.post(
            "/api/jobs/image",
            files={"file": ("part.png", _image_payload(), "image/png")},
            data={"width_mm": "50", "neural_predict": "true"},
        )
        manual = client.post(
            "/api/jobs/image",
            files={"file": ("part.png", _image_payload(), "image/png")},
            data={"width_mm": "50"},
        )

    assert neural.status_code == 409
    assert neural.json()["error"]["code"] == "neural_model_unavailable"
    assert manual.status_code == 422
    assert manual.json()["error"]["code"] == "invalid_request"
    assert not app.state.job_manager.root.exists()


def test_invalid_configured_checkpoint_is_not_advertised(tmp_path, monkeypatch):
    checkpoint = tmp_path / "invalid.npz"
    checkpoint.write_bytes(b"not a checkpoint")
    monkeypatch.setenv("CADPRO_NEURAL_ENABLED", "1")
    monkeypatch.setenv("CADPRO_NEURAL_CHECKPOINT", str(checkpoint))
    app = create_app(storage_parent=tmp_path / "jobs")

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        capability = client.get("/api/health").json()["neural_prediction"]

    assert capability["available"] is False
    assert capability["enabled"] is True
    assert capability["checkpoint_valid"] is False
    assert str(checkpoint) not in json.dumps(capability)

import os
import json

import cv2
import numpy as np
from typer.testing import CliRunner
import uvicorn

from cadpro.cli import app


def _turntable_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (160, 120))
    assert writer.isOpened()
    for _ in range(8):
        frame = np.full((120, 160, 3), 255, dtype=np.uint8)
        cv2.rectangle(frame, (40, 20), (120, 100), (0, 0, 0), -1)
        writer.write(frame)
    writer.release()


def test_convert_command_writes_step(tmp_path):
    source = tmp_path / "plate.png"
    output = tmp_path / "plate.step"
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (140, 100), (0, 0, 0), -1)
    assert cv2.imwrite(str(source), image)

    result = CliRunner().invoke(
        app,
        ["convert", str(source), "--width-mm", "60", "--depth-mm", "3", "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "60 x 3 mm" in result.output


def test_web_help_describes_all_website_capture_modes():
    result = CliRunner().invoke(app, ["web", "--help"])

    assert result.exit_code == 0, result.output
    assert "photo, photo-orbit, and turntable-video reconstruction website" in result.output


def test_convert_command_reports_unsupported_media(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("not an image", encoding="utf-8")

    result = CliRunner().invoke(app, ["convert", str(source)])

    assert result.exit_code != 0
    assert "Unsupported media type" in result.output


def test_turntable_command_writes_visual_hull_step(tmp_path):
    source = tmp_path / "turntable.avi"
    output = tmp_path / "turntable.step"
    _turntable_video(source)

    result = CliRunner().invoke(
        app,
        ["turntable", str(source), "--width-mm", "40", "--views", "4", "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "4 turntable views" in result.output
    assert "frames 0, 2, 4, 6" in result.output


def test_web_command_uses_its_actual_origin_and_disables_forwarded_headers(monkeypatch):
    calls = []
    monkeypatch.delenv("CADPRO_PUBLIC_ORIGIN", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = CliRunner().invoke(
        app,
        ["web", "--host", "::1", "--port", "8765", "--no-open"],
    )

    assert result.exit_code == 0, result.output
    assert "http://[::1]:8765" in result.output
    assert os.environ["CADPRO_PUBLIC_ORIGIN"] == "http://[::1]:8765"
    assert calls == [
        (("cadpro.web:app",), {"host": "::1", "port": 8765, "log_level": "info", "proxy_headers": False})
    ]


def test_neural_training_and_prediction_commands_create_checkpoint_and_step(tmp_path):
    images = []
    records = []
    for index, object_width in enumerate((44, 52, 60, 68)):
        source = tmp_path / f"sample-{index}.png"
        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        left = (120 - object_width) // 2
        cv2.rectangle(image, (left, 20), (left + object_width, 80), (0, 0, 0), -1)
        assert cv2.imwrite(str(source), image)
        images.append(source)
        records.append(
            json.dumps(
                {
                    "image": source.name,
                    "width_mm": 100,
                    "depth_mm": 12 + index * 4,
                }
            )
        )
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text("\n".join(records), encoding="utf-8")
    checkpoint = tmp_path / "model.npz"

    trained = CliRunner().invoke(
        app,
        [
            "neural-train",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "5",
            "--validation-fraction",
            "0",
        ],
    )
    assert trained.exit_code == 0, trained.output
    assert checkpoint.is_file()
    assert "Trained" in trained.output

    output = tmp_path / "prediction.step"
    predicted = CliRunner().invoke(
        app,
        [
            "neural-predict",
            str(images[0]),
            "--checkpoint",
            str(checkpoint),
            "--width-mm",
            "100",
            "--output",
            str(output),
        ],
    )
    assert predicted.exit_code == 0, predicted.output
    assert output.is_file()
    assert "predicted depth" in predicted.output

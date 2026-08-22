import cv2
import numpy as np
from typer.testing import CliRunner

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

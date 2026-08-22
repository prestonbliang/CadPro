from pathlib import Path

import cv2
import numpy as np

from cad_diff.step_io import load_step
from cadpro.media import extract_silhouette
from cadpro.pipeline import convert_media


def _part_image(path: Path) -> None:
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (270, 190), (20, 20, 20), -1)
    cv2.circle(image, (160, 120), 30, (255, 255, 255), -1)
    assert cv2.imwrite(str(path), image)


def test_extracts_outer_contour_and_hole():
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (260, 170), (0, 0, 0), -1)
    cv2.circle(image, (150, 100), 25, (255, 255, 255), -1)
    silhouette = extract_silhouette(image)
    assert len(silhouette.outer) == 4
    assert len(silhouette.holes) == 1


def test_converts_image_to_loadable_step(tmp_path):
    source = tmp_path / "part.png"
    output = tmp_path / "part.step"
    _part_image(source)
    result = convert_media(source, output, width_mm=110, depth_mm=8)
    assert result.output == output
    assert result.holes == 1
    assert len(load_step(output)) == 1


def test_converts_video_using_a_selected_frame(tmp_path):
    source = tmp_path / "part.avi"
    output = tmp_path / "part.step"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5, (320, 240))
    assert writer.isOpened()
    for offset in range(5):
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.rectangle(frame, (50 + offset, 50), (270 + offset, 190), (20, 20, 20), -1)
        writer.write(frame)
    writer.release()

    result = convert_media(source, output, width_mm=110, depth_mm=8)

    assert result.selected_frame is not None
    assert len(load_step(output)) == 1

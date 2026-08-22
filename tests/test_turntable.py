from pathlib import Path

import cv2
import numpy as np
import pytest

from cad_diff.signatures import fingerprint_solid
from cad_diff.step_io import load_step
from cadpro import media as media_module
from cadpro.media import Silhouette, silhouettes_from_turntable_video
from cadpro.pipeline import convert_turntable_video
from cadpro.step import visual_hull_from_silhouettes, write_step


def _rectangle_silhouette(width: float = 120, height: float = 60, frame: int = 0) -> Silhouette:
    center = np.asarray((100.0, 100.0))
    half = np.asarray((width / 2.0, height / 2.0))
    minimum = center - half
    maximum = center + half
    outer = np.asarray(
        [
            [minimum[0], minimum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
            [minimum[0], maximum[1]],
        ]
    )
    return Silhouette(outer=outer, holes=(), source_size=(201, 201), frame_index=frame)


def _turntable_video(path: Path, frame_count: int = 16, cropped: bool = False) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (320, 240))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        width = 120 - round(25 * abs(np.sin(2 * np.pi * index / frame_count)))
        left = 0 if cropped else 160 - width // 2
        cv2.rectangle(frame, (left, 60), (left + width, 180), (15, 15, 15), -1)
        writer.write(frame)
    writer.release()


def test_visual_hull_is_valid_scaled_cuboid():
    silhouettes = tuple(_rectangle_silhouette(frame=index) for index in range(4))

    shape = visual_hull_from_silhouettes(silhouettes, width_mm=100)
    fingerprint = fingerprint_solid("hull", shape)

    spans = tuple(
        maximum - minimum
        for minimum, maximum in zip(fingerprint.bbox_min, fingerprint.bbox_max)
    )
    assert spans == pytest.approx((100, 100, 50), abs=1e-6)
    assert fingerprint.volume == pytest.approx(500_000, rel=1e-8)


def test_visual_hull_preserves_visible_holes_through_step_round_trip(tmp_path):
    outer = _rectangle_silhouette()
    hole = np.asarray([[85.0, 90.0], [115.0, 90.0], [115.0, 110.0], [85.0, 110.0]])
    silhouettes = tuple(
        Silhouette(outer.outer, (hole,), outer.source_size, frame_index=index)
        for index in range(4)
    )

    shape = visual_hull_from_silhouettes(silhouettes, width_mm=100)

    original_volume = fingerprint_solid("perforated", shape).volume
    assert original_volume < 500_000
    output = write_step(shape, tmp_path / "perforated.step")
    loaded_shape = load_step(output)[0][1]
    assert fingerprint_solid("loaded", loaded_shape).volume == pytest.approx(original_volume)


def test_rotation_direction_mirrors_asymmetric_capture():
    silhouettes = []
    for index, center_x in enumerate((110.0, 120.0, 90.0, 80.0)):
        outer = np.asarray(
            [
                [center_x - 30, 75.0],
                [center_x + 30, 75.0],
                [center_x + 30, 125.0],
                [center_x - 30, 125.0],
            ]
        )
        silhouettes.append(Silhouette(outer, (), (201, 201), frame_index=index))

    counterclockwise = fingerprint_solid(
        "ccw", visual_hull_from_silhouettes(tuple(silhouettes), 60, clockwise=False)
    )
    clockwise = fingerprint_solid(
        "cw", visual_hull_from_silhouettes(tuple(silhouettes), 60, clockwise=True)
    )

    assert counterclockwise.center_of_mass[0] == pytest.approx(clockwise.center_of_mass[0])
    assert counterclockwise.center_of_mass[1] == pytest.approx(-clockwise.center_of_mass[1])


def test_finite_viewing_prisms_do_not_clip_off_axis_object():
    silhouettes = []
    for index, center_x in enumerate((300.0, 200.0, 100.0, 200.0)):
        outer = np.asarray(
            [
                [center_x - 10, 180.0],
                [center_x + 10, 180.0],
                [center_x + 10, 220.0],
                [center_x - 10, 220.0],
            ]
        )
        silhouettes.append(Silhouette(outer, (), (401, 401), frame_index=index))

    fingerprint = fingerprint_solid(
        "off-axis", visual_hull_from_silhouettes(tuple(silhouettes), width_mm=20)
    )

    assert fingerprint.center_of_mass == pytest.approx((100, 0, 0), abs=1e-8)
    assert fingerprint.volume == pytest.approx(16_000, rel=1e-8)


def test_turntable_sampler_uses_unique_evenly_spaced_frames(tmp_path):
    source = tmp_path / "turntable.avi"
    _turntable_video(source)

    silhouettes = silhouettes_from_turntable_video(source, view_count=4)

    assert tuple(item.frame_index for item in silhouettes) == (0, 4, 8, 12)


def test_turntable_sampler_rejects_invalid_frame_range(tmp_path):
    source = tmp_path / "turntable.avi"
    _turntable_video(source, frame_count=8)

    with pytest.raises(ValueError, match="Frame range"):
        silhouettes_from_turntable_video(source, view_count=4, start_frame=3, end_frame=2)
    with pytest.raises(ValueError, match="shorter than the requested view count"):
        silhouettes_from_turntable_video(source, view_count=4, start_frame=0, end_frame=3)


def test_turntable_capture_is_released_when_segmentation_fails(tmp_path, monkeypatch):
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"placeholder")

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, property_id):
            return 8

        def set(self, property_id, value):
            return True

        def read(self):
            return True, np.full((100, 100, 3), 255, dtype=np.uint8)

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(media_module.cv2, "VideoCapture", lambda path: capture)

    with pytest.raises(ValueError, match="Could not extract the object"):
        silhouettes_from_turntable_video(source, view_count=4)

    assert capture.released


def test_turntable_video_converts_to_loadable_step(tmp_path):
    source = tmp_path / "turntable.avi"
    output = tmp_path / "visual-hull.step"
    _turntable_video(source)

    result = convert_turntable_video(source, output, width_mm=80, views=4)

    assert result.sampled_frames == (0, 4, 8, 12)
    assert output.exists()
    assert len(load_step(output)) == 1


def test_turntable_rejects_cropped_object(tmp_path):
    source = tmp_path / "cropped.avi"
    output = tmp_path / "must-not-exist.step"
    _turntable_video(source, cropped=True)

    with pytest.raises(ValueError, match="touches the image border"):
        convert_turntable_video(source, output, width_mm=80, views=4)

    assert not output.exists()


def test_turntable_rejects_non_video_without_replacing_output(tmp_path):
    source = tmp_path / "part.png"
    output = tmp_path / "existing.step"
    assert cv2.imwrite(str(source), np.full((100, 100, 3), 255, dtype=np.uint8))
    output.write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="requires a supported video"):
        convert_turntable_video(source, output, width_mm=80, views=4)

    assert output.read_text(encoding="utf-8") == "keep me"

from pathlib import Path

import cv2
import numpy as np
import pytest

from cadpro import reconstruct as reconstruct_module


def _write_view(path: Path, *, size: tuple[int, int] = (160, 120), border: bool = False) -> None:
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    left = 0 if border else width // 4
    cv2.rectangle(
        image,
        (left, height // 4),
        (left + width // 2, height * 3 // 4),
        (0, 0, 0),
        -1,
    )
    assert cv2.imwrite(str(path), image)


def _photo_set(directory: Path, count: int = 20) -> list[Path]:
    paths = []
    for index in range(count):
        path = directory / f"view-{index:02d}.png"
        _write_view(path)
        paths.append(path)
    return paths


def test_photo_set_preserves_order_and_builds_diagnostics(tmp_path, monkeypatch):
    paths = _photo_set(tmp_path)
    sentinel = object()
    call = {}

    def fake_hull(silhouettes, width_mm, clockwise):
        call.update(silhouettes=silhouettes, width_mm=width_mm, clockwise=clockwise)
        return sentinel

    monkeypatch.setattr(reconstruct_module, "visual_hull_from_silhouettes", fake_hull)

    result = reconstruct_module.reconstruct_photo_set(paths, 80.0, clockwise=True)

    assert result.shape is sentinel
    assert result.mode == "photos"
    assert result.source_names == tuple(path.name for path in paths)
    assert len(result.silhouettes) == 20
    assert call == {
        "silhouettes": result.silhouettes,
        "width_mm": 80.0,
        "clockwise": True,
    }
    assert [item.order for item in result.input_diagnostics] == list(range(20))
    assert all(0 < item.foreground_fraction < 1 for item in result.input_diagnostics)


@pytest.mark.parametrize("count", [19, 51])
def test_photo_set_requires_twenty_to_fifty_views(tmp_path, count):
    with pytest.raises(ValueError, match="between 20 and 50"):
        reconstruct_module.reconstruct_photo_set(
            [tmp_path / f"missing-{index}.png" for index in range(count)],
            80,
        )


def test_photo_set_rejects_wrong_type_before_decode(tmp_path):
    paths = _photo_set(tmp_path)
    paths[7] = tmp_path / "view.txt"
    paths[7].write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported type"):
        reconstruct_module.reconstruct_photo_set(paths, 80)


def test_photo_set_rejects_mixed_frame_dimensions(tmp_path):
    paths = _photo_set(tmp_path)
    _write_view(paths[4], size=(180, 120))

    with pytest.raises(ValueError, match="same frame dimensions"):
        reconstruct_module.reconstruct_photo_set(paths, 80)


def test_photo_set_rejects_border_touching_object(tmp_path):
    paths = _photo_set(tmp_path)
    _write_view(paths[12], border=True)

    with pytest.raises(ValueError, match="touches the image border"):
        reconstruct_module.reconstruct_photo_set(paths, 80)


def test_video_wrapper_samples_twenty_evenly_spaced_views_and_releases(tmp_path, monkeypatch):
    source = tmp_path / "turntable.mp4"
    source.write_bytes(b"placeholder")
    sentinel = object()

    class FakeCapture:
        released = False
        position = 0

        def isOpened(self):
            return True

        def get(self, property_id):
            return 40

        def set(self, property_id, value):
            self.position = int(value)
            return True

        def read(self):
            frame = np.full((120, 160, 3), 255, dtype=np.uint8)
            cv2.rectangle(frame, (40, 30), (120, 90), (0, 0, 0), -1)
            return True, frame

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(reconstruct_module.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(
        reconstruct_module,
        "visual_hull_from_silhouettes",
        lambda silhouettes, width_mm, clockwise: sentinel,
    )

    result = reconstruct_module.reconstruct_turntable_video(source, 75, views=20)

    assert result.shape is sentinel
    assert result.mode == "video"
    assert tuple(item.frame_index for item in result.silhouettes) == tuple(range(0, 40, 2))
    assert result.source_names[0] == "turntable.mp4#frame=0"
    assert result.source_names[-1] == "turntable.mp4#frame=38"
    assert capture.released


def test_video_wrapper_rejects_short_range_before_reading(tmp_path, monkeypatch):
    source = tmp_path / "turntable.mp4"
    source.write_bytes(b"placeholder")

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, property_id):
            return 100

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(reconstruct_module.cv2, "VideoCapture", lambda path: capture)

    with pytest.raises(ValueError, match="shorter than the requested view count"):
        reconstruct_module.reconstruct_turntable_video(
            source,
            75,
            views=20,
            start_frame=10,
            end_frame=25,
        )

    assert capture.released

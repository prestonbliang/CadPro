from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
from OCP.TopoDS import TopoDS_Shape

from cadpro.media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, Silhouette, extract_silhouette
from cadpro.step import visual_hull_from_silhouettes


MIN_CAPTURE_VIEWS = 20
MAX_CAPTURE_VIEWS = 50
_BORDER_MARGIN_PX = 2


@dataclass(frozen=True)
class InputDiagnostic:
    """Segmentation facts for one ordered view used by a reconstruction."""

    order: int
    source_name: str
    source_size: tuple[int, int]
    frame_index: int | None
    outline_points: int
    hole_count: int
    foreground_fraction: float


@dataclass(frozen=True)
class Reconstruction:
    """A reconstructed CAD body plus the inputs needed for audit/reporting."""

    shape: TopoDS_Shape
    silhouettes: tuple[Silhouette, ...]
    mode: Literal["photos", "video"]
    source_names: tuple[str, ...]

    @property
    def input_diagnostics(self) -> tuple[InputDiagnostic, ...]:
        return tuple(
            _diagnostic(order, source_name, silhouette)
            for order, (source_name, silhouette) in enumerate(
                zip(self.source_names, self.silhouettes, strict=True)
            )
        )


def reconstruct_photo_set(
    paths: Sequence[str | Path],
    width_mm: float,
    clockwise: bool = False,
) -> Reconstruction:
    """Reconstruct one visual hull from 20--50 ordered, evenly spaced views.

    The order is the calibration: item ``n`` is interpreted as an angle of
    ``n * 360 / len(paths)`` degrees around one complete revolution.
    """
    _validate_width(width_mm)
    if isinstance(paths, (str, Path)):
        raise ValueError("paths must be an ordered collection of image files")
    ordered_paths = tuple(Path(path) for path in paths)
    _validate_view_count(len(ordered_paths), label="Photo set")

    silhouettes: list[Silhouette] = []
    source_names: list[str] = []
    expected_size: tuple[int, int] | None = None
    for order, source in enumerate(ordered_paths):
        if not source.is_file():
            raise FileNotFoundError(f"Photo {order + 1} does not exist: {source}")
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            supported = ", ".join(sorted(IMAGE_EXTENSIONS))
            raise ValueError(
                f"Photo {order + 1} has unsupported type '{source.suffix.lower()}'; "
                f"supported image types: {supported}"
            )
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV could not decode photo {order + 1}: {source}")
        try:
            silhouette = extract_silhouette(image)
        except ValueError as error:
            raise ValueError(
                f"Could not extract the object from photo {order + 1} ({source.name}): {error}"
            ) from error
        expected_size = _validate_silhouette(
            silhouette,
            expected_size=expected_size,
            source_label=f"photo {order + 1} ({source.name})",
        )
        silhouettes.append(silhouette)
        source_names.append(source.name)

    silhouette_tuple = tuple(silhouettes)
    shape = visual_hull_from_silhouettes(
        silhouette_tuple,
        width_mm=width_mm,
        clockwise=clockwise,
    )
    return Reconstruction(
        shape=shape,
        silhouettes=silhouette_tuple,
        mode="photos",
        source_names=tuple(source_names),
    )


def reconstruct_turntable_video(
    path: str | Path,
    width_mm: float,
    views: int = 24,
    start_frame: int = 0,
    end_frame: int | None = None,
    clockwise: bool = False,
) -> Reconstruction:
    """Sample 20--50 evenly spaced video frames and reconstruct a visual hull."""
    _validate_width(width_mm)
    _validate_view_count(views, label="Video")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Video does not exist: {source}")
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported video type '{source.suffix.lower()}'; supported video types: {supported}"
        )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not decode video: {source}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError("Video does not report a frame count; turntable sampling is unavailable")
        if isinstance(start_frame, bool) or not isinstance(start_frame, int):
            raise ValueError("start_frame must be an integer")
        if end_frame is not None and (
            isinstance(end_frame, bool) or not isinstance(end_frame, int)
        ):
            raise ValueError("end_frame must be an integer or None")
        stop = frame_count if end_frame is None else end_frame
        if start_frame < 0 or stop > frame_count or stop <= start_frame:
            raise ValueError(f"Frame range must satisfy 0 <= start < end <= {frame_count}")
        if stop - start_frame < views:
            raise ValueError("Selected frame range is shorter than the requested view count")

        indices = np.linspace(start_frame, stop, views, endpoint=False, dtype=int)
        silhouettes: list[Silhouette] = []
        source_names: list[str] = []
        expected_size: tuple[int, int] | None = None
        for order, raw_index in enumerate(indices):
            frame_index = int(raw_index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Could not decode sampled video frame {frame_index}")
            try:
                silhouette = extract_silhouette(frame, frame_index=frame_index)
            except ValueError as error:
                raise ValueError(
                    f"Could not extract the object at video frame {frame_index}: {error}"
                ) from error
            expected_size = _validate_silhouette(
                silhouette,
                expected_size=expected_size,
                source_label=f"video frame {frame_index}",
            )
            silhouettes.append(silhouette)
            source_names.append(f"{source.name}#frame={frame_index}")
    finally:
        capture.release()

    silhouette_tuple = tuple(silhouettes)
    shape = visual_hull_from_silhouettes(
        silhouette_tuple,
        width_mm=width_mm,
        clockwise=clockwise,
    )
    return Reconstruction(
        shape=shape,
        silhouettes=silhouette_tuple,
        mode="video",
        source_names=tuple(source_names),
    )


def _validate_width(width_mm: float) -> None:
    if isinstance(width_mm, bool) or not isinstance(width_mm, (int, float)):
        raise ValueError("width_mm must be a finite positive number")
    if not math.isfinite(float(width_mm)) or width_mm <= 0:
        raise ValueError("width_mm must be a finite positive number")


def _validate_view_count(view_count: int, *, label: str) -> None:
    if isinstance(view_count, bool) or not isinstance(view_count, int):
        raise ValueError(f"{label} view count must be an integer")
    if not MIN_CAPTURE_VIEWS <= view_count <= MAX_CAPTURE_VIEWS:
        raise ValueError(
            f"{label} view count must be between {MIN_CAPTURE_VIEWS} and {MAX_CAPTURE_VIEWS}"
        )


def _validate_silhouette(
    silhouette: Silhouette,
    *,
    expected_size: tuple[int, int] | None,
    source_label: str,
) -> tuple[int, int]:
    if expected_size is not None and silhouette.source_size != expected_size:
        raise ValueError(
            f"All views must have the same frame dimensions; {source_label} is "
            f"{silhouette.source_size[0]} x {silhouette.source_size[1]}, expected "
            f"{expected_size[0]} x {expected_size[1]}"
        )
    if _touches_border(silhouette):
        raise ValueError(
            f"Object touches the image border in {source_label}; keep the full object in view"
        )
    return silhouette.source_size


def _touches_border(silhouette: Silhouette) -> bool:
    width, height = silhouette.source_size
    minimum = silhouette.outer.min(axis=0)
    maximum = silhouette.outer.max(axis=0)
    return bool(
        minimum[0] <= _BORDER_MARGIN_PX
        or minimum[1] <= _BORDER_MARGIN_PX
        or maximum[0] >= width - 1 - _BORDER_MARGIN_PX
        or maximum[1] >= height - 1 - _BORDER_MARGIN_PX
    )


def _diagnostic(order: int, source_name: str, silhouette: Silhouette) -> InputDiagnostic:
    width, height = silhouette.source_size
    foreground_area = abs(cv2.contourArea(silhouette.outer.astype(np.float32)))
    return InputDiagnostic(
        order=order,
        source_name=source_name,
        source_size=silhouette.source_size,
        frame_index=silhouette.frame_index,
        outline_points=len(silhouette.outer),
        hole_count=len(silhouette.holes),
        foreground_fraction=foreground_area / float(width * height),
    )

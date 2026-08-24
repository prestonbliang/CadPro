from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from cadpro.scan.models import QualityPreset
from cadpro.scan.process import CancellationToken, ProcessCancelled
from cadpro.scan.quality import (
    analyze_and_normalize_images,
    blur_score,
    settings_for,
)


def _textured_rgb(
    width: int = 320,
    height: int = 240,
    *,
    phase: int = 0,
) -> np.ndarray:
    """Create a deterministic, feature-rich target without checked-in fixtures."""

    y, x = np.indices((height, width))
    checker = ((x // 16 + y // 16 + phase) % 2).astype(np.uint8)
    red = np.where(checker, 238, 18)
    green = (x * 5 + y * 3 + phase * 29) % 256
    blue = np.where(((x // 9 + phase) % 3) == 0, 220, 35)
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _write_jpeg(
    path: Path,
    rgb: np.ndarray,
    *,
    exif_orientation: int | None = None,
) -> Path:
    image = Image.fromarray(rgb, mode="RGB")
    if exif_orientation is None:
        image.save(path, format="JPEG", quality=97, subsampling=0)
    else:
        exif = Image.Exif()
        exif[274] = exif_orientation
        image.save(path, format="JPEG", quality=97, subsampling=0, exif=exif)
    return path


def _analyze(path: Path, output: Path):
    return analyze_and_normalize_images(
        [path],
        output,
        preset=QualityPreset.BALANCED,
        cancellation=CancellationToken(),
    )


def test_blurred_image_is_rejected_while_sharp_image_passes(tmp_path):
    sharp_rgb = _textured_rgb()
    sharp_bgr = cv2.cvtColor(sharp_rgb, cv2.COLOR_RGB2BGR)
    blurred_bgr = cv2.GaussianBlur(sharp_bgr, (51, 51), sigmaX=14)
    threshold = settings_for(QualityPreset.BALANCED).minimum_blur_score

    assert blur_score(sharp_bgr) > threshold
    assert blur_score(blurred_bgr) < threshold

    sharp_path = _write_jpeg(tmp_path / "sharp.jpg", sharp_rgb)
    blurred_path = _write_jpeg(
        tmp_path / "blurred.jpg",
        cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB),
    )
    sharp_outputs, sharp_diagnostics = _analyze(sharp_path, tmp_path / "sharp-output")
    blurred_outputs, blurred_diagnostics = _analyze(
        blurred_path,
        tmp_path / "blurred-output",
    )

    assert len(sharp_outputs) == 1
    assert sharp_diagnostics[0].accepted is True
    assert blurred_outputs == ()
    assert blurred_diagnostics[0].accepted is False
    assert "motion_blur_or_defocus" in blurred_diagnostics[0].rejection_reasons


@pytest.mark.parametrize(
    ("value", "warning", "fraction_name"),
    [
        (0, "severely_underexposed", "shadow_fraction"),
        (255, "severely_overexposed", "highlight_fraction"),
    ],
)
def test_extreme_exposure_is_reported(value, warning, fraction_name, tmp_path):
    path = _write_jpeg(
        tmp_path / f"exposure-{value}.jpg",
        np.full((180, 240, 3), value, dtype=np.uint8),
    )

    _, diagnostics = _analyze(path, tmp_path / f"output-{value}")

    diagnostic = diagnostics[0]
    assert getattr(diagnostic, fraction_name) > 0.99
    assert warning in diagnostic.warnings


def test_exif_orientation_is_applied_before_dimensions_and_output(tmp_path):
    source = _write_jpeg(
        tmp_path / "camera-oriented.jpg",
        _textured_rgb(width=300, height=180),
        exif_orientation=6,
    )

    outputs, diagnostics = _analyze(source, tmp_path / "normalized")

    assert len(outputs) == 1
    assert diagnostics[0].accepted is True
    assert (diagnostics[0].width, diagnostics[0].height) == (180, 300)
    with Image.open(outputs[0]) as normalized:
        assert normalized.size == (180, 300)


def test_near_duplicate_is_rejected_without_publishing_second_image(tmp_path):
    pixels = _textured_rgb()
    first = _write_jpeg(tmp_path / "first.jpg", pixels)
    duplicate = _write_jpeg(tmp_path / "duplicate.jpg", pixels)
    output = tmp_path / "normalized"

    accepted, diagnostics = analyze_and_normalize_images(
        [first, duplicate],
        output,
        preset=QualityPreset.BALANCED,
        cancellation=CancellationToken(),
    )

    assert [path.name for path in accepted] == ["image-0001.jpg"]
    assert diagnostics[0].accepted is True
    assert diagnostics[1].accepted is False
    assert diagnostics[1].normalized_name is None
    assert "near_duplicate" in diagnostics[1].rejection_reasons
    assert sorted(path.name for path in output.glob("*.jpg")) == ["image-0001.jpg"]


def test_cancellation_stops_between_images(tmp_path):
    first = _write_jpeg(tmp_path / "first.jpg", _textured_rgb(phase=0))
    second = _write_jpeg(tmp_path / "second.jpg", _textured_rgb(phase=3))
    checks = 0

    def cancel_on_second_check() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    output = tmp_path / "normalized"
    with pytest.raises(ProcessCancelled, match="cancelled"):
        analyze_and_normalize_images(
            [first, second],
            output,
            preset=QualityPreset.BALANCED,
            cancellation=CancellationToken(probe=cancel_on_second_check),
        )

    assert checks == 2
    assert sorted(path.name for path in output.glob("*.jpg")) == ["image-0001.jpg"]

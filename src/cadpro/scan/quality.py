"""Bounded image normalization, blur/exposure checks, and duplicate rejection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from cadpro.scan.models import ImageQuality, QualityPreset
from cadpro.scan.process import CancellationToken


MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True)
class QualitySettings:
    maximum_edge: int
    minimum_blur_score: float
    minimum_features: int
    duplicate_hamming_distance: int


_SETTINGS = {
    QualityPreset.DRAFT: QualitySettings(1_600, 45.0, 30, 2),
    QualityPreset.BALANCED: QualitySettings(2_400, 65.0, 50, 3),
    QualityPreset.HIGH: QualitySettings(3_200, 80.0, 70, 3),
}


def settings_for(preset: QualityPreset) -> QualitySettings:
    return _SETTINGS[preset]


def blur_score(image: np.ndarray) -> float:
    gray = _gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def perceptual_hash(image: np.ndarray) -> int:
    """Return a 64-bit dHash suitable for near-duplicate screening."""

    gray = _gray(image)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    comparisons = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in comparisons.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return int((first ^ second).bit_count())


def image_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Bounded grayscale correlation used only as a frame-selection heuristic."""

    left = cv2.resize(_gray(first), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    right = cv2.resize(_gray(second), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    left -= float(left.mean())
    right -= float(right.mean())
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-9:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.clip(np.sum(left * right) / denominator, -1.0, 1.0) * 0.5 + 0.5)


def analyze_and_normalize_images(
    paths: Iterable[str | Path],
    output_directory: str | Path,
    *,
    preset: QualityPreset,
    cancellation: CancellationToken,
) -> tuple[tuple[Path, ...], tuple[ImageQuality, ...]]:
    """Inspect inputs one at a time and publish only accepted, normalized images."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    settings = settings_for(preset)
    accepted: list[Path] = []
    diagnostics: list[ImageQuality] = []
    accepted_hashes: list[int] = []
    for index, raw_path in enumerate(paths, start=1):
        cancellation.raise_if_cancelled()
        source = Path(raw_path)
        normalized_name = f"image-{index:04d}.jpg"
        normalized_path = output / normalized_name
        image, width, height = _load_oriented(source, settings.maximum_edge)
        score = blur_score(image)
        gray = _gray(image)
        shadow_fraction = float(np.mean(gray <= 12))
        highlight_fraction = float(np.mean(gray >= 243))
        detector = cv2.ORB_create(nfeatures=2_000)
        keypoints = detector.detect(gray, None)
        features = len(keypoints)
        fingerprint = perceptual_hash(image)
        reasons: list[str] = []
        warnings: list[str] = []
        if any(
            hamming_distance(fingerprint, prior) <= settings.duplicate_hamming_distance
            for prior in accepted_hashes
        ):
            reasons.append("near_duplicate")
        if score < settings.minimum_blur_score:
            reasons.append("motion_blur_or_defocus")
        if features < settings.minimum_features:
            warnings.append("few_trackable_features")
        if shadow_fraction > 0.72:
            warnings.append("severely_underexposed")
        if highlight_fraction > 0.72:
            warnings.append("severely_overexposed")
        is_accepted = not reasons
        if is_accepted:
            if not cv2.imwrite(
                str(normalized_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
            ):
                raise RuntimeError(f"Could not write normalized image {normalized_name}.")
            accepted.append(normalized_path)
            accepted_hashes.append(fingerprint)
        diagnostics.append(
            ImageQuality(
                source_name=source.name,
                normalized_name=normalized_name if is_accepted else None,
                width=width,
                height=height,
                blur_score=score,
                shadow_fraction=shadow_fraction,
                highlight_fraction=highlight_fraction,
                feature_count=features,
                perceptual_hash=f"{fingerprint:016x}",
                accepted=is_accepted,
                rejection_reasons=reasons,
                warnings=warnings,
            )
        )
    return tuple(accepted), tuple(diagnostics)


def _load_oriented(path: Path, maximum_edge: int) -> tuple[np.ndarray, int, int]:
    try:
        with Image.open(path) as opened:
            if opened.width <= 0 or opened.height <= 0:
                raise ValueError("Image dimensions must be positive.")
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel scan-pipeline limit."
                )
            oriented = ImageOps.exif_transpose(opened).convert("RGB")
            oriented.thumbnail((maximum_edge, maximum_edge), Image.Resampling.LANCZOS)
            rgb = np.asarray(oriented, dtype=np.uint8)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError(f"{path.name} is not a supported, readable image.") from error
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, int(bgr.shape[1]), int(bgr.shape[0])


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError("image must be grayscale, BGR, or BGRA")
    conversion = cv2.COLOR_BGRA2GRAY if array.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(array, conversion)

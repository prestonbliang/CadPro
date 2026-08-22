from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class Silhouette:
    outer: np.ndarray
    holes: tuple[np.ndarray, ...]
    source_size: tuple[int, int]
    frame_index: int | None = None


def silhouette_from_media(path: str | Path) -> Silhouette:
    """Load an image, or select the clearest usable frame from a video."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {source}")
        return extract_silhouette(image)
    if suffix in VIDEO_EXTENSIONS:
        return _best_video_silhouette(source)
    supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
    raise ValueError(f"Unsupported media type '{suffix}'. Supported: {supported}")


def extract_silhouette(image: np.ndarray, frame_index: int | None = None) -> Silhouette:
    """Extract the largest foreground region and its holes from an image."""
    if image.ndim not in (2, 3):
        raise ValueError("Expected a grayscale, RGB, or RGBA image")
    height, width = image.shape[:2]
    if min(height, width) < 8:
        raise ValueError("Image is too small; use at least 8 x 8 pixels")

    mask = _foreground_mask(image)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        raise ValueError("No foreground object was found")

    parents = hierarchy[0, :, 3]
    outer_indices = [i for i, parent in enumerate(parents) if parent == -1]
    outer_index = max(outer_indices, key=lambda i: abs(cv2.contourArea(contours[i])))
    image_area = float(width * height)
    outer_area = abs(cv2.contourArea(contours[outer_index]))
    if outer_area < image_area * 0.001:
        raise ValueError("Detected object is too small relative to the image")

    epsilon = max(0.75, cv2.arcLength(contours[outer_index], True) * 0.0015)
    outer = _simplify(contours[outer_index], epsilon)
    holes = []
    for i, contour in enumerate(contours):
        if parents[i] == outer_index and abs(cv2.contourArea(contour)) >= image_area * 0.0001:
            holes.append(_simplify(contour, max(0.75, cv2.arcLength(contour, True) * 0.0015)))
    return Silhouette(outer=outer, holes=tuple(holes), source_size=(width, height), frame_index=frame_index)


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        if int(alpha.max()) != int(alpha.min()):
            _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return _clean_mask(mask)
        image = image[:, :, :3]

    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3]
    border = np.concatenate((color[0], color[-1], color[:, 0], color[:, -1]), axis=0)
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(color.astype(np.float32) - background, axis=2)
    scaled = np.uint8(np.clip(distance, 0, 255))
    threshold, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if threshold < 3:
        mask = np.uint8(distance > 8) * 255
    return _clean_mask(mask)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    size = max(3, int(round(min(mask.shape) * 0.006)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _simplify(contour: np.ndarray, epsilon: float) -> np.ndarray:
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64)
    if len(points) < 3:
        raise ValueError("Object outline does not contain enough usable points")
    return points


def _best_video_silhouette(path: Path, sample_count: int = 24) -> Silhouette:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not decode video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(sample_count)) if frame_count <= 0 else sorted(
            set(np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=int))
        )
        candidates: list[tuple[float, Silhouette]] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            try:
                silhouette = extract_silhouette(frame, frame_index=int(index))
            except ValueError:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            coverage = abs(cv2.contourArea(silhouette.outer.astype(np.float32))) / float(frame.shape[0] * frame.shape[1])
            score = sharpness * min(coverage, 0.75) * min(1.0, (1.0 - coverage) / 0.1)
            candidates.append((score, silhouette))
        if not candidates:
            raise ValueError("No video frame contained a usable foreground silhouette")
        return max(candidates, key=lambda item: item[0])[1]
    finally:
        capture.release()

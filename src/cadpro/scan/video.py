"""FFprobe inspection, FFmpeg extraction, useful-frame selection, and contact sheets."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from cadpro.scan.capabilities import executable
from cadpro.scan.models import (
    QualityPreset,
    ToolchainCapabilities,
    VideoFrame,
    VideoMetadata,
    VideoSelectionSettings,
)
from cadpro.scan.process import (
    CancellationToken,
    ProcessResult,
    run_process,
    run_process_capture,
)
from cadpro.scan.quality import blur_score, image_similarity, settings_for


@dataclass(frozen=True)
class VideoSelectionResult:
    metadata: VideoMetadata
    selected_paths: tuple[Path, ...]
    frames: tuple[VideoFrame, ...]
    contact_sheet_path: Path
    commands: tuple[tuple[str, ...], ...]


def inspect_video(
    path: str | Path,
    *,
    capabilities: ToolchainCapabilities,
    settings: VideoSelectionSettings,
    working_directory: str | Path,
    log_path: str | Path,
    cancellation: CancellationToken,
) -> tuple[VideoMetadata, tuple[str, ...]]:
    ffprobe = executable(capabilities, "ffprobe")
    source = Path(path).resolve(strict=True)
    arguments = (
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=format_name,duration,size:stream=index,codec_name,codec_type,width,height,"
        "pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(source),
    )
    captured = run_process_capture(
        arguments,
        working_directory=working_directory,
        log_path=log_path,
        timeout_seconds=30,
        cancellation=cancellation,
        maximum_stdout_bytes=512 * 1024,
    )
    try:
        document = json.loads(captured.stdout)
        streams = document["streams"]
        stream = streams[0]
        format_info = document.get("format", {})
        width = int(stream["width"])
        height = int(stream["height"])
        duration = _first_positive_float(stream.get("duration"), format_info.get("duration"))
        frame_rate = _first_positive_rate(
            stream.get("avg_frame_rate"),
            stream.get("r_frame_rate"),
        )
        frame_count_value = stream.get("nb_frames")
        frame_count = int(frame_count_value) if str(frame_count_value).isdigit() else None
        size_bytes = int(format_info.get("size") or source.stat().st_size)
        codec = str(stream["codec_name"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("FFprobe did not return valid metadata for one video stream.") from error
    if width <= 0 or height <= 0 or width * height > 40_000_000:
        raise ValueError("Video frame dimensions are invalid or exceed the 40-megapixel limit.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration is missing or invalid.")
    if duration > settings.maximum_duration_seconds:
        raise ValueError(
            f"Video duration {duration:.1f}s exceeds the configured "
            f"{settings.maximum_duration_seconds:g}s limit."
        )
    if not math.isfinite(frame_rate) or not 0 < frame_rate <= 1_000:
        raise ValueError("Video frame rate is missing or invalid.")
    metadata = VideoMetadata(
        codec=codec,
        width=width,
        height=height,
        duration_seconds=duration,
        frame_rate=frame_rate,
        frame_count=frame_count,
        size_bytes=size_bytes,
    )
    return metadata, arguments


def extract_video_candidates(
    path: str | Path,
    output_directory: str | Path,
    *,
    capabilities: ToolchainCapabilities,
    settings: VideoSelectionSettings,
    working_directory: str | Path,
    log_path: str | Path,
    cancellation: CancellationToken,
    maximum_edge: int,
) -> tuple[tuple[Path, ...], ProcessResult]:
    ffmpeg = executable(capabilities, "ffmpeg")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pattern = output / "candidate-%06d.jpg"
    spacing = 1.0 / settings.candidate_frames_per_second
    filter_graph = (
        "select=isnan(prev_selected_t)+"
        f"gte(t-prev_selected_t\\,{spacing:g}),"
        f"scale={maximum_edge}:-2:force_original_aspect_ratio=decrease"
    )
    arguments = (
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(Path(path).resolve(strict=True)),
        "-map",
        "0:v:0",
        "-vf",
        filter_graph,
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(settings.maximum_candidate_frames),
        "-q:v",
        "2",
        "-n",
        str(pattern),
    )
    result = run_process(
        arguments,
        working_directory=working_directory,
        log_path=log_path,
        timeout_seconds=600,
        cancellation=cancellation,
    )
    candidates = tuple(sorted(output.glob("candidate-*.jpg")))
    if len(candidates) < min(8, settings.target_frames):
        raise RuntimeError(
            f"FFmpeg extracted only {len(candidates)} candidate frames; at least 8 are needed."
        )
    return candidates, result


def select_video_frames(
    candidates: Sequence[str | Path],
    output_directory: str | Path,
    *,
    settings: VideoSelectionSettings,
    preset: QualityPreset,
    cancellation: CancellationToken,
) -> tuple[tuple[Path, ...], tuple[VideoFrame, ...]]:
    if len(candidates) < 2:
        raise ValueError("Video frame selection needs at least two candidates.")
    paths = tuple(Path(candidate) for candidate in candidates)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    quality = settings_for(preset)
    target = min(settings.target_frames, len(paths))
    boundaries = np.linspace(0, len(paths), target + 1, dtype=int)
    selected_paths: list[Path] = []
    selected_frames: list[VideoFrame] = []
    previous_image: np.ndarray | None = None
    previous_time: float | None = None
    for bucket in range(target):
        cancellation.raise_if_cancelled()
        start, stop = int(boundaries[bucket]), int(boundaries[bucket + 1])
        if stop <= start:
            continue
        ranked: list[tuple[float, int, np.ndarray]] = []
        for index in range(start, stop):
            image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            ranked.append((blur_score(image), index, image))
        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen: tuple[float, int, np.ndarray, float | None, float | None] | None = None
        for score, index, image in ranked:
            timestamp = index / settings.candidate_frames_per_second
            if score < quality.minimum_blur_score:
                continue
            if previous_time is not None and timestamp - previous_time < settings.minimum_spacing_seconds:
                continue
            similarity = image_similarity(previous_image, image) if previous_image is not None else None
            viewpoint = _viewpoint_change(previous_image, image) if previous_image is not None else None
            if similarity is not None and similarity > settings.maximum_similarity:
                continue
            if viewpoint is not None and viewpoint < settings.minimum_viewpoint_change:
                continue
            chosen = (score, index, image, similarity, viewpoint)
            break
        if chosen is None:
            continue
        score, index, image, similarity, viewpoint = chosen
        destination = output / f"frame-{len(selected_paths) + 1:04d}.jpg"
        if not cv2.imwrite(str(destination), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
            raise RuntimeError(f"Could not publish selected frame {destination.name}.")
        timestamp = index / settings.candidate_frames_per_second
        selected_paths.append(destination)
        selected_frames.append(
            VideoFrame(
                source_index=index,
                timestamp_seconds=timestamp,
                filename=destination.name,
                blur_score=score,
                similarity_to_previous=similarity,
                viewpoint_change=viewpoint,
            )
        )
        previous_image = image
        previous_time = timestamp
    if len(selected_paths) < 8:
        raise RuntimeError(
            "Fewer than 8 useful video views survived blur, similarity, spacing, and viewpoint "
            "checks. Capture a slower orbit with steadier focus and more visual texture."
        )
    return tuple(selected_paths), tuple(selected_frames)


def create_contact_sheet(
    paths: Sequence[str | Path],
    frames: Sequence[VideoFrame],
    destination: str | Path,
    *,
    columns: int = 5,
) -> Path:
    if len(paths) != len(frames) or not paths:
        raise ValueError("Contact-sheet paths and frame records must be non-empty and aligned.")
    if columns <= 0:
        raise ValueError("columns must be positive")
    thumb_width, thumb_height, label_height = 220, 150, 28
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#10151c")
    draw = ImageDraw.Draw(sheet)
    for index, (raw_path, frame) in enumerate(zip(paths, frames, strict=True)):
        with Image.open(raw_path) as image:
            thumbnail = ImageOps.fit(
                image.convert("RGB"),
                (thumb_width, thumb_height),
                method=Image.Resampling.LANCZOS,
            )
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(thumbnail, (x, y))
        draw.text(
            (x + 8, y + thumb_height + 6),
            f"#{frame.source_index}  {frame.timestamp_seconds:.2f}s  blur {frame.blur_score:.0f}",
            fill="#e8eef6",
        )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=88, optimize=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Contact-sheet exporter produced no output.")
    return path


def prepare_video_frames(
    video_path: str | Path,
    working_directory: str | Path,
    *,
    capabilities: ToolchainCapabilities,
    settings: VideoSelectionSettings,
    preset: QualityPreset,
    cancellation: CancellationToken,
) -> VideoSelectionResult:
    work = Path(working_directory)
    log = work / "video-tools.log"
    metadata, probe_command = inspect_video(
        video_path,
        capabilities=capabilities,
        settings=settings,
        working_directory=work,
        log_path=log,
        cancellation=cancellation,
    )
    maximum_edge = settings_for(preset).maximum_edge
    candidates, extraction = extract_video_candidates(
        video_path,
        work / "candidate-frames",
        capabilities=capabilities,
        settings=settings,
        working_directory=work,
        log_path=log,
        cancellation=cancellation,
        maximum_edge=maximum_edge,
    )
    selected, frame_records = select_video_frames(
        candidates,
        work / "selected-frames",
        settings=settings,
        preset=preset,
        cancellation=cancellation,
    )
    contact_sheet = create_contact_sheet(
        selected,
        frame_records,
        work / "selected-frames-contact-sheet.jpg",
    )
    # Candidate frames are intermediate data; selected frames and the sheet remain reproducible.
    shutil.rmtree(work / "candidate-frames", ignore_errors=True)
    return VideoSelectionResult(
        metadata=metadata,
        selected_paths=selected,
        frames=frame_records,
        contact_sheet_path=contact_sheet,
        commands=(probe_command, extraction.arguments),
    )


def _rate(value: str) -> float:
    fraction = Fraction(str(value))
    if fraction.denominator == 0:
        raise ValueError("Frame rate denominator is zero.")
    return float(fraction)


def _first_positive_float(*values: object) -> float:
    for value in values:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    raise ValueError("No positive numeric value was available.")


def _first_positive_rate(*values: object) -> float:
    for value in values:
        if value in {None, "", "N/A"}:
            continue
        try:
            parsed = _rate(str(value))
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    raise ValueError("No positive frame rate was available.")


def _viewpoint_change(first: np.ndarray | None, second: np.ndarray) -> float | None:
    if first is None:
        return None
    left = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=800)
    left_points, left_descriptors = detector.detectAndCompute(left, None)
    right_points, right_descriptors = detector.detectAndCompute(right, None)
    if left_descriptors is None or right_descriptors is None:
        return 1.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
        left_descriptors, right_descriptors
    )
    if len(matches) < 8:
        return 1.0
    matches = sorted(matches, key=lambda match: match.distance)[:200]
    displacements = [
        np.linalg.norm(
            np.asarray(left_points[match.queryIdx].pt)
            - np.asarray(right_points[match.trainIdx].pt)
        )
        for match in matches
    ]
    diagonal = math.hypot(second.shape[0], second.shape[1])
    return float(np.clip(np.median(displacements) / max(diagonal, 1.0), 0.0, 1.0))

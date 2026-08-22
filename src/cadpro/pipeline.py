from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cadpro.media import silhouette_from_media, silhouettes_from_turntable_video
from cadpro.step import solid_from_silhouette, visual_hull_from_silhouettes, write_step


@dataclass(frozen=True)
class ConversionResult:
    output: Path
    source_size: tuple[int, int]
    outline_points: int
    holes: int
    selected_frame: int | None


@dataclass(frozen=True)
class TurntableResult:
    output: Path
    source_size: tuple[int, int]
    sampled_frames: tuple[int, ...]
    outline_points: int
    holes: int


def convert_media(
    input_path: str | Path,
    output_path: str | Path,
    width_mm: float = 100.0,
    depth_mm: float = 10.0,
) -> ConversionResult:
    silhouette = silhouette_from_media(input_path)
    solid = solid_from_silhouette(silhouette, width_mm=width_mm, depth_mm=depth_mm)
    output = write_step(solid, output_path)
    return ConversionResult(output, silhouette.source_size, len(silhouette.outer), len(silhouette.holes), silhouette.frame_index)


def convert_turntable_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    width_mm: float,
    views: int = 8,
    start_frame: int = 0,
    end_frame: int | None = None,
    clockwise: bool = False,
) -> TurntableResult:
    silhouettes = silhouettes_from_turntable_video(
        input_path,
        view_count=views,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    solid = visual_hull_from_silhouettes(silhouettes, width_mm=width_mm, clockwise=clockwise)
    output = write_step(solid, output_path)
    return TurntableResult(
        output=output,
        source_size=silhouettes[0].source_size,
        sampled_frames=tuple(
            silhouette.frame_index
            for silhouette in silhouettes
            if silhouette.frame_index is not None
        ),
        outline_points=sum(len(silhouette.outer) for silhouette in silhouettes),
        holes=sum(len(silhouette.holes) for silhouette in silhouettes),
    )

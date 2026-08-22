from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cadpro.media import silhouette_from_media
from cadpro.step import solid_from_silhouette, write_step


@dataclass(frozen=True)
class ConversionResult:
    output: Path
    source_size: tuple[int, int]
    outline_points: int
    holes: int
    selected_frame: int | None


def convert_media(input_path: str | Path, output_path: str | Path, width_mm: float = 100.0, depth_mm: float = 10.0) -> ConversionResult:
    silhouette = silhouette_from_media(input_path)
    solid = solid_from_silhouette(silhouette, width_mm=width_mm, depth_mm=depth_mm)
    output = write_step(solid, output_path)
    return ConversionResult(output, silhouette.source_size, len(silhouette.outer), len(silhouette.holes), silhouette.frame_index)

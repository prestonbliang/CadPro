from __future__ import annotations

from pathlib import Path

import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Pnt, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Shape

from cadpro.media import Silhouette


def solid_from_silhouette(silhouette: Silhouette, width_mm: float, depth_mm: float) -> TopoDS_Shape:
    """Turn a 2D silhouette into a centered, watertight prismatic B-rep solid."""
    if width_mm <= 0 or depth_mm <= 0:
        raise ValueError("width_mm and depth_mm must both be positive")
    minimum = silhouette.outer.min(axis=0)
    maximum = silhouette.outer.max(axis=0)
    pixel_width = float(maximum[0] - minimum[0])
    if pixel_width <= 0:
        raise ValueError("The detected silhouette has zero width")
    scale = width_mm / pixel_width
    center = (minimum + maximum) / 2.0

    outer = _wire(silhouette.outer, center, scale, counter_clockwise=True)
    face_builder = BRepBuilderAPI_MakeFace(outer)
    for hole in silhouette.holes:
        face_builder.Add(_wire(hole, center, scale, counter_clockwise=False))
    if not face_builder.IsDone():
        raise RuntimeError("Could not build a planar CAD face from the detected outline")
    solid = BRepPrimAPI_MakePrism(face_builder.Face(), gp_Vec(0, 0, depth_mm)).Shape()
    if not BRepCheck_Analyzer(solid).IsValid():
        raise RuntimeError("Generated CAD body failed OpenCascade validity checks")
    return solid


def write_step(shape: TopoDS_Shape, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError("OpenCascade could not transfer the generated body to STEP")
    if writer.Write(str(destination)) != IFSelect_RetDone:
        raise RuntimeError(f"Could not write STEP file: {destination}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"STEP writer produced no output: {destination}")
    return destination


def _wire(points: np.ndarray, center: np.ndarray, scale: float, counter_clockwise: bool):
    transformed = np.column_stack(((points[:, 0] - center[0]) * scale, (center[1] - points[:, 1]) * scale))
    signed_area = 0.5 * np.sum(
        transformed[:, 0] * np.roll(transformed[:, 1], -1)
        - np.roll(transformed[:, 0], -1) * transformed[:, 1]
    )
    if (signed_area > 0) != counter_clockwise:
        transformed = transformed[::-1]
    wire = BRepBuilderAPI_MakeWire()
    for index, point in enumerate(transformed):
        following = transformed[(index + 1) % len(transformed)]
        wire.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(*point, 0), gp_Pnt(*following, 0)).Edge())
    if not wire.IsDone():
        raise RuntimeError("Could not build a closed CAD wire from the detected outline")
    return wire.Wire()

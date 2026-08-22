from __future__ import annotations

from contextlib import suppress
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ListOfShape
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


def visual_hull_from_silhouettes(
    silhouettes: tuple[Silhouette, ...],
    width_mm: float,
    clockwise: bool = False,
) -> TopoDS_Shape:
    """Intersect calibrated viewing prisms into a multi-view turntable visual hull."""
    if len(silhouettes) < 4:
        raise ValueError("Visual-hull reconstruction requires at least four views")
    if width_mm <= 0:
        raise ValueError("width_mm must be positive")
    if len({silhouette.source_size for silhouette in silhouettes}) != 1:
        raise ValueError("All sampled video frames must have the same dimensions")

    bounds = [
        (silhouette.outer.min(axis=0), silhouette.outer.max(axis=0))
        for silhouette in silhouettes
    ]
    pixel_width = max(float(maximum[0] - minimum[0]) for minimum, maximum in bounds)
    if pixel_width <= 0:
        raise ValueError("The sampled silhouettes have zero width")
    scale = width_mm / pixel_width
    frame_width, frame_height = silhouettes[0].source_size
    rotation_center = np.asarray(((frame_width - 1) / 2.0, (frame_height - 1) / 2.0))
    max_height_mm = max(float(maximum[1] - minimum[1]) * scale for minimum, maximum in bounds)
    max_radius_mm = max(
        float(np.max(np.abs(silhouette.outer[:, 0] - rotation_center[0]))) * scale
        for silhouette in silhouettes
    )
    angle_step = 2.0 * math.pi / len(silhouettes)
    profile_span = max(width_mm, max_height_mm, max_radius_mm * 2.0)
    half_span = profile_span * 1.1 / max(abs(math.sin(angle_step)), 0.1)

    result = None
    for view_index, silhouette in enumerate(silhouettes):
        angle = 2.0 * math.pi * view_index / len(silhouettes)
        inverse_angle = angle if clockwise else -angle
        view_solid = _viewing_prism(silhouette, rotation_center, scale, half_span, inverse_angle)
        if result is None:
            result = view_solid
            continue
        arguments = TopTools_ListOfShape()
        arguments.Append(result)
        tools = TopTools_ListOfShape()
        tools.Append(view_solid)
        operation = BRepAlgoAPI_Common()
        operation.SetArguments(arguments)
        operation.SetTools(tools)
        operation.SetFuzzyValue(1e-7)
        operation.Build()
        if not operation.IsDone():
            raise RuntimeError(f"Visual-hull intersection failed at view {view_index + 1}")
        result = operation.Shape()
        if result.IsNull() or _solid_count(result) == 0:
            raise RuntimeError(
                f"Views no longer overlap at view {view_index + 1}; check the turntable framing and frame range"
            )

    if result is None or not BRepCheck_Analyzer(result).IsValid():
        raise RuntimeError("Generated visual hull failed OpenCascade validity checks")
    solid_count = _solid_count(result)
    if solid_count != 1:
        raise RuntimeError(f"Visual hull produced {solid_count} separate solids; expected one connected object")
    if _volume(result) <= 1e-9:
        raise RuntimeError("Generated visual hull has no measurable volume")
    return result


def write_step(shape: TopoDS_Shape, output: str | Path) -> Path:
    destination = Path(output)
    if destination.suffix.lower() not in {".step", ".stp"}:
        raise ValueError("STEP output must use a .step or .stp extension")
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=".step",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        writer = STEPControl_Writer()
        if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
            raise RuntimeError("OpenCascade could not transfer the generated body to STEP")
        if writer.Write(str(temporary)) != IFSelect_RetDone:
            raise RuntimeError(f"Could not write STEP file: {destination}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"STEP writer produced no output: {destination}")
        os.replace(temporary, destination)
    except OSError as error:
        raise RuntimeError(f"Could not publish STEP file: {destination}") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
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


def _viewing_prism(
    silhouette: Silhouette,
    center: np.ndarray,
    scale: float,
    half_span: float,
    angle: float,
) -> TopoDS_Shape:
    outer = _xz_wire(silhouette.outer, center, scale, -half_span, counter_clockwise=True)
    face_builder = BRepBuilderAPI_MakeFace(outer)
    for hole in silhouette.holes:
        face_builder.Add(_xz_wire(hole, center, scale, -half_span, counter_clockwise=False))
    if not face_builder.IsDone():
        raise RuntimeError("Could not build a viewing face from a sampled silhouette")
    prism = BRepPrimAPI_MakePrism(face_builder.Face(), gp_Vec(0, half_span * 2.0, 0)).Shape()
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), angle)
    return BRepBuilderAPI_Transform(prism, transform, True).Shape()


def _xz_wire(
    points: np.ndarray,
    center: np.ndarray,
    scale: float,
    y: float,
    counter_clockwise: bool,
):
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
        edge = BRepBuilderAPI_MakeEdge(
            gp_Pnt(point[0], y, point[1]),
            gp_Pnt(following[0], y, following[1]),
        ).Edge()
        wire.Add(edge)
    if not wire.IsDone():
        raise RuntimeError("Could not build a viewing wire from a sampled silhouette")
    return wire.Wire()


def _solid_count(shape: TopoDS_Shape) -> int:
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _volume(shape: TopoDS_Shape) -> float:
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return properties.Mass()

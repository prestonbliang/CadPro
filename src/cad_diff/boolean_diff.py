from __future__ import annotations

from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape

from cad_diff.diff_model import BooleanCrossCheck


def _volume_of(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def _run_boolean(
    operation_type,
    left: TopoDS_Shape,
    right: TopoDS_Shape,
    description: str,
) -> TopoDS_Shape:
    """Run one OpenCascade boolean and reject incomplete or null results."""
    try:
        operation = operation_type(left, right)
    except Exception as exc:
        raise RuntimeError(f"Boolean {description} could not be started") from exc

    if not operation.IsDone():
        raise RuntimeError(f"Boolean {description} did not complete")

    try:
        result = operation.Shape()
    except Exception as exc:
        raise RuntimeError(f"Boolean {description} completed without a readable result") from exc
    if result.IsNull():
        raise RuntimeError(f"Boolean {description} produced a null shape")
    return result


def boolean_cross_check(base: TopoDS_Shape, modified: TopoDS_Shape) -> BooleanCrossCheck:
    """Tier 5: independent volumetric ground truth via boolean ops — the same
    technique SolidWorks's own Compare Geometry tool uses internally. Slow on
    complex many-body assemblies and blind to internal/hidden faces; used here
    as a cross-check on the face matcher's classification, not a replacement.

    On real (not synthetic) STEP data, BRepAlgoAPI_Cut can report IsDone()
    without erroring while still producing a geometrically invalid result —
    observed on a real Fusion-360-exported part, where two independently
    STEP-round-tripped copies of "the same" solid don't align to the same
    sub-micron tolerance everywhere, and the boolean op silently degenerates
    instead of failing loudly. Reporting that shape's volume as ground truth
    would be actively misleading (it read as "+1768/-1772mm3" for an actual
    ~4mm3 edit), so the result is checked and flagged unreliable rather than
    trusted blind."""
    added = _run_boolean(BRepAlgoAPI_Cut, modified, base, "added-volume cut")
    removed = _run_boolean(BRepAlgoAPI_Cut, base, modified, "removed-volume cut")
    common = _run_boolean(BRepAlgoAPI_Common, base, modified, "common-volume intersection")
    reliable = all(BRepCheck_Analyzer(shape).IsValid() for shape in (added, removed, common))
    return BooleanCrossCheck(
        added_volume=_volume_of(added),
        removed_volume=_volume_of(removed),
        common_volume=_volume_of(common),
        reliable=reliable,
    )

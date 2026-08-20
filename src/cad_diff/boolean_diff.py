from __future__ import annotations

from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape

from cad_diff.diff_model import BooleanCrossCheck


def _volume_of(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def boolean_cross_check(base: TopoDS_Shape, modified: TopoDS_Shape) -> BooleanCrossCheck:
    """Tier 5: independent volumetric ground truth via boolean ops — the same
    technique SolidWorks's own Compare Geometry tool uses internally. Slow on
    complex many-body assemblies and blind to internal/hidden faces; used here
    as a cross-check on the face matcher's classification, not a replacement."""
    added = BRepAlgoAPI_Cut(modified, base).Shape()
    removed = BRepAlgoAPI_Cut(base, modified).Shape()
    common = BRepAlgoAPI_Common(base, modified).Shape()
    return BooleanCrossCheck(
        added_volume=_volume_of(added),
        removed_volume=_volume_of(removed),
        common_volume=_volume_of(common),
    )

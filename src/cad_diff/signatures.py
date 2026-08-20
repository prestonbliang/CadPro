from __future__ import annotations

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape

from cad_diff.diff_model import SolidFingerprint


def fingerprint_solid(name: str, shape: TopoDS_Shape) -> SolidFingerprint:
    """Compute a geometric identity for a solid — independent of STEP entity IDs."""
    volume_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume_props)

    surface_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surface_props)

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    com = volume_props.CentreOfMass().Coord()

    return SolidFingerprint(
        name=name,
        volume=volume_props.Mass(),
        surface_area=surface_props.Mass(),
        center_of_mass=(com[0], com[1], com[2]),
        bbox_min=(xmin, ymin, zmin),
        bbox_max=(xmax, ymax, zmax),
    )

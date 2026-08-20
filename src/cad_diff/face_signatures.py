from __future__ import annotations

from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

from cad_diff.diff_model import FaceFingerprint

_SURFACE_TYPE_NAMES = {
    GeomAbs_SurfaceType.GeomAbs_Plane: "Plane",
    GeomAbs_SurfaceType.GeomAbs_Cylinder: "Cylinder",
    GeomAbs_SurfaceType.GeomAbs_Cone: "Cone",
    GeomAbs_SurfaceType.GeomAbs_Sphere: "Sphere",
    GeomAbs_SurfaceType.GeomAbs_Torus: "Torus",
    GeomAbs_SurfaceType.GeomAbs_BezierSurface: "BezierSurface",
    GeomAbs_SurfaceType.GeomAbs_BSplineSurface: "BSplineSurface",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution: "SurfaceOfRevolution",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion: "SurfaceOfExtrusion",
    GeomAbs_SurfaceType.GeomAbs_OffsetSurface: "OffsetSurface",
    GeomAbs_SurfaceType.GeomAbs_OtherSurface: "OtherSurface",
}


def extract_faces(shape: TopoDS_Shape) -> list[FaceFingerprint]:
    """Enumerate every face of a solid with a stable index, geometric signature,
    and adjacency graph — independent of STEP entity IDs and export ordering."""
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    adjacency: dict[int, set[int]] = {i: set() for i in range(1, face_map.Extent() + 1)}
    for e in range(1, edge_face_map.Extent() + 1):
        faces_on_edge = [face_map.FindIndex(f) for f in edge_face_map.FindFromIndex(e)]
        if len(faces_on_edge) == 2:
            a, b = faces_on_edge
            adjacency[a].add(b)
            adjacency[b].add(a)

    return [
        _fingerprint_face(i, TopoDS.Face_s(face_map.FindKey(i)), frozenset(adjacency[i]))
        for i in range(1, face_map.Extent() + 1)
    ]


def _fingerprint_face(index: int, face, adjacent: frozenset[int]) -> FaceFingerprint:
    area_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, area_props)

    bbox = Bnd_Box()
    BRepBndLib.Add_s(face, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    adaptor = BRepAdaptor_Surface(face)
    surface_type = _SURFACE_TYPE_NAMES.get(adaptor.GetType(), "OtherSurface")
    params = _analytic_params(adaptor, surface_type)

    centroid = area_props.CentreOfMass().Coord()
    return FaceFingerprint(
        index=index,
        surface_type=surface_type,
        area=area_props.Mass(),
        centroid=(centroid[0], centroid[1], centroid[2]),
        bbox_min=(xmin, ymin, zmin),
        bbox_max=(xmax, ymax, zmax),
        adjacent=adjacent,
        params=params,
    )


def _analytic_params(adaptor: BRepAdaptor_Surface, surface_type: str) -> dict[str, float]:
    if surface_type == "Cylinder":
        return {"radius": adaptor.Cylinder().Radius()}
    if surface_type == "Sphere":
        return {"radius": adaptor.Sphere().Radius()}
    if surface_type == "Cone":
        cone = adaptor.Cone()
        return {"radius": cone.RefRadius(), "semi_angle_deg": cone.SemiAngle() * 180.0 / 3.141592653589793}
    if surface_type == "Torus":
        torus = adaptor.Torus()
        return {"major_radius": torus.MajorRadius(), "minor_radius": torus.MinorRadius()}
    if surface_type == "Plane":
        plane = adaptor.Plane()
        loc = plane.Location().Coord()
        normal = plane.Axis().Direction().Coord()
        offset = loc[0] * normal[0] + loc[1] * normal[1] + loc[2] * normal[2]
        return {"offset": offset}
    return {}

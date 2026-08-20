from __future__ import annotations

from dataclasses import dataclass

from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_FACE, TopAbs_Orientation
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape


@dataclass(frozen=True)
class FaceMesh:
    """Triangulated geometry for one face, in the shape's own coordinate space."""

    vertices: list[tuple[float, float, float]]
    triangles: list[tuple[int, int, int]]  # 0-based indices into vertices, wound for an outward-facing normal


def tessellate_shape(shape: TopoDS_Shape, linear_deflection: float = 0.1) -> dict[int, FaceMesh]:
    """Triangulate every face of a shape, keyed by the same 1-based index
    face_signatures.py uses (a TopTools_IndexedMapOfShape over TopAbs_FACE),
    so tessellated triangles can be matched back to a face's diff status."""
    BRepMesh_IncrementalMesh(shape, linear_deflection)

    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)

    return {i: _tessellate_face(TopoDS.Face_s(face_map.FindKey(i))) for i in range(1, face_map.Extent() + 1)}


def _tessellate_face(face) -> FaceMesh:
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)
    if triangulation is None:
        return FaceMesh(vertices=[], triangles=[])

    transform = location.Transformation()
    vertices = [_point(triangulation.Node(i), transform) for i in range(1, triangulation.NbNodes() + 1)]

    # A REVERSED face's parametric winding runs opposite to the solid's
    # outward normal — flip it so every triangle in the combined mesh faces
    # the same way, or half the faces render as backfacing (dark/invisible).
    reversed_face = face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED

    triangles = []
    for i in range(1, triangulation.NbTriangles() + 1):
        a, b, c = triangulation.Triangle(i).Get()  # 1-based node indices
        if reversed_face:
            a, b, c = c, b, a
        triangles.append((a - 1, b - 1, c - 1))  # 0-based, into `vertices`

    return FaceMesh(vertices=vertices, triangles=triangles)


def _point(node, transform) -> tuple[float, float, float]:
    p = node.Transformed(transform)
    return (p.X(), p.Y(), p.Z())

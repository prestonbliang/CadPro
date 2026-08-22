import pytest
from OCP.TopoDS import TopoDS_Shape

import cad_diff.tessellate as tessellate


class _MissingTriangulation:
    @staticmethod
    def Triangulation_s(face, location):
        return None


class _FailedMesher:
    def __init__(self, shape, linear_deflection):
        pass

    def IsDone(self):
        return False

    def GetStatusFlags(self):
        return 42


def test_shape_without_faces_is_rejected():
    with pytest.raises(RuntimeError, match="Shape contains no faces"):
        tessellate.tessellate_shape(TopoDS_Shape())


def test_failed_meshing_operation_reports_status(monkeypatch):
    monkeypatch.setattr(tessellate, "BRepMesh_IncrementalMesh", _FailedMesher)

    with pytest.raises(RuntimeError, match=r"did not complete tessellation \(status flags: 42\)"):
        tessellate.tessellate_shape(object())


def test_face_without_triangulation_is_rejected(monkeypatch):
    monkeypatch.setattr(tessellate, "BRep_Tool", _MissingTriangulation)

    with pytest.raises(RuntimeError, match="no triangulation for face 7"):
        tessellate._tessellate_face(object(), 7)

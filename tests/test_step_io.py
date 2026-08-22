from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound
import pytest

from cad_diff.signatures import fingerprint_solid
from cad_diff.step_io import load_step


def test_compound_step_is_flattened_into_individual_solids(tmp_path):
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape())
    builder.Add(compound, BRepPrimAPI_MakeBox(2.0, 1.0, 1.0).Shape())

    step_path = tmp_path / "compound.step"
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    assert writer.Write(str(step_path)) == IFSelect_ReturnStatus.IFSelect_RetDone

    items = load_step(step_path)

    assert len(items) == 2
    volumes = sorted(fingerprint_solid(name, shape).volume for name, shape in items)
    assert volumes == pytest.approx([1.0, 2.0])

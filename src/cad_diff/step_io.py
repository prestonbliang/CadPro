from __future__ import annotations

from pathlib import Path

from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool


def load_step(path: str | Path) -> list[tuple[str, TopoDS_Shape]]:
    """Load a STEP file and return (name, shape) for each top-level free solid."""
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    app.InitDocument(doc)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")
    if not reader.Transfer(doc):
        raise RuntimeError(f"Failed to transfer STEP shapes into document: {path}")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)

    solids: list[tuple[str, TopoDS_Shape]] = []
    for i in range(1, labels.Length() + 1):
        name = _label_name(labels.Value(i), fallback=f"solid_{i}")
        shape = shape_tool.GetShape_s(labels.Value(i))
        children = _solids_in(shape)
        if len(children) == 1:
            solids.append((name, children[0]))
        else:
            solids.extend((f"{name}_{index}", child) for index, child in enumerate(children, start=1))

    if not solids:
        raise RuntimeError(f"STEP file contains no solid geometry: {path}")
    return solids


def _solids_in(shape: TopoDS_Shape) -> list[TopoDS_Shape]:
    """Flatten compounds/assemblies into located solid occurrences."""
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = []
    while explorer.More():
        solids.append(TopoDS.Solid_s(explorer.Current()))
        explorer.Next()
    return solids


def _label_name(label: TDF_Label, fallback: str) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        name = attr.Get().ToExtString()
        # STEP writers stamp this generic translator string when no real
        # product name was set at export time — not a name worth keeping.
        if name and "STEP translator" not in name:
            return name
    return fallback

from __future__ import annotations

from pathlib import Path

from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopoDS import TopoDS_Shape
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

    return [
        (_label_name(labels.Value(i), fallback=f"solid_{i}"), shape_tool.GetShape_s(labels.Value(i)))
        for i in range(1, labels.Length() + 1)
    ]


def _label_name(label: TDF_Label, fallback: str) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        name = attr.Get().ToExtString()
        # STEP writers stamp this generic translator string when no real
        # product name was set at export time — not a name worth keeping.
        if name and "STEP translator" not in name:
            return name
    return fallback

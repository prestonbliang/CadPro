from __future__ import annotations

from pathlib import Path

from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool


def load_step(path: str | Path) -> list[tuple[str, TopoDS_Shape]]:
    """Load a STEP file and return (name, shape) for every leaf solid — real
    assemblies come back from XCAF as one compound wrapping multiple parts
    (e.g. a housing + PCB + antenna), not one atomic shape, so each is walked
    down to its named, correctly-positioned leaf rather than treated as a
    single opaque blob."""
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

    results = []
    for i in range(1, labels.Length() + 1):
        label = labels.Value(i)
        name = _label_name(label, fallback=f"solid_{i}")
        results.extend(_expand(shape_tool, label, TopLoc_Location(), name))
    return results


def _expand(shape_tool: XCAFDoc_ShapeTool, label: TDF_Label, accumulated_loc: TopLoc_Location, name_hint: str) -> list[tuple[str, TopoDS_Shape]]:
    if not shape_tool.IsAssembly_s(label):
        shape = shape_tool.GetShape_s(label)
        located = shape.Located(accumulated_loc.Multiplied(shape.Location()))
        return [(name_hint, located)]

    results: list[tuple[str, TopoDS_Shape]] = []
    components = TDF_LabelSequence()
    shape_tool.GetComponents_s(label, components)
    for i in range(1, components.Length() + 1):
        comp_label = components.Value(i)
        comp_loc = shape_tool.GetLocation_s(comp_label)

        referred = TDF_Label()
        shape_tool.GetReferredShape_s(comp_label, referred)
        # The component label itself usually just holds a generic reference
        # id (e.g. "NAUO1"); the real product name lives on what it refers to.
        child_name = _label_name(referred, fallback=_label_name(comp_label, fallback=f"{name_hint}_{i}"))

        results.extend(_expand(shape_tool, referred, accumulated_loc.Multiplied(comp_loc), child_name))
    return results


def _label_name(label: TDF_Label, fallback: str) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        name = attr.Get().ToExtString()
        # STEP writers stamp this generic translator string when no real
        # product name was set at export time — not a name worth keeping.
        if name and "STEP translator" not in name:
            return name
    return fallback

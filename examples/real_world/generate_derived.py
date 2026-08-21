"""Regenerates the derived real-world fixtures from SAM_AP203.STEP: real,
SolidWorks-authored parts (not OCP-synthetic) with a small, deliberate,
verified-in-material edit cut into each — a housing wall (planes + a few
cylinders) and the antenna (which has genuine BSplineSurface faces).

The hole location for each part was found by scanning with
BRepClass3d_SolidClassifier — these parts have real internal cavities, so an
arbitrary point inside the bounding box usually lands in empty space, not
material. Run with: .venv/bin/python examples/real_world/generate_derived.py
"""

from __future__ import annotations

from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from cad_diff.step_io import load_step

HERE = Path(__file__).parent
SOURCE = HERE / "SAM_AP203.STEP"

# (part name in the assembly, hole center, hole radius, hole length)
EDITS = [
    ("Sam cavity", gp_Pnt(-10.0, -2.0, 8.0), 0.3, 3.0),
    ("SAM ANT", gp_Pnt(-10.0, 0.0, 6.0), 0.4, 8.0),
]


def write_step(shape, path: Path) -> None:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(path))


def main() -> None:
    parts = dict(load_step(SOURCE))
    for name, center, radius, length in EDITS:
        shape = parts[name]
        slug = name.lower().replace(" ", "_")

        write_step(shape, HERE / f"{slug}_v1.step")

        hole_axis = gp_Ax2(center, gp_Dir(0.0, 1.0, 0.0))
        hole = BRepPrimAPI_MakeCylinder(hole_axis, radius, length).Shape()
        modified = BRepAlgoAPI_Cut(shape, hole).Shape()
        write_step(modified, HERE / f"{slug}_v2_hole.step")

        print(f"wrote {slug}_v1.step and {slug}_v2_hole.step")


if __name__ == "__main__":
    main()

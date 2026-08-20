"""Regenerate the example STEP fixtures used in the README demo.

v1: a plain bracket base.
v2: the same base with a boss fused on top (a "modified" solid) plus a
    separate bolt (an "added" solid) — exercises both diff classes at once.
"""

from __future__ import annotations

from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound
from OCP.BRep import BRep_Builder

OUT_DIR = Path(__file__).parent


def write_step(shapes, path: Path) -> None:
    writer = STEPControl_Writer()
    for shape in shapes:
        writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(path))


def main() -> None:
    base = BRepPrimAPI_MakeBox(40.0, 30.0, 10.0).Shape()
    write_step([base], OUT_DIR / "bracket_v1.step")

    boss_axis = gp_Ax2(gp_Pnt(20.0, 15.0, 10.0), gp_Dir(0.0, 0.0, 1.0))
    boss = BRepPrimAPI_MakeCylinder(boss_axis, 5.0, 8.0).Shape()
    base_with_boss = BRepAlgoAPI_Fuse(base, boss).Shape()

    bolt_axis = gp_Ax2(gp_Pnt(60.0, 15.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    bolt = BRepPrimAPI_MakeCylinder(bolt_axis, 3.0, 20.0).Shape()

    write_step([base_with_boss, bolt], OUT_DIR / "bracket_v2.step")
    print("wrote", OUT_DIR / "bracket_v1.step", "and", OUT_DIR / "bracket_v2.step")


if __name__ == "__main__":
    main()

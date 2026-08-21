"""Regenerates the derived real-world fixtures: real, non-OCP-authored parts
(SolidWorks and Fusion 360) with a small, deliberate, verified-in-material
edit cut into each. See NOTICE.md for provenance and licensing.

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

# (source file, part name within it, output slug, hole center, hole direction, hole radius, hole length)
EDITS = [
    (HERE / "SAM_AP203.STEP", "Sam cavity", "sam_cavity", gp_Pnt(-10.0, -2.0, 8.0), (0.0, 1.0, 0.0), 0.3, 3.0),
    (HERE / "SAM_AP203.STEP", "SAM ANT", "sam_ant", gp_Pnt(-10.0, 0.0, 6.0), (0.0, 1.0, 0.0), 0.4, 8.0),
    (HERE / "tactile_switch.step", "Tactile On/Off Button v3", "tactile_switch", gp_Pnt(-6.42, -8.33, -1.0), (0.0, 0.0, 1.0), 0.5, 10.0),
]


def write_step(shape, path: Path) -> None:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(path))


def main() -> None:
    parts_by_source: dict[Path, dict[str, object]] = {}
    for source, part_name, slug, center, direction, radius, length in EDITS:
        if source not in parts_by_source:
            parts_by_source[source] = dict(load_step(source))
        shape = parts_by_source[source][part_name]

        write_step(shape, HERE / f"{slug}_v1.step")

        hole_axis = gp_Ax2(center, gp_Dir(*direction))
        hole = BRepPrimAPI_MakeCylinder(hole_axis, radius, length).Shape()
        modified = BRepAlgoAPI_Cut(shape, hole).Shape()
        write_step(modified, HERE / f"{slug}_v2_hole.step")

        print(f"wrote {slug}_v1.step and {slug}_v2_hole.step")


if __name__ == "__main__":
    main()

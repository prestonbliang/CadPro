from __future__ import annotations

import typer

from cad_diff.boolean_diff import boolean_cross_check
from cad_diff.diff_model import DiffReport, SolidFaceDiff
from cad_diff.face_matcher import match_faces
from cad_diff.face_signatures import extract_faces
from cad_diff.matcher import match_solids
from cad_diff.report import print_report
from cad_diff.signatures import fingerprint_solid
from cad_diff.step_io import load_step

app = typer.Typer(add_completion=False)


@app.command()
def diff(base: str, modified: str) -> None:
    """Diff two STEP files and print what changed."""
    base_items = load_step(base)
    mod_items = load_step(modified)

    base_fps = [fingerprint_solid(name, shape) for name, shape in base_items]
    mod_fps = [fingerprint_solid(name, shape) for name, shape in mod_items]
    # Keyed by object identity, not value equality — two structurally
    # identical parts (e.g. two same-size bolts) must not collide here.
    base_shape_of = {id(fp): shape for fp, (_, shape) in zip(base_fps, base_items)}
    mod_shape_of = {id(fp): shape for fp, (_, shape) in zip(mod_fps, mod_items)}

    solid_diffs = match_solids(base_fps, mod_fps)

    face_diffs = []
    for solid_diff in solid_diffs:
        if solid_diff.status != "modified":
            continue
        base_shape = base_shape_of[id(solid_diff.base)]
        mod_shape = mod_shape_of[id(solid_diff.modified)]
        face_diffs.append(
            SolidFaceDiff(
                solid=solid_diff,
                faces=match_faces(extract_faces(base_shape), extract_faces(mod_shape)),
                boolean=boolean_cross_check(base_shape, mod_shape),
            )
        )

    report = DiffReport(base_path=base, modified_path=modified, solids=solid_diffs, face_diffs=face_diffs)
    print_report(report)


if __name__ == "__main__":
    app()

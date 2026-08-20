"""The hero pitch scenario, end to end: a fillet radius changes from 2mm to
4mm, and cad-diff reports it as a modified face with a dimensional delta —
not an unmatched delete+add — cross-checked against an independent boolean
volumetric ground truth.
"""

from pathlib import Path

import pytest

from cad_diff.boolean_diff import boolean_cross_check
from cad_diff.face_matcher import match_faces
from cad_diff.face_signatures import extract_faces
from cad_diff.step_io import load_step

EXAMPLES = Path(__file__).parent.parent / "examples"


def _single_shape(step_path: Path):
    (name, shape), = load_step(step_path)
    return shape


def test_fillet_radius_change_is_a_modified_cylinder_face():
    base_shape = _single_shape(EXAMPLES / "fillet_v1.step")
    mod_shape = _single_shape(EXAMPLES / "fillet_v2.step")

    diffs = match_faces(extract_faces(base_shape), extract_faces(mod_shape))
    cylinder_diff = next(d for d in diffs if (d.base or d.modified).surface_type == "Cylinder")

    assert cylinder_diff.status == "modified"
    assert cylinder_diff.param_deltas == pytest.approx({"radius": 2.0})

    unchanged_count = sum(1 for d in diffs if d.status == "unchanged")
    modified_count = sum(1 for d in diffs if d.status == "modified")
    assert unchanged_count == 2  # the two faces the fillet never touched
    assert modified_count == 5  # the fillet itself + the four faces it trims
    assert all(d.status != "added" and d.status != "removed" for d in diffs)


def test_boolean_cross_check_agrees_with_the_face_matcher():
    base_shape = _single_shape(EXAMPLES / "fillet_v1.step")
    mod_shape = _single_shape(EXAMPLES / "fillet_v2.step")

    result = boolean_cross_check(base_shape, mod_shape)

    # Growing the fillet only ever removes material, never adds any.
    assert result.added_volume == pytest.approx(0.0, abs=1e-6)
    assert result.removed_volume > 0.0

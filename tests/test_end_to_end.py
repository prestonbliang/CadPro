from pathlib import Path

import pytest

from cad_diff.matcher import match_solids
from cad_diff.signatures import fingerprint_solid
from cad_diff.step_io import load_step

EXAMPLES = Path(__file__).parent.parent / "examples"


def _fingerprints(step_path: Path):
    return [fingerprint_solid(name, shape) for name, shape in load_step(step_path)]


def test_bracket_v1_is_a_single_box():
    (box,) = _fingerprints(EXAMPLES / "bracket_v1.step")
    assert box.volume == 12000.0


def test_v1_to_v2_shows_modified_boss_and_added_bolt():
    diffs = match_solids(_fingerprints(EXAMPLES / "bracket_v1.step"), _fingerprints(EXAMPLES / "bracket_v2.step"))
    statuses = sorted(d.status for d in diffs)
    assert statuses == ["added", "modified"]

    modified = next(d for d in diffs if d.status == "modified")
    assert modified.volume_delta == pytest.approx(628.3185307179587)  # pi * 5^2 * 8, the fused boss


def test_self_diff_is_unchanged():
    fps = _fingerprints(EXAMPLES / "bracket_v1.step")
    (diff,) = match_solids(fps, fps)
    assert diff.status == "unchanged"

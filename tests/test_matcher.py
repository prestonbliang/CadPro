import pytest

from cad_diff.diff_model import SolidFingerprint
from cad_diff.matcher import match_solids

BASE_BOX = SolidFingerprint(
    name="box",
    volume=12000.0,
    surface_area=3800.0,
    center_of_mass=(20.0, 15.0, 5.0),
    bbox_min=(0.0, 0.0, 0.0),
    bbox_max=(40.0, 30.0, 10.0),
)


def _shifted(fp: SolidFingerprint, **overrides) -> SolidFingerprint:
    return SolidFingerprint(**{**fp.__dict__, **overrides})


def test_identical_solid_is_unchanged():
    (diff,) = match_solids([BASE_BOX], [BASE_BOX])
    assert diff.status == "unchanged"
    assert diff.volume_delta == 0.0


def test_volume_change_is_modified():
    grown = _shifted(BASE_BOX, volume=12628.319, surface_area=4051.327)
    (diff,) = match_solids([BASE_BOX], [grown])
    assert diff.status == "modified"
    assert diff.volume_delta == pytest.approx(628.319)


def test_extra_solid_is_added():
    bolt = SolidFingerprint(
        name="bolt", volume=565.5, surface_area=433.5,
        center_of_mass=(60.0, 15.0, 10.0), bbox_min=(57.0, 12.0, 0.0), bbox_max=(63.0, 18.0, 20.0),
    )
    diffs = match_solids([BASE_BOX], [BASE_BOX, bolt])
    statuses = {d.status for d in diffs}
    assert statuses == {"unchanged", "added"}


def test_missing_solid_is_removed():
    bolt = SolidFingerprint(
        name="bolt", volume=565.5, surface_area=433.5,
        center_of_mass=(60.0, 15.0, 10.0), bbox_min=(57.0, 12.0, 0.0), bbox_max=(63.0, 18.0, 20.0),
    )
    diffs = match_solids([BASE_BOX, bolt], [BASE_BOX])
    statuses = {d.status for d in diffs}
    assert statuses == {"unchanged", "removed"}


def test_empty_base_is_all_added():
    (diff,) = match_solids([], [BASE_BOX])
    assert diff.status == "added"

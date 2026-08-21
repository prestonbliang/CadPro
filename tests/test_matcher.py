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


def test_equal_volume_and_center_but_different_geometry_is_modified():
    reshaped = _shifted(
        BASE_BOX,
        surface_area=3900.0,
        bbox_min=(-1.0, 0.0, 0.0),
        bbox_max=(41.0, 30.0, 10.0),
    )

    (diff,) = match_solids([BASE_BOX], [reshaped])

    assert diff.status == "modified"


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


def test_unrelated_solids_are_removed_and_added_not_force_matched():
    # A tiny bracket and a huge, distant plate as the only solid on each side
    # must not become "the bracket turned into a plate" just because
    # Hungarian assignment always pairs whatever's left over.
    tiny_bracket = SolidFingerprint(
        name="bracket", volume=100.0, surface_area=200.0,
        center_of_mass=(0.0, 0.0, 0.0), bbox_min=(0.0, 0.0, 0.0), bbox_max=(5.0, 5.0, 4.0),
    )
    huge_plate = SolidFingerprint(
        name="plate", volume=500_000.0, surface_area=90_000.0,
        center_of_mass=(2000.0, 2000.0, 2000.0), bbox_min=(0.0, 0.0, 0.0), bbox_max=(500.0, 500.0, 2.0),
    )

    diffs = match_solids([tiny_bracket], [huge_plate])

    statuses = sorted(d.status for d in diffs)
    assert statuses == ["added", "removed"]

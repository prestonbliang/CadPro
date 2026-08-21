from cad_diff.diff_model import FaceFingerprint
from cad_diff.face_matcher import match_faces


def _face(index, surface_type, area, centroid, adjacent, params=None):
    return FaceFingerprint(
        index=index,
        surface_type=surface_type,
        area=area,
        centroid=centroid,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(0.0, 0.0, 0.0),
        adjacent=frozenset(adjacent),
        params=params or {},
    )


def _tiers(diffs):
    return {(d.base or d.modified).index: d.tier for d in diffs}


def _statuses(diffs):
    return {(d.base or d.modified).index: d.status for d in diffs}


def test_tier1_matches_unique_types_directly():
    base = [_face(1, "Sphere", 10.0, (0, 0, 0), [2]), _face(2, "Plane", 20.0, (5, 0, 0), [1])]
    modified = [_face(1, "Sphere", 10.0, (0, 0, 0), [2]), _face(2, "Plane", 20.0, (5, 0, 0), [1])]

    diffs = match_faces(base, modified)

    assert _tiers(diffs) == {1: "T1", 2: "T1"}
    assert _statuses(diffs) == {1: "unchanged", 2: "unchanged"}


def test_tier1_rejects_a_pair_too_dissimilar_to_be_the_same_face():
    # Only one Plane on each side, but they're unrelated features, not the same
    # face having moved a little — Tier 1 must not force-match them.
    base = [_face(1, "Plane", 20.0, (0, 0, 0), [])]
    modified = [_face(1, "Plane", 20.0, (500, 500, 500), [])]

    diffs = match_faces(base, modified)

    statuses = sorted(d.status for d in diffs)
    assert statuses == ["added", "removed"]


def test_tier3_propagates_through_an_already_matched_anchor():
    # F2's own signature shifted enough that a bare global bucket match
    # (Tier 1) would reject it — but it's the only unmatched neighbor of an
    # anchor (F1) that Tier 1 *did* match confidently, so Tier 3 should
    # recover it via adjacency propagation instead of leaving it unmatched.
    base = [
        _face(1, "Sphere", 10.0, (0, 0, 0), [2]),
        _face(2, "Plane", 20.0, (5, 0, 0), [1]),
    ]
    modified = [
        _face(1, "Sphere", 10.0, (0, 0, 0), [2]),  # identical -> trivial Tier 1 anchor
        _face(2, "Plane", 20.0, (9, 0, 0), [1]),  # centroid moved 4.0 -> cost 4.0, between the two thresholds
    ]

    diffs = match_faces(base, modified)

    assert _tiers(diffs) == {1: "T1", 2: "T3"}
    assert _statuses(diffs)[2] == "modified"


def test_tier4_resolves_pure_topology_when_positions_are_useless():
    # Both faces relocated wildly (as if the whole part's datum/origin moved
    # between exports), so raw geometric cost rejects everything in both
    # Tier 1 and Tier 3, and there's no already-matched anchor to propagate
    # from. Only the surface-type + adjacency shape can resolve this.
    base = [
        _face(1, "Sphere", 10.0, (0, 0, 0), [2]),
        _face(2, "Plane", 20.0, (5, 0, 0), [1]),
    ]
    modified = [
        _face(1, "Sphere", 10.0, (1000, 1000, 1000), [2]),
        _face(2, "Plane", 20.0, (1005, 1000, 1000), [1]),
    ]

    diffs = match_faces(base, modified)

    assert _tiers(diffs) == {1: "T4", 2: "T4"}


def test_tier4_does_not_fire_on_mismatched_residual_sizes():
    # A genuinely added face alongside an unrelated survivor must not get
    # dragged into a forced isomorphism just because residual matching ran.
    base = [_face(1, "Sphere", 10.0, (0, 0, 0), [])]
    modified = [
        _face(1, "Sphere", 10.0, (0, 0, 0), []),
        _face(2, "Plane", 5.0, (50, 50, 50), []),
    ]

    diffs = match_faces(base, modified)

    statuses = _statuses(diffs)
    assert statuses[1] == "unchanged"
    assert 2 in [(d.base or d.modified).index for d in diffs if d.status == "added"]


def test_param_deltas_report_the_dimensional_change():
    base = [_face(1, "Cylinder", 31.42, (0, 0, 0), [], params={"radius": 2.0})]
    modified = [_face(1, "Cylinder", 62.83, (0, 0, 0), [], params={"radius": 4.0})]

    (diff,) = match_faces(base, modified)

    assert diff.status == "modified"
    assert diff.param_deltas == {"radius": 2.0}


def test_large_single_type_bucket_stays_fast():
    # A real 811-face PCB (see examples/real_world/NOTICE.md) put 671 faces
    # in a single Plane bucket for Tier 1 and self-diffed in 0.33s --
    # confirming scipy's linear_sum_assignment on a 671x671 cost matrix
    # isn't a scaling cliff. That PCB isn't vendored here (would mean
    # shipping its 7MB source assembly); this locks in a bound on the same
    # scale with clean synthetic data, isolating the actual risk (bucket
    # size) without depending on an unvendored real fixture.
    import time

    faces = [_face(i, "Plane", 10.0 + i * 0.01, (float(i), 0.0, 0.0), []) for i in range(1, 701)]

    t0 = time.time()
    diffs = match_faces(faces, faces)
    assert time.time() - t0 < 5.0
    assert all(d.status == "unchanged" for d in diffs)

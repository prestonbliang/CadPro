from __future__ import annotations

import math

from scipy.optimize import linear_sum_assignment

from cad_diff.diff_model import SolidDiff, SolidFingerprint

# Below this, a delta is STEP round-trip noise, not a real geometric change.
UNCHANGED_VOLUME_TOL = 1e-4  # mm^3
UNCHANGED_AREA_TOL = 1e-4  # mm^2
UNCHANGED_COM_TOL = 1e-4  # mm
UNCHANGED_BBOX_TOL = 1e-4  # mm

# Above this, two fingerprints are too dissimilar to be the same solid at all —
# treat them as separate added/removed rather than a wildly "modified" match.
# Solids can legitimately move further than faces during a real edit (a part
# getting repositioned in an assembly), so this is looser than the face-level
# equivalent (TIER1_REJECT_COST in face_matcher.py).
NOT_A_MATCH_COST = 5.0


def _cost(a: SolidFingerprint, b: SolidFingerprint) -> float:
    volume_delta = abs(a.volume - b.volume)
    area_delta = abs(a.surface_area - b.surface_area)
    com_delta = math.dist(a.center_of_mass, b.center_of_mass)
    # Scale-relative: a 1mm^3 shift matters on a small bracket, not on a chassis.
    scale = max(a.volume, b.volume, 1.0)
    return (volume_delta + area_delta) / scale + com_delta


def _max_bbox_delta(a: SolidFingerprint, b: SolidFingerprint) -> float:
    return max(
        abs(left - right)
        for left, right in zip((*a.bbox_min, *a.bbox_max), (*b.bbox_min, *b.bbox_max))
    )


def match_solids(base: list[SolidFingerprint], modified: list[SolidFingerprint]) -> list[SolidDiff]:
    """Tier 0: match whole solids by geometric fingerprint, not STEP entity ID."""
    if not base or not modified:
        diffs = [SolidDiff(status="removed", base=fp, modified=None) for fp in base]
        diffs += [SolidDiff(status="added", base=None, modified=fp) for fp in modified]
        return diffs

    cost_matrix = [[_cost(a, b) for b in modified] for a in base]
    base_idx, mod_idx = linear_sum_assignment(cost_matrix)

    matched_base: set[int] = set()
    matched_mod: set[int] = set()
    diffs: list[SolidDiff] = []

    for i, j in zip(base_idx, mod_idx):
        if cost_matrix[i][j] > NOT_A_MATCH_COST:
            continue  # too dissimilar to be the same solid — leave for removed/added below
        a, b = base[i], modified[j]
        volume_delta = abs(a.volume - b.volume)
        area_delta = abs(a.surface_area - b.surface_area)
        com_delta = math.dist(a.center_of_mass, b.center_of_mass)
        bbox_delta = _max_bbox_delta(a, b)
        status = (
            "unchanged"
            if volume_delta < UNCHANGED_VOLUME_TOL
            and area_delta < UNCHANGED_AREA_TOL
            and com_delta < UNCHANGED_COM_TOL
            and bbox_delta < UNCHANGED_BBOX_TOL
            else "modified"
        )
        diffs.append(SolidDiff(status=status, base=a, modified=b))
        matched_base.add(i)
        matched_mod.add(j)

    diffs += [SolidDiff(status="removed", base=fp, modified=None) for i, fp in enumerate(base) if i not in matched_base]
    diffs += [SolidDiff(status="added", base=None, modified=fp) for j, fp in enumerate(modified) if j not in matched_mod]
    return diffs

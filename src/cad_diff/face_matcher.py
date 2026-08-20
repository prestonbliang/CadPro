from __future__ import annotations

import math

import networkx as nx
from scipy.optimize import linear_sum_assignment

from cad_diff.diff_model import FaceDiff, FaceFingerprint

# Tier 1 (bucketed Hungarian assignment): reject a same-type pairing this
# dissimilar rather than force-matching two unrelated faces of the same type.
TIER1_REJECT_COST = 3.0

# Tier 3 (adjacency propagation): looser than Tier 1 on purpose — a face
# reached via already-matched neighbors is trusted more than raw similarity,
# since that's exactly how "this fillet's radius changed" gets recognized as
# a modification instead of an unmatched delete+add.
TIER3_REJECT_COST = 8.0

# Tier 4 (VF2 residual isomorphism): general graph isomorphism is NP-complete,
# so this only ever runs on the small leftover island after Tiers 1 and 3.
TIER4_MAX_RESIDUAL = 12

UNCHANGED_AREA_TOL = 1e-4  # mm^2, round-trip noise
UNCHANGED_CENTROID_TOL = 1e-4  # mm


def _cost(a: FaceFingerprint, b: FaceFingerprint) -> float:
    if a.surface_type != b.surface_type:
        return math.inf
    area_delta = abs(a.area - b.area)
    centroid_delta = math.dist(a.centroid, b.centroid)
    scale = max(a.area, b.area, 1.0)
    return area_delta / scale + centroid_delta


def _status(a: FaceFingerprint, b: FaceFingerprint) -> str:
    area_delta = abs(a.area - b.area)
    centroid_delta = math.dist(a.centroid, b.centroid)
    if area_delta < UNCHANGED_AREA_TOL and centroid_delta < UNCHANGED_CENTROID_TOL:
        return "unchanged"
    return "modified"


def match_faces(base: list[FaceFingerprint], modified: list[FaceFingerprint]) -> list[FaceDiff]:
    """Match faces between two versions of a solid by geometric signature and
    topology, not STEP entity ID — entity numbering isn't stable across
    re-exports of "the same" geometry."""
    base_by_index = {f.index: f for f in base}
    mod_by_index = {f.index: f for f in modified}
    unmatched_base = dict(base_by_index)
    unmatched_mod = dict(mod_by_index)

    correspondence: dict[int, int] = {}
    tier_of: dict[int, str] = {}

    _tier1_bucketed_assignment(unmatched_base, unmatched_mod, correspondence, tier_of)
    _tier3_adjacency_propagation(base_by_index, mod_by_index, unmatched_base, unmatched_mod, correspondence, tier_of)
    _tier4_residual_isomorphism(unmatched_base, unmatched_mod, correspondence, tier_of)

    diffs = [
        FaceDiff(status=_status(base_by_index[bi], mod_by_index[mi]), tier=tier_of[bi], base=base_by_index[bi], modified=mod_by_index[mi])
        for bi, mi in correspondence.items()
    ]
    diffs += [FaceDiff(status="removed", tier="unmatched", base=f, modified=None) for f in unmatched_base.values()]
    diffs += [FaceDiff(status="added", tier="unmatched", base=None, modified=f) for f in unmatched_mod.values()]
    return diffs


def _tier1_bucketed_assignment(
    unmatched_base: dict[int, FaceFingerprint],
    unmatched_mod: dict[int, FaceFingerprint],
    correspondence: dict[int, int],
    tier_of: dict[int, str],
) -> None:
    """Bucket by surface type (a hard invariant — a cylinder can't become a
    plane without being an add+remove), then resolve each bucket with optimal
    assignment. This is Hungarian for every bucket, not just ambiguous ones:
    it degenerates to the trivial 1:1 case for free when a bucket has a single
    candidate on each side, so there's no need for a separate code path."""
    types = {f.surface_type for f in unmatched_base.values()} & {f.surface_type for f in unmatched_mod.values()}
    for surface_type in types:
        base_group = [f for f in unmatched_base.values() if f.surface_type == surface_type]
        mod_group = [f for f in unmatched_mod.values() if f.surface_type == surface_type]
        cost_matrix = [[_cost(a, b) for b in mod_group] for a in base_group]
        for r, c in zip(*linear_sum_assignment(cost_matrix)):
            if cost_matrix[r][c] > TIER1_REJECT_COST:
                continue
            a, b = base_group[r], mod_group[c]
            correspondence[a.index] = b.index
            tier_of[a.index] = "T1"
            del unmatched_base[a.index]
            del unmatched_mod[b.index]


def _tier3_adjacency_propagation(
    base_by_index: dict[int, FaceFingerprint],
    mod_by_index: dict[int, FaceFingerprint],
    unmatched_base: dict[int, FaceFingerprint],
    unmatched_mod: dict[int, FaceFingerprint],
    correspondence: dict[int, int],
    tier_of: dict[int, str],
) -> None:
    """BFS outward from confident (Tier 1) anchors through the face-adjacency
    graph: the unmatched face next to several already-matched faces is almost
    certainly their shared neighbor's counterpart, even if its own signature
    shifted — this is what turns a resized fillet into a "modified" face
    instead of an unmatched delete+add pair."""
    worklist = list(correspondence.keys())
    while worklist:
        bi = worklist.pop()
        mi = correspondence[bi]
        b_face, m_face = base_by_index[bi], mod_by_index[mi]

        for bn_index in b_face.adjacent:
            if bn_index not in unmatched_base:
                continue
            bn_face = unmatched_base[bn_index]
            candidates = [
                mod_by_index[mn]
                for mn in m_face.adjacent
                if mn in unmatched_mod and mod_by_index[mn].surface_type == bn_face.surface_type
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda c: _cost(bn_face, c))
            if _cost(bn_face, best) > TIER3_REJECT_COST:
                continue
            correspondence[bn_index] = best.index
            tier_of[bn_index] = "T3"
            del unmatched_base[bn_index]
            del unmatched_mod[best.index]
            worklist.append(bn_index)


def _tier4_residual_isomorphism(
    unmatched_base: dict[int, FaceFingerprint],
    unmatched_mod: dict[int, FaceFingerprint],
    correspondence: dict[int, int],
    tier_of: dict[int, str],
) -> None:
    """General graph isomorphism is NP-complete — only ever run it on the
    small leftover island after Tiers 1 and 3, never the whole model. If the
    residual sizes differ there's no clean isomorphism (real topology was
    added or removed, not just relabeled) — leave it to added/removed."""
    if not unmatched_base or not unmatched_mod:
        return
    if len(unmatched_base) != len(unmatched_mod) or len(unmatched_base) > TIER4_MAX_RESIDUAL:
        return

    base_graph = _adjacency_subgraph(unmatched_base)
    mod_graph = _adjacency_subgraph(unmatched_mod)
    if base_graph.number_of_edges() == 0:
        # No adjacency structure left to match on — every face here is
        # isolated within the residual. Matching by surface type alone, with
        # no position and no topology, is exactly the unconstrained guess
        # Tier 1's cost threshold exists to reject; leave these as added/removed.
        return
    matcher = nx.algorithms.isomorphism.GraphMatcher(base_graph, mod_graph, node_match=lambda a, b: a["t"] == b["t"])
    if not matcher.is_isomorphic():
        return

    for bi, mi in dict(matcher.mapping).items():
        correspondence[bi] = mi
        tier_of[bi] = "T4"
        del unmatched_base[bi]
        del unmatched_mod[mi]


def _adjacency_subgraph(faces: dict[int, FaceFingerprint]) -> nx.Graph:
    graph = nx.Graph()
    for f in faces.values():
        graph.add_node(f.index, t=f.surface_type)
    for f in faces.values():
        for neighbor in f.adjacent:
            if neighbor in faces:
                graph.add_edge(f.index, neighbor)
    return graph

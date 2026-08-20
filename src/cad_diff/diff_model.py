from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolidFingerprint:
    """Geometric identity of a solid, independent of STEP entity IDs."""

    name: str
    volume: float
    surface_area: float
    center_of_mass: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]


@dataclass(frozen=True)
class SolidDiff:
    """Result of matching one solid between the base and modified assembly."""

    status: str  # "unchanged" | "modified" | "added" | "removed"
    base: SolidFingerprint | None
    modified: SolidFingerprint | None

    @property
    def volume_delta(self) -> float | None:
        if self.base is None or self.modified is None:
            return None
        return self.modified.volume - self.base.volume

    @property
    def surface_area_delta(self) -> float | None:
        if self.base is None or self.modified is None:
            return None
        return self.modified.surface_area - self.base.surface_area


@dataclass(frozen=True)
class FaceFingerprint:
    """Geometric identity of a face, independent of STEP entity ID and index order."""

    index: int  # 1-based, stable within this shape's IndexedMapOfShape only
    surface_type: str  # GeomAbs_SurfaceType name: "Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSplineSurface", ...
    area: float
    centroid: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    adjacent: frozenset[int]  # indices of faces sharing an edge with this one, within the same shape
    params: dict[str, float]  # analytic surface params where extractable: radius, major_radius, minor_radius, ...


@dataclass(frozen=True)
class FaceDiff:
    """Result of matching one face between the base and modified solid."""

    status: str  # "unchanged" | "modified" | "added" | "removed"
    tier: str  # which matcher tier produced this correspondence, for debugging/reporting
    base: FaceFingerprint | None
    modified: FaceFingerprint | None

    @property
    def area_delta(self) -> float | None:
        if self.base is None or self.modified is None:
            return None
        return self.modified.area - self.base.area

    @property
    def param_deltas(self) -> dict[str, float]:
        """Deltas for analytic params shared by both sides (e.g. radius: 2.0 -> 4.0mm)."""
        if self.base is None or self.modified is None:
            return {}
        shared = self.base.params.keys() & self.modified.params.keys()
        return {k: self.modified.params[k] - self.base.params[k] for k in shared}


@dataclass(frozen=True)
class BooleanCrossCheck:
    """Tier 5 ground truth: what BRepAlgoAPI_Cut/Common say actually changed,
    independent of face correspondence — the ground truth the face matcher is checked against."""

    added_volume: float
    removed_volume: float
    common_volume: float


@dataclass(frozen=True)
class SolidFaceDiff:
    """Face-level breakdown for one matched (modified) solid pair."""

    solid: SolidDiff
    faces: list[FaceDiff]
    boolean: BooleanCrossCheck


@dataclass(frozen=True)
class DiffReport:
    """Top-level result of diffing two STEP assemblies."""

    base_path: str
    modified_path: str
    solids: list[SolidDiff]
    face_diffs: list[SolidFaceDiff]

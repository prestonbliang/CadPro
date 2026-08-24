"""Conservative analytic feature fitting and validated OpenCascade STEP output."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import least_squares

from cad_diff.signatures import fingerprint_solid
from cad_diff.step_io import load_step
from cadpro.scan.models import FeatureFit, ScaleInformation
from cadpro.scan.scale import convert_distance
from cadpro.step import write_step


@dataclass(frozen=True)
class CadFitResult:
    features: tuple[FeatureFit, ...]
    selected: FeatureFit | None
    explanation: str


@dataclass(frozen=True)
class CadExportResult:
    step_path: Path
    script_path: Path
    feature: FeatureFit


def fit_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold: float,
    minimum_inlier_ratio: float = 0.7,
    iterations: int = 300,
    seed: int = 17,
) -> FeatureFit:
    cloud = _points(points)
    if len(cloud) < 3:
        raise ValueError("Plane fitting needs at least three points.")
    if distance_threshold <= 0 or iterations <= 0:
        raise ValueError("Plane fitting thresholds and iterations must be positive.")
    random = np.random.default_rng(seed)
    best_mask = np.zeros(len(cloud), dtype=bool)
    for _ in range(iterations):
        sample = cloud[random.choice(len(cloud), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1e-12:
            continue
        normal /= magnitude
        offset = -float(np.dot(normal, sample[0]))
        mask = np.abs(cloud @ normal + offset) <= distance_threshold
        if int(mask.sum()) > int(best_mask.sum()):
            best_mask = mask
    if int(best_mask.sum()) < 3:
        return _failed_fit("plane", "No stable plane hypothesis was found.")
    inliers = cloud[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, right = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = right[-1]
    normal /= np.linalg.norm(normal)
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    residuals = np.abs(cloud @ normal + offset)
    refined_mask = residuals <= distance_threshold
    inlier_ratio = float(refined_mask.mean())
    rms = float(np.sqrt(np.mean(np.square(residuals[refined_mask]))))
    confidence = _confidence(inlier_ratio, rms, distance_threshold)
    accepted = inlier_ratio >= minimum_inlier_ratio and confidence >= 0.75
    return FeatureFit(
        feature_type="plane",
        parameters={
            "normal": normal.tolist(),
            "offset": offset,
            "centroid": centroid.tolist(),
        },
        inlier_ratio=inlier_ratio,
        rms_residual=rms,
        confidence=confidence,
        accepted=accepted,
    )


def fit_cylinder(
    points: np.ndarray,
    *,
    distance_threshold: float,
    minimum_inlier_ratio: float = 0.72,
) -> FeatureFit:
    """Fit a closed right cylinder by testing PCA axis candidates robustly."""

    cloud = _points(points)
    if len(cloud) < 12:
        raise ValueError("Cylinder fitting needs at least twelve points.")
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive.")
    centroid = cloud.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov((cloud - centroid).T))
    best: tuple[float, np.ndarray, np.ndarray, float, np.ndarray, float] | None = None
    for axis in eigenvectors.T:
        axis = axis / np.linalg.norm(axis)
        basis_u, basis_v = _perpendicular_basis(axis)
        projected = np.column_stack(((cloud - centroid) @ basis_u, (cloud - centroid) @ basis_v))
        circle = _fit_circle(projected)
        if circle is None:
            continue
        center_2d, radius = circle
        radial = np.linalg.norm(projected - center_2d, axis=1)
        residuals = np.abs(radial - radius)
        mask = residuals <= distance_threshold
        if int(mask.sum()) >= 6:
            refined = least_squares(
                lambda value: (
                    np.linalg.norm(projected[mask] - value[:2], axis=1) - value[2]
                ),
                np.array([center_2d[0], center_2d[1], radius]),
                bounds=([-np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf]),
                loss="soft_l1",
                f_scale=distance_threshold,
            ).x
            center_2d = refined[:2]
            radius = float(refined[2])
            radial = np.linalg.norm(projected - center_2d, axis=1)
            residuals = np.abs(radial - radius)
            mask = residuals <= distance_threshold
        inlier_ratio = float(mask.mean())
        rms = (
            float(np.sqrt(np.mean(np.square(residuals[mask]))))
            if bool(mask.any())
            else float("inf")
        )
        score = inlier_ratio - min(1.0, rms / distance_threshold) * 0.15
        axis_positions = (cloud - centroid) @ axis
        height = float(axis_positions.max() - axis_positions.min())
        center_3d = centroid + center_2d[0] * basis_u + center_2d[1] * basis_v
        if best is None or score > best[0]:
            best = (score, axis.copy(), center_3d, radius, mask, height)
    if best is None:
        return _failed_fit("cylinder", "No stable cylindrical surface was found.")
    _, axis, center, radius, mask, height = best
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    basis_u, basis_v = _perpendicular_basis(axis)
    projected = np.column_stack(((cloud - center) @ basis_u, (cloud - center) @ basis_v))
    residuals = np.abs(np.linalg.norm(projected, axis=1) - radius)
    rms = float(np.sqrt(np.mean(np.square(residuals[mask])))) if bool(mask.any()) else math.inf
    inlier_ratio = float(mask.mean())
    confidence = _confidence(inlier_ratio, rms, distance_threshold)
    aspect_is_sane = radius > distance_threshold * 2 and height > distance_threshold * 4
    accepted = (
        inlier_ratio >= minimum_inlier_ratio and confidence >= 0.80 and aspect_is_sane
    )
    return FeatureFit(
        feature_type="cylinder",
        parameters={
            "axis": axis.tolist(),
            "center": center.tolist(),
            "radius": radius,
            "height": height,
        },
        inlier_ratio=inlier_ratio,
        rms_residual=rms,
        confidence=confidence,
        accepted=accepted,
    )


def fit_axis_aligned_box(
    points: np.ndarray,
    *,
    distance_threshold: float,
    minimum_inlier_ratio: float = 0.88,
) -> FeatureFit:
    cloud = _points(points)
    if len(cloud) < 8:
        raise ValueError("Box fitting needs at least eight points.")
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive.")
    minimum = cloud.min(axis=0)
    maximum = cloud.max(axis=0)
    dimensions = maximum - minimum
    if bool(np.any(dimensions <= distance_threshold * 2)):
        return _failed_fit("box", "The point cloud does not span three box dimensions.")
    face_distances = np.minimum(np.abs(cloud - minimum), np.abs(maximum - cloud))
    residuals = face_distances.min(axis=1)
    mask = residuals <= distance_threshold
    inlier_ratio = float(mask.mean())
    rms = float(np.sqrt(np.mean(np.square(residuals[mask])))) if bool(mask.any()) else math.inf
    confidence = _confidence(inlier_ratio, rms, distance_threshold)
    accepted = inlier_ratio >= minimum_inlier_ratio and confidence >= 0.86
    return FeatureFit(
        feature_type="box",
        parameters={
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "dimensions": dimensions.tolist(),
        },
        inlier_ratio=inlier_ratio,
        rms_residual=rms,
        confidence=confidence,
        accepted=accepted,
    )


def fit_supported_cad(points: np.ndarray, *, distance_threshold: float) -> CadFitResult:
    """Try only solid primitives that can become meaningful, compact CAD."""

    candidates_list: list[FeatureFit] = []
    for fitter in (fit_axis_aligned_box, fit_cylinder):
        try:
            candidates_list.append(fitter(points, distance_threshold=distance_threshold))
        except ValueError as error:
            feature_type: Literal["box", "cylinder"] = (
                "box" if fitter is fit_axis_aligned_box else "cylinder"
            )
            candidates_list.append(_failed_fit(feature_type, str(error)))
    candidates = tuple(candidates_list)
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if not accepted:
        return CadFitResult(
            features=candidates,
            selected=None,
            explanation=(
                "STEP was skipped because the reconstructed surface did not match a supported "
                "box or cylinder with sufficient inliers and low residual error. Mesh outputs "
                "remain available."
            ),
        )
    selected = max(accepted, key=lambda candidate: candidate.confidence)
    return CadFitResult(
        features=candidates,
        selected=selected,
        explanation=(
            f"A {selected.feature_type} fit passed the configured confidence and residual gates."
        ),
    )


def export_fitted_step(
    feature: FeatureFit,
    output_directory: str | Path,
    *,
    scale: ScaleInformation,
    stem: str = "cadpro-fitted-cad",
) -> CadExportResult:
    """Create and reopen a compact analytic solid; never emit faceted STEP."""

    if not feature.accepted:
        raise ValueError("Only an accepted analytic fit can be exported to STEP.")
    if not scale.calibrated or not scale.output_unit:
        raise ValueError("STEP export requires explicit scale calibration.")
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    factor_to_mm = convert_distance(1.0, scale.output_unit, "mm")
    parameters = feature.parameters
    if feature.feature_type == "box":
        minimum = _vector_parameter(parameters, "minimum") * factor_to_mm
        maximum = _vector_parameter(parameters, "maximum") * factor_to_mm
        dimensions = maximum - minimum
        shape = BRepPrimAPI_MakeBox(
            gp_Pnt(*minimum.tolist()), *dimensions.tolist()
        ).Shape()
        script = _box_script(minimum, dimensions)
    elif feature.feature_type == "cylinder":
        axis = _vector_parameter(parameters, "axis")
        center = _vector_parameter(parameters, "center") * factor_to_mm
        radius = _numeric_parameter(parameters, "radius") * factor_to_mm
        height = _numeric_parameter(parameters, "height") * factor_to_mm
        base = center - axis * (height * 0.5)
        frame = gp_Ax2(gp_Pnt(*base.tolist()), gp_Dir(*axis.tolist()))
        shape = BRepPrimAPI_MakeCylinder(frame, radius, height).Shape()
        script = _cylinder_script(base, axis, radius, height)
    else:
        raise ValueError(f"STEP export does not support {feature.feature_type!r} fits.")
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise RuntimeError("The fitted analytic topology failed OpenCascade validation.")
    expected = fingerprint_solid("expected", shape)
    if not math.isfinite(expected.volume) or expected.volume <= 0:
        raise RuntimeError("The fitted analytic topology has no positive finite volume.")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    step_path = write_step(shape, destination / f"{stem}.step")
    reopened = load_step(step_path)
    if len(reopened) != 1 or not BRepCheck_Analyzer(reopened[0][1]).IsValid():
        step_path.unlink(missing_ok=True)
        raise RuntimeError("The fitted STEP file did not reopen as one valid solid.")
    actual = fingerprint_solid(reopened[0][0], reopened[0][1])
    expected_dimensions = np.asarray(expected.bbox_max) - np.asarray(expected.bbox_min)
    actual_dimensions = np.asarray(actual.bbox_max) - np.asarray(actual.bbox_min)
    if actual.volume <= 0 or not math.isclose(actual.volume, expected.volume, rel_tol=1e-7):
        step_path.unlink(missing_ok=True)
        raise RuntimeError("The reopened STEP volume did not match the analytic source.")
    if not np.allclose(actual_dimensions, expected_dimensions, rtol=1e-7, atol=1e-6):
        step_path.unlink(missing_ok=True)
        raise RuntimeError("The reopened STEP dimensions did not match the analytic source.")
    script_path = destination / f"{stem}.py"
    script_path.write_text(script, encoding="utf-8")
    return CadExportResult(step_path, script_path, feature)


def _points(value: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    cloud = np.asarray(value, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError("points must be a finite N x 3 array")
    return cloud


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    matrix = np.column_stack((2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))))
    target = np.square(points[:, 0]) + np.square(points[:, 1])
    try:
        solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center = solution[:2]
    radius_squared = float(solution[2] + np.dot(center, center))
    if radius_squared <= 1e-12 or not math.isfinite(radius_squared):
        return None
    return center, math.sqrt(radius_squared)


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, reference))) > 0.85:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return first, second


def _confidence(inlier_ratio: float, rms: float, threshold: float) -> float:
    residual_score = max(0.0, 1.0 - rms / threshold) if math.isfinite(rms) else 0.0
    return float(np.clip(inlier_ratio * 0.75 + residual_score * 0.25, 0.0, 1.0))


def _failed_fit(
    feature_type: Literal["plane", "cylinder", "box"], reason: str
) -> FeatureFit:
    return FeatureFit(
        feature_type=feature_type,
        parameters={"reason": reason},
        inlier_ratio=0,
        rms_residual=0,
        confidence=0,
        accepted=False,
    )


def _vector_parameter(parameters: dict[str, object], name: str) -> np.ndarray:
    vector = np.asarray(parameters[name], dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"Fitted {name} must be a finite 3D vector.")
    return vector


def _numeric_parameter(parameters: dict[str, object], name: str) -> float:
    value = parameters[name]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Fitted {name} must be a finite number.")
    return float(value)


def _box_script(minimum: np.ndarray, dimensions: np.ndarray) -> str:
    center = minimum + dimensions * 0.5
    return (
        '"""Editable CadQuery reproduction of CadPro\'s accepted box fit (millimetres)."""\n'
        "import cadquery as cq\n\n"
        f"result = cq.Workplane(\"XY\").box({dimensions[0]:.9g}, {dimensions[1]:.9g}, "
        f"{dimensions[2]:.9g}).translate(({center[0]:.9g}, {center[1]:.9g}, "
        f"{center[2]:.9g}))\n"
        'cq.exporters.export(result, "cadpro-fitted-cad.step")\n'
    )


def _cylinder_script(
    base: np.ndarray, axis: np.ndarray, radius: float, height: float
) -> str:
    return (
        '"""Editable CadQuery reproduction of CadPro\'s accepted cylinder fit (millimetres)."""\n'
        "import cadquery as cq\n\n"
        f"base = cq.Vector({base[0]:.9g}, {base[1]:.9g}, {base[2]:.9g})\n"
        f"axis = cq.Vector({axis[0]:.9g}, {axis[1]:.9g}, {axis[2]:.9g})\n"
        f"result = cq.Solid.makeCylinder({radius:.9g}, {height:.9g}, base, axis)\n"
        'cq.exporters.export(result, "cadpro-fitted-cad.step")\n'
    )

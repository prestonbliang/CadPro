from __future__ import annotations

import math

import numpy as np
from OCP.BRepCheck import BRepCheck_Analyzer
import pytest

from cad_diff.signatures import fingerprint_solid
from cad_diff.step_io import load_step
from cadpro.scan.cad_fit import (
    export_fitted_step,
    fit_axis_aligned_box,
    fit_cylinder,
    fit_plane_ransac,
    fit_supported_cad,
)
from cadpro.scan.models import ScaleInformation


def _box_surface(
    minimum: tuple[float, float, float] = (-2.0, -3.0, -4.0),
    maximum: tuple[float, float, float] = (2.0, 3.0, 4.0),
    *,
    samples_per_face: int = 250,
) -> np.ndarray:
    rng = np.random.default_rng(410)
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    faces: list[np.ndarray] = []
    for axis in range(3):
        for coordinate in (low[axis], high[axis]):
            points = rng.uniform(low, high, size=(samples_per_face, 3))
            points[:, axis] = coordinate
            faces.append(points)
    return np.vstack(faces)


def _cylinder_surface(
    *,
    center: tuple[float, float, float] = (1.0, -2.0, 0.4),
    radius: float = 2.0,
    height: float = 10.0,
    noise: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(811)
    angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    axial = np.linspace(-height / 2.0, height / 2.0, 21)
    angle_grid, axial_grid = np.meshgrid(angles, axial)
    radial = radius + rng.normal(0.0, noise, angle_grid.size)
    center_array = np.asarray(center, dtype=np.float64)
    return np.column_stack(
        (
            center_array[0] + radial * np.cos(angle_grid.ravel()),
            center_array[1] + radial * np.sin(angle_grid.ravel()),
            center_array[2]
            + axial_grid.ravel()
            + rng.normal(0.0, noise * 0.5, angle_grid.size),
        )
    )


def _centimeter_scale() -> ScaleInformation:
    return ScaleInformation(
        calibrated=True,
        output_unit="cm",
        scale_factor=1.0,
        calibration_method="two_reconstructed_points",
        user_distance=10.0,
        estimated_uncertainty=0.01,
        warning="Test calibration in centimetres.",
    )


def test_plane_ransac_recovers_plane_despite_outliers():
    rng = np.random.default_rng(741)
    xy = rng.uniform(-4.0, 4.0, size=(320, 2))
    z = 0.25 * xy[:, 0] - 0.12 * xy[:, 1] + 2.0
    z += rng.normal(0.0, 0.003, size=len(xy))
    inliers = np.column_stack((xy, z))
    outliers = rng.uniform((-4.0, -4.0, -5.0), (4.0, 4.0, 8.0), size=(80, 3))

    fit = fit_plane_ransac(
        np.vstack((inliers, outliers)),
        distance_threshold=0.015,
        seed=17,
    )

    expected_normal = np.asarray((-0.25, 0.12, 1.0))
    expected_normal /= np.linalg.norm(expected_normal)
    actual_normal = np.asarray(fit.parameters["normal"])
    assert fit.accepted is True
    assert fit.inlier_ratio == pytest.approx(0.8, abs=0.02)
    assert fit.rms_residual < 0.005
    assert float(np.dot(actual_normal, expected_normal)) > 0.999


def test_noisy_cylinder_is_accepted_with_recovered_dimensions():
    cloud = _cylinder_surface(noise=0.008)

    fit = fit_cylinder(cloud, distance_threshold=0.04)

    assert fit.accepted is True
    assert fit.inlier_ratio > 0.98
    assert fit.rms_residual < 0.015
    assert float(fit.parameters["radius"]) == pytest.approx(2.0, abs=0.02)
    assert float(fit.parameters["height"]) == pytest.approx(10.0, abs=0.05)
    axis = np.asarray(fit.parameters["axis"])
    assert abs(float(np.dot(axis, (0.0, 0.0, 1.0)))) > 0.999


def test_axis_aligned_box_is_accepted_and_selected_for_cad():
    cloud = _box_surface()

    box = fit_axis_aligned_box(cloud, distance_threshold=0.01)
    result = fit_supported_cad(cloud, distance_threshold=0.01)

    assert box.accepted is True
    assert box.inlier_ratio == pytest.approx(1.0)
    assert box.parameters["minimum"] == pytest.approx([-2.0, -3.0, -4.0])
    assert box.parameters["maximum"] == pytest.approx([2.0, 3.0, 4.0])
    assert box.parameters["dimensions"] == pytest.approx([4.0, 6.0, 8.0])
    assert result.selected is not None
    assert result.selected.feature_type == "box"


def test_freeform_volume_is_not_misrepresented_as_supported_cad():
    rng = np.random.default_rng(90210)
    freeform = rng.uniform(-1.0, 1.0, size=(2_000, 3))
    freeform[:, 2] += 0.18 * np.sin(4.0 * freeform[:, 0]) * np.cos(3.0 * freeform[:, 1])

    result = fit_supported_cad(freeform, distance_threshold=0.01)

    assert result.selected is None
    assert all(not feature.accepted for feature in result.features)
    assert {feature.feature_type for feature in result.features} == {"box", "cylinder"}
    assert "STEP was skipped" in result.explanation


def test_calibrated_box_step_reopens_as_valid_metric_ocp_solid(tmp_path):
    cloud = _box_surface(
        minimum=(-1.0, -2.0, 0.0),
        maximum=(1.0, 2.0, 3.0),
    )
    fit = fit_axis_aligned_box(cloud, distance_threshold=0.001)

    exported = export_fitted_step(
        fit,
        tmp_path,
        scale=_centimeter_scale(),
        stem="calibrated-box",
    )

    assert exported.step_path.name == "calibrated-box.step"
    assert exported.script_path.name == "calibrated-box.py"
    (name, shape), = load_step(exported.step_path)
    assert BRepCheck_Analyzer(shape).IsValid()
    fingerprint = fingerprint_solid(name, shape)
    assert fingerprint.volume == pytest.approx(20.0 * 40.0 * 30.0, rel=1e-7)
    dimensions = np.asarray(fingerprint.bbox_max) - np.asarray(fingerprint.bbox_min)
    assert dimensions == pytest.approx([20.0, 40.0, 30.0], abs=1e-5)


def test_calibrated_cylinder_step_reopens_as_valid_metric_ocp_solid(tmp_path):
    cloud = _cylinder_surface(
        center=(2.0, -1.0, 3.0),
        radius=1.5,
        height=5.0,
    )
    fit = fit_cylinder(cloud, distance_threshold=0.001)

    exported = export_fitted_step(
        fit,
        tmp_path,
        scale=_centimeter_scale(),
        stem="calibrated-cylinder",
    )

    assert exported.step_path.name == "calibrated-cylinder.step"
    assert exported.script_path.name == "calibrated-cylinder.py"
    (name, shape), = load_step(exported.step_path)
    assert BRepCheck_Analyzer(shape).IsValid()
    fingerprint = fingerprint_solid(name, shape)
    expected_volume = math.pi * 15.0**2 * 50.0
    assert fingerprint.volume == pytest.approx(expected_volume, rel=1e-7)

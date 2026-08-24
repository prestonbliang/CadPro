"""Explicit two-point scale calibration; no implicit engineering units."""

from __future__ import annotations

import math

import numpy as np

from cadpro.scan.models import ScaleInformation, ScaleMeasurement


_TO_METERS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254}


def calculate_scale(measurement: ScaleMeasurement) -> ScaleInformation:
    point_a = np.asarray(measurement.point_a, dtype=np.float64)
    point_b = np.asarray(measurement.point_b, dtype=np.float64)
    reconstructed_distance = float(np.linalg.norm(point_b - point_a))
    if not math.isfinite(reconstructed_distance) or reconstructed_distance <= 1e-12:
        raise ValueError("Calibration points must be two distinct reconstructed points.")
    scale_factor = measurement.real_distance / reconstructed_distance
    uncertainty = measurement.selection_uncertainty * scale_factor
    return ScaleInformation(
        calibrated=True,
        output_unit=measurement.unit,
        scale_factor=scale_factor,
        calibration_method="two_reconstructed_points",
        user_distance=measurement.real_distance,
        estimated_uncertainty=uncertainty,
        warning=(
            "Scale was set from a user-selected point pair; accuracy depends on reconstruction "
            "quality and point placement."
        ),
    )


def convert_distance(value: float, source_unit: str, destination_unit: str) -> float:
    if source_unit not in _TO_METERS or destination_unit not in _TO_METERS:
        raise ValueError("Units must be one of mm, cm, m, or in.")
    if not math.isfinite(value):
        raise ValueError("Distance must be finite.")
    return value * _TO_METERS[source_unit] / _TO_METERS[destination_unit]


def apply_scale(points: np.ndarray, scale: ScaleInformation) -> np.ndarray:
    if not scale.calibrated or scale.scale_factor is None:
        raise ValueError("Cannot apply an unknown scale.")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("points must be a finite N x 3 array")
    return array * scale.scale_factor

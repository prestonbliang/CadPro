from __future__ import annotations

import math
import subprocess
import sys
import time

import numpy as np
from pydantic import ValidationError
import pytest

from cadpro.scan import capabilities as capability_module
from cadpro.scan.capabilities import detect_toolchain, executable
from cadpro.scan.models import ScaleInformation, ScaleMeasurement
from cadpro.scan.process import (
    CancellationToken,
    ExternalProcessError,
    ProcessCancelled,
    run_process,
)
from cadpro.scan.scale import apply_scale, calculate_scale, convert_distance


_TOOL_PATH_VARIABLES = (
    "CADPRO_FFMPEG_PATH",
    "CADPRO_FFPROBE_PATH",
    "CADPRO_COLMAP_PATH",
    "CADPRO_OPENMVS_INTERFACE_PATH",
    "CADPRO_OPENMVS_DENSIFY_PATH",
    "CADPRO_OPENMVS_RECONSTRUCT_PATH",
    "CADPRO_OPENMVS_REFINE_PATH",
    "CADPRO_OPENMVS_TEXTURE_PATH",
    "CADPRO_BLENDER_PATH",
)


@pytest.fixture
def isolated_capability_environment(monkeypatch, tmp_path):
    for variable in _TOOL_PATH_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(capability_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(capability_module.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "empty-program-files"))


def test_missing_dependencies_are_reported_without_starting_processes(
    isolated_capability_environment, monkeypatch
):
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("a missing dependency must not be executed")

    monkeypatch.setattr(capability_module.subprocess, "run", unexpected_run)

    capabilities = detect_toolchain()

    assert set(capabilities.tools) == {
        "ffmpeg",
        "ffprobe",
        "colmap",
        "interface_colmap",
        "densify_point_cloud",
        "reconstruct_mesh",
        "refine_mesh",
        "texture_mesh",
        "blender",
        "trimesh",
        "ocp",
    }
    assert all(not tool.available for tool in capabilities.tools.values())
    assert capabilities.video_ingest is False
    assert capabilities.photo_reconstruction is False
    assert capabilities.dense_reconstruction is False
    assert capabilities.standard_pipeline_uses_paid_cloud is False
    assert "not found on PATH" in capabilities.tools["ffmpeg"].reason
    assert "https://ffmpeg.org" in capabilities.tools["ffmpeg"].install_hint
    with pytest.raises(RuntimeError, match="FFmpeg is unavailable"):
        executable(capabilities, "ffmpeg")


def test_environment_overrides_are_probed_as_argument_arrays(
    isolated_capability_environment, monkeypatch, tmp_path
):
    fake_tool = tmp_path / "fake ffmpeg executable"
    fake_tool.write_bytes(b"not actually executed")
    monkeypatch.setenv("CADPRO_FFMPEG_PATH", str(fake_tool))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, stdout="ffmpeg fake-version 1.2\n")

    monkeypatch.setattr(capability_module.subprocess, "run", fake_run)

    capabilities = detect_toolchain()

    ffmpeg = capabilities.tools["ffmpeg"]
    assert ffmpeg.available is True
    assert ffmpeg.executable == str(fake_tool.resolve())
    assert ffmpeg.version == "ffmpeg fake-version 1.2"
    assert executable(capabilities, "ffmpeg") == str(fake_tool.resolve())
    assert len(calls) == 1
    arguments, options = calls[0]
    assert arguments == [str(fake_tool.resolve()), "-version"]
    assert options["shell"] is False
    assert options["timeout"] == 8
    assert options["check"] is False


def test_invalid_environment_override_does_not_fall_back_to_path(
    isolated_capability_environment, monkeypatch, tmp_path
):
    missing = tmp_path / "missing-colmap"
    monkeypatch.setenv("CADPRO_COLMAP_PATH", str(missing))

    capabilities = detect_toolchain()

    colmap = capabilities.tools["colmap"]
    assert colmap.available is False
    assert colmap.executable is None
    assert colmap.reason == "CADPRO_COLMAP_PATH does not point to a file."
    assert colmap.install_hint is not None


def test_nonzero_dependency_probe_with_only_loader_error_is_unavailable(
    isolated_capability_environment, monkeypatch, tmp_path
):
    fake_tool = tmp_path / "ffmpeg.exe"
    fake_tool.write_bytes(b"not actually executed")
    monkeypatch.setenv("CADPRO_FFMPEG_PATH", str(fake_tool))
    monkeypatch.setattr(
        capability_module.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments,
            3221225781,
            stdout="VCRUNTIME140.dll was not found",
        ),
    )

    capability = detect_toolchain().tools["ffmpeg"]

    assert capability.available is False
    assert capability.version is None
    assert "exited with code" in capability.reason


def test_run_process_preserves_each_argument_and_captures_output(tmp_path):
    token = "value with spaces; & shell metacharacters"
    log_path = tmp_path / "logs" / "success.log"

    result = run_process(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", token],
        working_directory=tmp_path,
        log_path=log_path,
        timeout_seconds=5,
        cancellation=CancellationToken(),
    )

    assert result.return_code == 0
    assert result.arguments[-1] == token
    assert result.log_path == log_path
    assert token in result.output_tail
    assert token in log_path.read_text(encoding="utf-8")


def test_run_process_reports_nonzero_exit_and_log_tail(tmp_path):
    with pytest.raises(ExternalProcessError) as captured:
        run_process(
            [
                sys.executable,
                "-c",
                "import sys; print('native failure', file=sys.stderr); raise SystemExit(7)",
            ],
            working_directory=tmp_path,
            log_path=tmp_path / "nonzero.log",
            timeout_seconds=5,
            cancellation=CancellationToken(),
        )

    assert captured.value.return_code == 7
    assert "exited with code 7" in str(captured.value)
    assert "native failure" in captured.value.tail


def test_run_process_enforces_timeout(tmp_path):
    started = time.monotonic()
    with pytest.raises(ExternalProcessError, match="exceeded the 0.3-second timeout") as captured:
        run_process(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(10)",
            ],
            working_directory=tmp_path,
            log_path=tmp_path / "timeout.log",
            timeout_seconds=0.3,
            cancellation=CancellationToken(),
        )

    assert time.monotonic() - started < 5
    assert "started" in captured.value.tail


def test_run_process_honors_cancellation_probe(tmp_path):
    cancel_after = time.monotonic() + 0.2
    cancellation = CancellationToken(probe=lambda: time.monotonic() >= cancel_after)

    with pytest.raises(ProcessCancelled, match="Cancelled"):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            working_directory=tmp_path,
            log_path=tmp_path / "cancel.log",
            timeout_seconds=5,
            cancellation=cancellation,
        )

    assert cancellation.cancelled is True


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -math.inf])
def test_scale_measurement_rejects_nonfinite_calibration_points(coordinate):
    with pytest.raises(ValidationError, match="finite coordinates"):
        ScaleMeasurement(
            point_a=(coordinate, 0.0, 0.0),
            point_b=(1.0, 0.0, 0.0),
            real_distance=1.0,
            unit="mm",
        )


def test_scale_measurement_forbids_unexpected_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ScaleMeasurement(
            point_a=(0.0, 0.0, 0.0),
            point_b=(1.0, 0.0, 0.0),
            real_distance=1.0,
            unit="mm",
            guessed_scale=True,
        )


def test_two_point_scale_calculates_factor_and_propagates_uncertainty():
    measurement = ScaleMeasurement(
        point_a=(0.0, 0.0, 0.0),
        point_b=(3.0, 4.0, 0.0),
        real_distance=10.0,
        unit="cm",
        selection_uncertainty=0.25,
    )

    scale = calculate_scale(measurement)

    assert scale.calibrated is True
    assert scale.output_unit == "cm"
    assert scale.scale_factor == pytest.approx(2.0)
    assert scale.user_distance == pytest.approx(10.0)
    assert scale.estimated_uncertainty == pytest.approx(0.5)
    assert scale.calibration_method == "two_reconstructed_points"


def test_two_point_scale_rejects_coincident_points():
    measurement = ScaleMeasurement(
        point_a=(2.0, 3.0, 4.0),
        point_b=(2.0, 3.0, 4.0),
        real_distance=25.0,
        unit="mm",
    )

    with pytest.raises(ValueError, match="two distinct reconstructed points"):
        calculate_scale(measurement)


@pytest.mark.parametrize(
    ("value", "source_unit", "destination_unit", "expected"),
    [
        (25.4, "mm", "in", 1.0),
        (1.0, "in", "mm", 25.4),
        (100.0, "cm", "m", 1.0),
        (2.0, "m", "cm", 200.0),
    ],
)
def test_convert_distance_supports_engineering_units(
    value, source_unit, destination_unit, expected
):
    assert convert_distance(value, source_unit, destination_unit) == pytest.approx(expected)


def test_scale_helpers_reject_unknown_units_and_nonfinite_values():
    with pytest.raises(ValueError, match="Units must be one of"):
        convert_distance(1.0, "feet", "mm")
    with pytest.raises(ValueError, match="Distance must be finite"):
        convert_distance(math.inf, "mm", "m")


def test_apply_scale_requires_known_scale_and_finite_xyz_points():
    scale = ScaleInformation(calibrated=True, output_unit="mm", scale_factor=2.5)
    points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])

    scaled = apply_scale(points, scale)

    assert np.array_equal(scaled, points * 2.5)
    assert np.array_equal(points, np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]]))
    with pytest.raises(ValueError, match="unknown scale"):
        apply_scale(points, ScaleInformation())
    with pytest.raises(ValueError, match="finite N x 3"):
        apply_scale(np.asarray([[math.nan, 0.0, 0.0]]), scale)

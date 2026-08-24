from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from cadpro.scan import photogrammetry as photogrammetry_module
from cadpro.scan import video as video_module
from cadpro.scan.models import (
    InputMode,
    QualityPreset,
    ScanConfiguration,
    ToolCapability,
    ToolchainCapabilities,
    VideoSelectionSettings,
)
from cadpro.scan.photogrammetry import ColmapOpenMvsAdapter
from cadpro.scan.process import (
    CancellationToken,
    CapturedProcessResult,
    ExternalProcessError,
    ProcessResult,
)


_EXECUTABLE_NAMES = {
    "ffmpeg": "ffmpeg.exe",
    "ffprobe": "ffprobe.exe",
    "colmap": "colmap.exe",
    "interface_colmap": "InterfaceCOLMAP.exe",
    "densify_point_cloud": "DensifyPointCloud.exe",
    "reconstruct_mesh": "ReconstructMesh.exe",
    "refine_mesh": "RefineMesh.exe",
    "texture_mesh": "TextureMesh.exe",
}


def _capabilities(
    tmp_path: Path,
    *,
    openmvs: bool,
    colmap: bool = True,
    ffmpeg: bool = True,
    ffprobe: bool = True,
) -> ToolchainCapabilities:
    availability = {
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "colmap": colmap,
        "interface_colmap": openmvs,
        "densify_point_cloud": openmvs,
        "reconstruct_mesh": openmvs,
        "refine_mesh": openmvs,
        "texture_mesh": openmvs,
    }
    tools: dict[str, ToolCapability] = {}
    for key, executable_name in _EXECUTABLE_NAMES.items():
        available = availability[key]
        tools[key] = ToolCapability(
            name=key,
            available=available,
            executable=str((tmp_path / "native tools" / executable_name).resolve())
            if available
            else None,
            version="test-version" if available else None,
            reason=None if available else f"{key} is absent",
            install_hint=None if available else f"Install {key}.",
        )
    return ToolchainCapabilities(
        tools=tools,
        photo_reconstruction=colmap,
        video_ingest=ffmpeg and ffprobe,
        dense_reconstruction=colmap or openmvs,
        texture_generation=openmvs,
        mesh_processing=True,
        analytic_cad=True,
    )


def _executable(capabilities: ToolchainCapabilities, key: str) -> str:
    value = capabilities.tools[key].executable
    assert value is not None
    return value


def _source_images(tmp_path: Path, count: int) -> tuple[Path, ...]:
    source = tmp_path / "uploads"
    source.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = source / f"view-{index + 1:02d}.jpg"
        path.write_bytes(f"fake-jpeg-{index}".encode())
        paths.append(path)
    return tuple(paths)


def _argument_after(arguments: Sequence[str], option: str) -> str:
    index = arguments.index(option)
    return arguments[index + 1]


def _publish(path: str | Path, payload: bytes = b"native-artifact") -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


class _NativeHarness:
    def __init__(
        self,
        capabilities: ToolchainCapabilities,
        components: dict[str, tuple[int, int, float]],
        *,
        missing_help: set[str] | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.components = components
        self.missing_help = missing_help or set()
        self.run_calls: list[tuple[str, ...]] = []
        self.capture_calls: list[tuple[str, ...]] = []

    def run_process(self, arguments, **kwargs) -> ProcessResult:
        normalized = tuple(str(argument) for argument in arguments)
        self.run_calls.append(normalized)
        executable_name = Path(normalized[0]).stem.lower()

        if normalized[0] == _executable(self.capabilities, "colmap"):
            command = normalized[1]
            if command == "mapper":
                sparse = Path(_argument_after(normalized, "--output_path"))
                for component in self.components:
                    (sparse / component).mkdir(parents=True, exist_ok=True)
            elif command == "model_converter":
                _publish(_argument_after(normalized, "--output_path"))
            elif command == "image_undistorter":
                dense = Path(_argument_after(normalized, "--output_path"))
                (dense / "images").mkdir(parents=True, exist_ok=True)
            elif command == "stereo_fusion":
                _publish(_argument_after(normalized, "--output_path"))
            elif command in {"poisson_mesher", "delaunay_mesher"}:
                _publish(_argument_after(normalized, "--output_path"))
        elif executable_name == "interfacecolmap":
            _publish(_argument_after(normalized, "-o"))
        elif executable_name == "densifypointcloud":
            output = Path(_argument_after(normalized, "-o"))
            _publish(output)
            _publish(output.with_suffix(".ply"))
        elif executable_name == "reconstructmesh":
            output = Path(_argument_after(normalized, "-o"))
            _publish(output)
            _publish(output.with_suffix(".ply"))
        elif executable_name == "refinemesh":
            output = Path(_argument_after(normalized, "-o"))
            _publish(output)
            _publish(output.with_suffix(".ply"))
        elif executable_name == "texturemesh":
            output = Path(_argument_after(normalized, "-o"))
            export_type = _argument_after(normalized, "--export-type")
            _publish(output.with_suffix(f".{export_type}"))
            if export_type == "obj":
                _publish(output.with_suffix(".mtl"), b"newmtl material\nmap_Kd atlas.png\n")
                _publish(output.parent / "atlas.png")

        return ProcessResult(
            normalized,
            0,
            0.01,
            Path(kwargs["log_path"]),
            "",
        )

    def run_process_capture(self, arguments, **kwargs) -> CapturedProcessResult:
        normalized = tuple(str(argument) for argument in arguments)
        self.capture_calls.append(normalized)
        command = normalized[1]
        if command in self.missing_help:
            raise ExternalProcessError(
                f"missing {command}",
                return_code=1,
                tail="unknown command",
            )
        if command == "model_analyzer" and "--path" in normalized:
            component = Path(_argument_after(normalized, "--path")).name
            registered, points, error = self.components[component]
            stdout = (
                f"Registered images: {registered}\n"
                f"Points: {points}\n"
                f"Mean reprojection error: {error}px\n"
            ).encode()
        else:
            stdout = f"{command}\nOptions:\n".encode()
        process = ProcessResult(
            normalized,
            0,
            0.01,
            Path(kwargs["log_path"]),
            "",
        )
        return CapturedProcessResult(process, stdout)


def _install_native_harness(monkeypatch, harness: _NativeHarness) -> None:
    monkeypatch.setattr(photogrammetry_module, "run_process", harness.run_process)
    monkeypatch.setattr(
        photogrammetry_module,
        "run_process_capture",
        harness.run_process_capture,
    )


def test_cpu_openmvs_pipeline_uses_exact_argv_and_best_component(monkeypatch, tmp_path):
    capabilities = _capabilities(tmp_path, openmvs=True)
    harness = _NativeHarness(
        capabilities,
        {
            "0": (3, 900, 0.9),
            "7": (5, 700, 0.4),
        },
    )
    _install_native_harness(monkeypatch, harness)
    work = (tmp_path / "cpu job").resolve()
    configuration = ScanConfiguration(
        mode=InputMode.PHOTOS,
        quality_preset=QualityPreset.BALANCED,
        use_gpu=False,
    )

    products = ColmapOpenMvsAdapter(capabilities).reconstruct(
        _source_images(tmp_path, 6),
        work,
        configuration=configuration,
        cancellation=CancellationToken(),
        progress=lambda _stage, _percent: None,
    )

    colmap = _executable(capabilities, "colmap")
    images = work / "images"
    sparse = work / "sparse"
    dense = work / "dense"
    database = work / "database.db"
    selected = sparse / "7"
    native = work / "native-textured"
    expected = (
        (
            colmap,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--FeatureExtraction.max_image_size",
            "2400",
            "--FeatureExtraction.use_gpu",
            "0",
        ),
        (
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.use_gpu",
            "0",
        ),
        (
            colmap,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ),
        (colmap, "model_analyzer", "--path", str(sparse / "0")),
        (colmap, "model_analyzer", "--path", str(sparse / "7")),
        (
            colmap,
            "model_converter",
            "--input_path",
            str(selected),
            "--output_path",
            str(work / "sparse.ply"),
            "--output_type",
            "PLY",
        ),
        (
            colmap,
            "image_undistorter",
            "--image_path",
            str(images),
            "--input_path",
            str(selected),
            "--output_path",
            str(dense),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            "2000",
        ),
        (
            _executable(capabilities, "interface_colmap"),
            "-i",
            str(dense),
            "-o",
            str(work / "scene.mvs"),
            "--image-folder",
            str(dense / "images"),
        ),
        (
            _executable(capabilities, "densify_point_cloud"),
            str(work / "scene.mvs"),
            "-o",
            str(work / "scene_dense.mvs"),
        ),
        (
            _executable(capabilities, "reconstruct_mesh"),
            str(work / "scene_dense.mvs"),
            "-p",
            str(work / "scene_dense.ply"),
            "-o",
            str(work / "scene_dense_mesh.mvs"),
        ),
        (
            _executable(capabilities, "refine_mesh"),
            str(work / "scene_dense.mvs"),
            "-m",
            str(work / "scene_dense_mesh.ply"),
            "-o",
            str(work / "scene_dense_mesh_refine.mvs"),
            "--scales",
            "1",
            "--max-face-area",
            "16",
        ),
        (
            _executable(capabilities, "texture_mesh"),
            str(work / "scene_dense.mvs"),
            "-m",
            str(work / "scene_dense_mesh_refine.ply"),
            "-o",
            str(native / "model_glb.mvs"),
            "--export-type",
            "glb",
        ),
        (
            _executable(capabilities, "texture_mesh"),
            str(work / "scene_dense.mvs"),
            "-m",
            str(work / "scene_dense_mesh_refine.ply"),
            "-o",
            str(native / "model_obj.mvs"),
            "--export-type",
            "obj",
        ),
    )
    assert products.commands == expected
    assert products.registered_cameras == 5
    assert products.sparse_points == 700
    assert products.reprojection_error_px == pytest.approx(0.4)
    assert products.mesh_path == work / "scene_dense_mesh_refine.ply"
    assert products.textured_glb_path == native / "model_glb.glb"
    assert products.textured_obj_path == native / "model_obj.obj"
    assert all("patch_match_stereo" not in command for command in products.commands)
    assert str(selected) in products.commands[5]
    assert str(sparse / "0") not in products.commands[5]


def test_gpu_colmap_pipeline_uses_geometric_patchmatch_and_poisson(monkeypatch, tmp_path):
    capabilities = _capabilities(tmp_path, openmvs=False)
    harness = _NativeHarness(capabilities, {"3": (6, 1_250, 0.35)})
    _install_native_harness(monkeypatch, harness)
    work = (tmp_path / "gpu job").resolve()
    configuration = ScanConfiguration(
        mode=InputMode.PHOTOS,
        quality_preset=QualityPreset.DRAFT,
        mesher="poisson",
        use_gpu=True,
    )

    products = ColmapOpenMvsAdapter(capabilities).reconstruct(
        _source_images(tmp_path, 6),
        work,
        configuration=configuration,
        cancellation=CancellationToken(),
        progress=lambda _stage, _percent: None,
    )

    colmap = _executable(capabilities, "colmap")
    images = work / "images"
    sparse = work / "sparse"
    dense = work / "dense"
    database = work / "database.db"
    assert products.commands == (
        (
            colmap,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--FeatureExtraction.max_image_size",
            "1600",
            "--FeatureExtraction.use_gpu",
            "1",
        ),
        (
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.use_gpu",
            "1",
        ),
        (
            colmap,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ),
        (colmap, "model_analyzer", "--path", str(sparse / "3")),
        (
            colmap,
            "model_converter",
            "--input_path",
            str(sparse / "3"),
            "--output_path",
            str(work / "sparse.ply"),
            "--output_type",
            "PLY",
        ),
        (
            colmap,
            "image_undistorter",
            "--image_path",
            str(images),
            "--input_path",
            str(sparse / "3"),
            "--output_path",
            str(dense),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            "1200",
        ),
        (
            colmap,
            "patch_match_stereo",
            "--workspace_path",
            str(dense),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ),
        (
            colmap,
            "stereo_fusion",
            "--workspace_path",
            str(dense),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(dense / "fused.ply"),
        ),
        (
            colmap,
            "poisson_mesher",
            "--input_path",
            str(dense / "fused.ply"),
            "--output_path",
            str(dense / "meshed-poisson.ply"),
        ),
    )
    assert products.dense_cloud_path == dense / "fused.ply"
    assert products.mesh_path == dense / "meshed-poisson.ply"
    assert products.textured_glb_path is None
    assert products.textured_obj_path is None


def test_cpu_dense_reconstruction_requires_openmvs_before_running_colmap(
    monkeypatch, tmp_path
):
    capabilities = _capabilities(tmp_path, openmvs=False)
    monkeypatch.setattr(
        photogrammetry_module,
        "run_process",
        lambda *_args, **_kwargs: pytest.fail("no native process should start"),
    )
    monkeypatch.setattr(
        photogrammetry_module,
        "run_process_capture",
        lambda *_args, **_kwargs: pytest.fail("no native capability probe should start"),
    )

    with pytest.raises(RuntimeError, match="install OpenMVS for CPU dense reconstruction"):
        ColmapOpenMvsAdapter(capabilities).reconstruct(
            _source_images(tmp_path, 3),
            tmp_path / "cpu without openmvs",
            configuration=ScanConfiguration(mode=InputMode.PHOTOS, use_gpu=False),
            cancellation=CancellationToken(),
            progress=lambda _stage, _percent: None,
        )


def test_missing_colmap_subcommand_is_actionable_and_stops_before_reconstruction(
    monkeypatch, tmp_path
):
    capabilities = _capabilities(tmp_path, openmvs=False)
    harness = _NativeHarness(
        capabilities,
        {"0": (4, 400, 0.5)},
        missing_help={"patch_match_stereo"},
    )
    _install_native_harness(monkeypatch, harness)

    with pytest.raises(RuntimeError, match=r"lacks required command\(s\): patch_match_stereo"):
        ColmapOpenMvsAdapter(capabilities).reconstruct(
            _source_images(tmp_path, 4),
            tmp_path / "missing command",
            configuration=ScanConfiguration(mode=InputMode.PHOTOS, use_gpu=True),
            cancellation=CancellationToken(),
            progress=lambda _stage, _percent: None,
        )

    assert harness.run_calls == []
    probed = [arguments[1] for arguments in harness.capture_calls]
    assert "patch_match_stereo" in probed
    assert all(arguments[-1] == "-h" for arguments in harness.capture_calls)


def _captured_json(arguments, kwargs, document: dict[str, object]) -> CapturedProcessResult:
    normalized = tuple(str(argument) for argument in arguments)
    process = ProcessResult(
        normalized,
        0,
        0.01,
        Path(kwargs["log_path"]),
        "",
    )
    return CapturedProcessResult(process, json.dumps(document).encode())


def _video_document(
    *,
    stream_duration: str | None = None,
    format_duration: str = "12.5",
    average_rate: str = "30000/1001",
    real_rate: str = "30000/1001",
) -> dict[str, object]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p",
                "avg_frame_rate": average_rate,
                "r_frame_rate": real_rate,
                "nb_frames": "N/A",
                "duration": stream_duration,
            }
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": format_duration,
            "size": "123456",
        },
    }


def test_ffprobe_uses_exact_json_argv_and_parses_format_duration(monkeypatch, tmp_path):
    capabilities = _capabilities(tmp_path, openmvs=False)
    source = tmp_path / "orbit video.mp4"
    source.write_bytes(b"video")
    calls: list[tuple[str, ...]] = []

    def fake_capture(arguments, **kwargs):
        calls.append(tuple(str(argument) for argument in arguments))
        return _captured_json(arguments, kwargs, _video_document())

    monkeypatch.setattr(video_module, "run_process_capture", fake_capture)
    metadata, arguments = video_module.inspect_video(
        source,
        capabilities=capabilities,
        settings=VideoSelectionSettings(maximum_duration_seconds=30),
        working_directory=tmp_path,
        log_path=tmp_path / "ffprobe.log",
        cancellation=CancellationToken(),
    )

    expected = (
        _executable(capabilities, "ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=format_name,duration,size:stream=index,codec_name,codec_type,width,height,"
        "pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(source.resolve()),
    )
    assert arguments == expected
    assert calls == [expected]
    assert metadata.codec == "h264"
    assert metadata.duration_seconds == pytest.approx(12.5)
    assert metadata.frame_rate == pytest.approx(30000 / 1001)
    assert metadata.frame_count is None
    assert metadata.size_bytes == 123456


def test_ffprobe_treats_na_stream_duration_as_missing(monkeypatch, tmp_path):
    capabilities = _capabilities(tmp_path, openmvs=False)
    source = tmp_path / "duration-na.mp4"
    source.write_bytes(b"video")

    monkeypatch.setattr(
        video_module,
        "run_process_capture",
        lambda arguments, **kwargs: _captured_json(
            arguments,
            kwargs,
            _video_document(stream_duration="N/A", format_duration="8.75"),
        ),
    )

    metadata, _arguments = video_module.inspect_video(
        source,
        capabilities=capabilities,
        settings=VideoSelectionSettings(maximum_duration_seconds=30),
        working_directory=tmp_path,
        log_path=tmp_path / "duration-na.log",
        cancellation=CancellationToken(),
    )

    assert metadata.duration_seconds == pytest.approx(8.75)


def test_ffprobe_falls_back_from_zero_average_rate_to_real_rate(monkeypatch, tmp_path):
    capabilities = _capabilities(tmp_path, openmvs=False)
    source = tmp_path / "rate-fallback.mp4"
    source.write_bytes(b"video")

    monkeypatch.setattr(
        video_module,
        "run_process_capture",
        lambda arguments, **kwargs: _captured_json(
            arguments,
            kwargs,
            _video_document(average_rate="0/0", real_rate="30/1"),
        ),
    )

    metadata, _arguments = video_module.inspect_video(
        source,
        capabilities=capabilities,
        settings=VideoSelectionSettings(maximum_duration_seconds=30),
        working_directory=tmp_path,
        log_path=tmp_path / "rate-fallback.log",
        cancellation=CancellationToken(),
    )

    assert metadata.frame_rate == pytest.approx(30.0)


def test_ffprobe_enforces_configured_duration_limit(monkeypatch, tmp_path):
    capabilities = _capabilities(tmp_path, openmvs=False)
    source = tmp_path / "too-long.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        video_module,
        "run_process_capture",
        lambda arguments, **kwargs: _captured_json(
            arguments,
            kwargs,
            _video_document(format_duration="31.0"),
        ),
    )

    with pytest.raises(ValueError, match=r"31\.0s exceeds the configured 30s limit"):
        video_module.inspect_video(
            source,
            capabilities=capabilities,
            settings=VideoSelectionSettings(maximum_duration_seconds=30),
            working_directory=tmp_path,
            log_path=tmp_path / "too-long.log",
            cancellation=CancellationToken(),
        )


def test_ffmpeg_candidate_extraction_uses_timestamp_select_and_safe_output_flags(
    monkeypatch, tmp_path
):
    capabilities = _capabilities(tmp_path, openmvs=False)
    source = tmp_path / "object orbit; no shell.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "candidate frames"
    calls: list[tuple[str, ...]] = []

    def fake_run(arguments, **kwargs):
        normalized = tuple(str(argument) for argument in arguments)
        calls.append(normalized)
        output.mkdir(parents=True, exist_ok=True)
        for index in range(1, 9):
            (output / f"candidate-{index:06d}.jpg").write_bytes(b"jpeg")
        return ProcessResult(
            normalized,
            0,
            0.01,
            Path(kwargs["log_path"]),
            "",
        )

    monkeypatch.setattr(video_module, "run_process", fake_run)
    settings = VideoSelectionSettings(
        maximum_duration_seconds=30,
        candidate_frames_per_second=2.0,
        maximum_candidate_frames=12,
        target_frames=8,
    )
    candidates, result = video_module.extract_video_candidates(
        source,
        output,
        capabilities=capabilities,
        settings=settings,
        working_directory=tmp_path,
        log_path=tmp_path / "ffmpeg.log",
        cancellation=CancellationToken(),
        maximum_edge=1200,
    )

    expected = (
        _executable(capabilities, "ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source.resolve()),
        "-map",
        "0:v:0",
        "-vf",
        "select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,0.5),"
        "scale=1200:-2:force_original_aspect_ratio=decrease",
        "-fps_mode",
        "vfr",
        "-frames:v",
        "12",
        "-q:v",
        "2",
        "-n",
        str(output / "candidate-%06d.jpg"),
    )
    assert calls == [expected]
    assert result.arguments == expected
    assert len(candidates) == 8
    assert "object orbit; no shell.mp4" in result.arguments[6]

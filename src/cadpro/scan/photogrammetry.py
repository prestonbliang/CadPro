"""Replaceable real COLMAP/OpenMVS adapter plus an explicitly injected test double."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable, Protocol, Sequence

import numpy as np
import trimesh

from cadpro.scan.capabilities import executable
from cadpro.scan.models import JobStage, QualityPreset, ScanConfiguration, ToolchainCapabilities
from cadpro.scan.process import (
    CancellationToken,
    ExternalProcessError,
    ProcessResult,
    run_process,
    run_process_capture,
)


ProgressCallback = Callable[[JobStage, int], None]


@dataclass(frozen=True)
class PhotogrammetryProducts:
    sparse_cloud_path: Path
    dense_cloud_path: Path
    mesh_path: Path
    textured_glb_path: Path | None
    textured_obj_path: Path | None
    texture_resources: tuple[Path, ...]
    registered_cameras: int
    sparse_points: int
    reprojection_error_px: float | None
    commands: tuple[tuple[str, ...], ...]
    tool_versions: dict[str, str]
    log_paths: tuple[Path, ...]
    warnings: tuple[str, ...] = ()


class PhotogrammetryAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def reconstruct(
        self,
        image_paths: Sequence[Path],
        working_directory: Path,
        *,
        configuration: ScanConfiguration,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> PhotogrammetryProducts: ...


class ColmapOpenMvsAdapter:
    """Run the official staged local pipeline; never falls back to silhouettes or cloud AI."""

    def __init__(self, capabilities: ToolchainCapabilities) -> None:
        self.capabilities = capabilities

    @property
    def name(self) -> str:
        return "colmap-openmvs"

    def reconstruct(
        self,
        image_paths: Sequence[Path],
        working_directory: Path,
        *,
        configuration: ScanConfiguration,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> PhotogrammetryProducts:
        if len(image_paths) < 3:
            raise ValueError("Real photogrammetry needs at least three accepted overlapping images.")
        colmap = _colmap_executable(executable(self.capabilities, "colmap"))
        work = working_directory.resolve()
        work.mkdir(parents=True, exist_ok=True)
        images = work / "images"
        sparse = work / "sparse"
        dense = work / "dense"
        logs = work / "logs"
        for directory in (images, sparse, dense, logs):
            directory.mkdir(parents=True, exist_ok=True)
        _link_or_copy_images(image_paths, images)
        log_path = logs / "colmap.log"
        commands: list[tuple[str, ...]] = []

        required = [
            "feature_extractor",
            f"{configuration.feature_matcher}_matcher",
            "mapper",
            "model_analyzer",
            "model_converter",
            "image_undistorter",
        ]
        use_openmvs = _openmvs_ready(self.capabilities)
        if not use_openmvs:
            if not configuration.use_gpu:
                raise RuntimeError(
                    "Dense reconstruction is unavailable: OpenMVS is not installed and COLMAP "
                    "PatchMatch requires a supported GPU build. Enable GPU only with a compatible "
                    "COLMAP build, or install OpenMVS for CPU dense reconstruction."
                )
            required.extend(
                ["patch_match_stereo", "stereo_fusion", f"{configuration.mesher}_mesher"]
            )
        _require_colmap_commands(
            colmap,
            required,
            work,
            logs / "colmap-capabilities.log",
            cancellation,
        )

        database = work / "database.db"
        feature = [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--FeatureExtraction.max_image_size",
            str(_feature_image_size(configuration.quality_preset)),
            "--FeatureExtraction.use_gpu",
            "1" if configuration.use_gpu else "0",
        ]
        _run(feature, work, log_path, 3_600, cancellation, commands)
        progress(JobStage.ESTIMATING_CAMERAS, 35)

        matcher = [
            colmap,
            f"{configuration.feature_matcher}_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.use_gpu",
            "1" if configuration.use_gpu else "0",
        ]
        _run(matcher, work, log_path, 3_600, cancellation, commands)
        mapper = [
            colmap,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ]
        _run(mapper, work, log_path, 7_200, cancellation, commands)
        progress(JobStage.ESTIMATING_CAMERAS, 48)

        components = _analyze_components(colmap, sparse, work, log_path, cancellation, commands)
        if not components:
            raise RuntimeError(
                "COLMAP could not register a connected camera model. Increase overlap, avoid blur "
                "and reflections, and include trackable background detail."
            )
        selected = max(components, key=lambda item: (item.registered_images, item.points))
        required_registered = max(3, math.ceil(len(image_paths) * 0.5))
        if selected.registered_images < required_registered:
            raise RuntimeError(
                f"COLMAP registered only {selected.registered_images}/{len(image_paths)} cameras; "
                f"at least {required_registered} are required for dense reconstruction."
            )
        sparse_ply = work / "sparse.ply"
        convert = [
            colmap,
            "model_converter",
            "--input_path",
            str(selected.path),
            "--output_path",
            str(sparse_ply),
            "--output_type",
            "PLY",
        ]
        _run(convert, work, log_path, 600, cancellation, commands)
        if not sparse_ply.is_file() or sparse_ply.stat().st_size == 0:
            raise RuntimeError("COLMAP did not publish the sparse PLY point cloud.")

        undistort = [
            colmap,
            "image_undistorter",
            "--image_path",
            str(images),
            "--input_path",
            str(selected.path),
            "--output_path",
            str(dense),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(_dense_image_size(configuration.quality_preset)),
        ]
        _run(undistort, work, log_path, 1_800, cancellation, commands)
        progress(JobStage.BUILDING_DENSE_CLOUD, 55)

        log_paths: tuple[Path, ...]
        if use_openmvs:
            native = self._run_openmvs(dense, work, logs, cancellation, progress, commands)
            dense_cloud = native.dense_cloud
            mesh = native.mesh
            textured_glb = native.textured_glb
            textured_obj = native.textured_obj
            texture_resources = native.texture_resources
            tool_versions = {
                "COLMAP": self.capabilities.tools["colmap"].version or "detected",
                "OpenMVS": self.capabilities.tools["reconstruct_mesh"].version or "detected",
            }
            warnings: tuple[str, ...] = (
                ()
                if native.textured_obj is not None
                else (
                    "OpenMVS geometry reconstruction completed without TextureMesh; GLB and OBJ "
                    "will be exported without claiming camera-projected textures.",
                )
            )
            log_paths = (log_path, native.log_path)
        else:
            patch_match = [
                colmap,
                "patch_match_stereo",
                "--workspace_path",
                str(dense),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
            ]
            try:
                _run(patch_match, work, log_path, 14_400, cancellation, commands)
            except ExternalProcessError as error:
                raise RuntimeError(
                    "COLMAP PatchMatch dense reconstruction failed. Confirm that this COLMAP "
                    "build supports MVS and that its CUDA/accelerator runtime is available. "
                    f"Native exit status: {error.return_code}."
                ) from error
            fusion = [
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
            ]
            _run(fusion, work, log_path, 7_200, cancellation, commands)
            progress(JobStage.BUILDING_MESH, 68)
            mesh_path = dense / f"meshed-{configuration.mesher}.ply"
            if configuration.mesher == "poisson":
                mesher = [
                    colmap,
                    "poisson_mesher",
                    "--input_path",
                    str(dense / "fused.ply"),
                    "--output_path",
                    str(mesh_path),
                ]
            else:
                mesher = [
                    colmap,
                    "delaunay_mesher",
                    "--input_path",
                    str(dense),
                    "--output_path",
                    str(mesh_path),
                ]
            _run(mesher, work, log_path, 7_200, cancellation, commands)
            dense_cloud = dense / "fused.ply"
            mesh = mesh_path
            textured_glb = None
            textured_obj = None
            texture_resources = ()
            tool_versions = {
                "COLMAP": self.capabilities.tools["colmap"].version or "detected"
            }
            warnings = (
                "OpenMVS texture generation is unavailable; GLB and OBJ will be exported without "
                "claiming camera-projected textures.",
            )
            log_paths = (log_path,)
        for required_path, label in (
            (dense_cloud, "dense PLY point cloud"),
            (mesh, "triangle mesh"),
        ):
            if not required_path.is_file() or required_path.stat().st_size == 0:
                raise RuntimeError(f"The native pipeline produced no {label}.")
        discarded = len(components) - 1
        if discarded:
            warnings = (
                *warnings,
                f"COLMAP produced {len(components)} disconnected camera components; the component "
                f"with {selected.registered_images} registered images was selected.",
            )
        return PhotogrammetryProducts(
            sparse_cloud_path=sparse_ply,
            dense_cloud_path=dense_cloud,
            mesh_path=mesh,
            textured_glb_path=textured_glb,
            textured_obj_path=textured_obj,
            texture_resources=texture_resources,
            registered_cameras=selected.registered_images,
            sparse_points=selected.points,
            reprojection_error_px=selected.reprojection_error,
            commands=tuple(commands),
            tool_versions=tool_versions,
            log_paths=log_paths,
            warnings=warnings,
        )

    def _run_openmvs(
        self,
        colmap_dense: Path,
        work: Path,
        logs: Path,
        cancellation: CancellationToken,
        progress: ProgressCallback,
        commands: list[tuple[str, ...]],
    ) -> _OpenMvsProducts:
        interface = executable(self.capabilities, "interface_colmap")
        densify = executable(self.capabilities, "densify_point_cloud")
        reconstruct = executable(self.capabilities, "reconstruct_mesh")
        refine = executable(self.capabilities, "refine_mesh")
        log_path = logs / "openmvs.log"
        scene = work / "scene.mvs"
        scene_dense = work / "scene_dense.mvs"
        dense_ply = work / "scene_dense.ply"
        scene_mesh = work / "scene_dense_mesh.mvs"
        mesh_ply = work / "scene_dense_mesh.ply"
        scene_refined = work / "scene_dense_mesh_refine.mvs"
        refined_ply = work / "scene_dense_mesh_refine.ply"
        native_output = work / "native-textured"
        native_output.mkdir(parents=True, exist_ok=True)
        _run(
            [
                interface,
                "-i",
                str(colmap_dense),
                "-o",
                str(scene),
                "--image-folder",
                str(colmap_dense / "images"),
            ],
            work,
            log_path,
            1_800,
            cancellation,
            commands,
        )
        _run(
            [densify, str(scene), "-o", str(scene_dense)],
            work,
            log_path,
            14_400,
            cancellation,
            commands,
        )
        progress(JobStage.BUILDING_MESH, 68)
        _run(
            [reconstruct, str(scene_dense), "-p", str(dense_ply), "-o", str(scene_mesh)],
            work,
            log_path,
            7_200,
            cancellation,
            commands,
        )
        _run(
            [
                refine,
                str(scene_dense),
                "-m",
                str(mesh_ply),
                "-o",
                str(scene_refined),
                "--scales",
                "1",
                "--max-face-area",
                "16",
            ],
            work,
            log_path,
            7_200,
            cancellation,
            commands,
        )
        textured_glb: Path | None = None
        textured_obj: Path | None = None
        resources: tuple[Path, ...] = ()
        if self.capabilities.tools["texture_mesh"].available:
            texture = executable(self.capabilities, "texture_mesh")
            progress(JobStage.TEXTURING, 76)
            _run(
                [
                    texture,
                    str(scene_dense),
                    "-m",
                    str(refined_ply),
                    "-o",
                    str(native_output / "model_glb.mvs"),
                    "--export-type",
                    "glb",
                ],
                work,
                log_path,
                7_200,
                cancellation,
                commands,
            )
            _run(
                [
                    texture,
                    str(scene_dense),
                    "-m",
                    str(refined_ply),
                    "-o",
                    str(native_output / "model_obj.mvs"),
                    "--export-type",
                    "obj",
                ],
                work,
                log_path,
                7_200,
                cancellation,
                commands,
            )
            glbs = sorted(native_output.glob("*.glb"))
            objs = sorted(native_output.glob("*.obj"))
            if len(glbs) != 1 or len(objs) != 1:
                raise RuntimeError("OpenMVS did not produce exactly one textured GLB and OBJ.")
            textured_glb = glbs[0]
            textured_obj = objs[0]
            resource_suffixes = {".mtl", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            resources = tuple(
                path
                for path in sorted(native_output.iterdir())
                if path.is_file() and path.suffix.lower() in resource_suffixes
            )
        return _OpenMvsProducts(
            dense_cloud=dense_ply,
            mesh=refined_ply,
            textured_glb=textured_glb,
            textured_obj=textured_obj,
            texture_resources=resources,
            log_path=log_path,
        )


@dataclass(frozen=True)
class _ModelComponent:
    path: Path
    registered_images: int
    points: int
    reprojection_error: float | None


@dataclass(frozen=True)
class _OpenMvsProducts:
    dense_cloud: Path
    mesh: Path
    textured_glb: Path | None
    textured_obj: Path | None
    texture_resources: tuple[Path, ...]
    log_path: Path


class SyntheticTestAdapter:
    """Small deterministic adapter that is available only by explicit dependency injection."""

    def __init__(self, *, allow_test_only: bool = False, delay_probe: Callable[[], None] | None = None) -> None:
        if not allow_test_only:
            raise ValueError("SyntheticTestAdapter requires explicit allow_test_only=True.")
        self._delay_probe = delay_probe

    @property
    def name(self) -> str:
        return "synthetic-test-only"

    def reconstruct(
        self,
        image_paths: Sequence[Path],
        working_directory: Path,
        *,
        configuration: ScanConfiguration,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> PhotogrammetryProducts:
        del configuration
        cancellation.raise_if_cancelled()
        if self._delay_probe:
            self._delay_probe()
        work = working_directory.resolve()
        work.mkdir(parents=True, exist_ok=True)
        progress(JobStage.ESTIMATING_CAMERAS, 45)
        box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
        box.visual.vertex_colors = np.tile(np.array([35, 156, 214, 255], dtype=np.uint8), (8, 1))
        mesh_path = work / "synthetic-mesh.ply"
        mesh_path.write_bytes(bytes(box.export(file_type="ply")))
        dense_points, _ = trimesh.sample.sample_surface(box, 2_000, seed=17)
        dense_cloud = trimesh.points.PointCloud(dense_points)
        dense_path = work / "synthetic-dense.ply"
        dense_path.write_bytes(bytes(dense_cloud.export(file_type="ply")))
        sparse_points = dense_points[::20]
        sparse_path = work / "synthetic-sparse.ply"
        sparse_path.write_bytes(
            bytes(trimesh.points.PointCloud(sparse_points).export(file_type="ply"))
        )
        progress(JobStage.BUILDING_DENSE_CLOUD, 62)
        progress(JobStage.BUILDING_MESH, 72)
        cancellation.raise_if_cancelled()
        return PhotogrammetryProducts(
            sparse_cloud_path=sparse_path,
            dense_cloud_path=dense_path,
            mesh_path=mesh_path,
            textured_glb_path=None,
            textured_obj_path=None,
            texture_resources=(),
            registered_cameras=len(image_paths),
            sparse_points=len(sparse_points),
            reprojection_error_px=0.35,
            commands=(),
            tool_versions={"synthetic_test_adapter": "1"},
            log_paths=(),
            warnings=(
                "Synthetic test geometry was injected; this is not a real reconstruction result.",
            ),
        )


def _run(
    arguments: Sequence[str],
    work: Path,
    log_path: Path,
    timeout: float,
    cancellation: CancellationToken,
    commands: list[tuple[str, ...]],
) -> ProcessResult:
    result = run_process(
        arguments,
        working_directory=work,
        log_path=log_path,
        timeout_seconds=timeout,
        cancellation=cancellation,
    )
    commands.append(result.arguments)
    return result


def _require_colmap_commands(
    colmap: str,
    names: Sequence[str],
    work: Path,
    log_path: Path,
    cancellation: CancellationToken,
) -> None:
    missing: list[str] = []
    for name in names:
        command_log = log_path.with_name(f"{log_path.stem}-{name}{log_path.suffix}")
        try:
            captured = run_process_capture(
                [colmap, name, "-h"],
                working_directory=work,
                log_path=command_log,
                timeout_seconds=20,
                cancellation=cancellation,
                maximum_stdout_bytes=2 * 1024 * 1024,
            )
        except ExternalProcessError:
            missing.append(name)
            continue
        help_text = (
            captured.stdout.decode("utf-8", errors="replace")
            + "\n"
            + captured.process.output_tail
        ).lower()
        if name.lower() not in help_text and "options" not in help_text:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "The configured COLMAP build lacks required command(s): " + ", ".join(missing) + "."
        )


def _analyze_components(
    colmap: str,
    sparse_directory: Path,
    work: Path,
    log_path: Path,
    cancellation: CancellationToken,
    commands: list[tuple[str, ...]],
) -> list[_ModelComponent]:
    components: list[_ModelComponent] = []
    for directory in sorted(path for path in sparse_directory.iterdir() if path.is_dir()):
        arguments = [colmap, "model_analyzer", "--path", str(directory)]
        component_log = log_path.with_name(
            f"{log_path.stem}-model-{directory.name}{log_path.suffix}"
        )
        captured = run_process_capture(
            arguments,
            working_directory=work,
            log_path=component_log,
            timeout_seconds=300,
            cancellation=cancellation,
            maximum_stdout_bytes=2 * 1024 * 1024,
        )
        commands.append(tuple(arguments))
        output = (
            captured.stdout.decode("utf-8", errors="replace")
            + "\n"
            + captured.process.output_tail
        )
        registered = _metric(output, r"Registered images\s*:\s*(\d+)", integer=True)
        points = _metric(output, r"Points\s*:\s*(\d+)", integer=True)
        reprojection = _metric(
            output, r"Mean reprojection error\s*:\s*([0-9.eE+-]+)", integer=False
        )
        if registered is not None and points is not None:
            components.append(
                _ModelComponent(directory, int(registered), int(points), reprojection)
            )
    return components


def _metric(text: str, pattern: str, *, integer: bool) -> int | float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1)) if integer else float(match.group(1))
    except ValueError:
        return None


def _link_or_copy_images(sources: Sequence[Path], destination: Path) -> None:
    for index, source in enumerate(sources, start=1):
        resolved = source.resolve(strict=True)
        suffix = resolved.suffix.lower() if resolved.suffix else ".jpg"
        target = destination / f"image-{index:04d}{suffix}"
        try:
            target.hardlink_to(resolved)
        except OSError:
            target.write_bytes(resolved.read_bytes())


def _feature_image_size(preset: QualityPreset) -> int:
    return {
        QualityPreset.DRAFT: 1_600,
        QualityPreset.BALANCED: 2_400,
        QualityPreset.HIGH: 3_200,
    }[preset]


def _dense_image_size(preset: QualityPreset) -> int:
    return {
        QualityPreset.DRAFT: 1_200,
        QualityPreset.BALANCED: 2_000,
        QualityPreset.HIGH: 3_200,
    }[preset]


def _openmvs_ready(capabilities: ToolchainCapabilities) -> bool:
    return all(
        capabilities.tools[name].available
        for name in (
            "interface_colmap",
            "densify_point_cloud",
            "reconstruct_mesh",
            "refine_mesh",
        )
    )


def _colmap_executable(configured: str) -> str:
    path = Path(configured)
    if path.suffix.lower() in {".bat", ".cmd"}:
        direct = path.parent / "bin" / "colmap.exe"
        if direct.is_file():
            return str(direct.resolve())
        raise RuntimeError(
            "CADPRO_COLMAP_PATH points to a batch wrapper. Configure it to the release's "
            "bin/colmap.exe and ensure the release DLL directories are on PATH; CadPro does not "
            "construct shell command strings from job data."
        )
    return configured

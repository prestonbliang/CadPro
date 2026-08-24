"""End-to-end scan orchestration with explicit representation and export gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Callable, Literal

import numpy as np

from cadpro.scan.artifacts import (
    PublishedArtifact,
    publish_file,
    sha256_file,
    write_bundle,
    write_manifest,
    write_preview,
    write_report,
)
from cadpro.scan.cad_fit import export_fitted_step, fit_supported_cad
from cadpro.scan.capabilities import detect_toolchain, redacted_toolchain
from cadpro.scan.jobs import ScanJobContext, ScanJobResult
from cadpro.scan.mesh import export_mesh_products, export_point_cloud, load_valid_mesh, validate_point_cloud
from cadpro.scan.models import (
    ArtifactKind,
    InputMode,
    JobStage,
    JobStatus,
    ReconstructionMetrics,
    ReconstructionReport,
    ReconstructionReuse,
    ScaleInformation,
    StageTiming,
    StructuredNotice,
    ToolchainCapabilities,
    VideoFrame,
    VideoMetadata,
)
from cadpro.scan.photogrammetry import PhotogrammetryAdapter, PhotogrammetryProducts
from cadpro.scan.quality import analyze_and_normalize_images
from cadpro.scan.scale import calculate_scale
from cadpro.scan.video import prepare_video_frames


@dataclass
class _StageClock:
    callback: Callable[[JobStage, int], None]
    current_stage: JobStage | None = None
    started_at: datetime | None = None
    started_monotonic: float | None = None
    timings: list[StageTiming] | None = None

    def __post_init__(self) -> None:
        self.timings = []

    def advance(self, stage: JobStage, progress: int) -> None:
        now = datetime.now(timezone.utc)
        monotonic = time.monotonic()
        if self.current_stage != stage:
            self._close(now, monotonic)
            self.current_stage = stage
            self.started_at = now
            self.started_monotonic = monotonic
        self.callback(stage, progress)

    def finish(self) -> tuple[StageTiming, ...]:
        self._close(datetime.now(timezone.utc), time.monotonic())
        self.current_stage = None
        return tuple(self.timings or ())

    def _close(self, now: datetime, monotonic: float) -> None:
        if (
            self.current_stage is None
            or self.started_at is None
            or self.started_monotonic is None
            or self.timings is None
        ):
            return
        self.timings.append(
            StageTiming(
                stage=self.current_stage,
                started_at=self.started_at,
                finished_at=now,
                duration_seconds=max(0.0, monotonic - self.started_monotonic),
            )
        )


class ScanPipeline:
    def __init__(
        self,
        adapter: PhotogrammetryAdapter,
        *,
        capabilities: ToolchainCapabilities | None = None,
    ) -> None:
        self.adapter = adapter
        self.capabilities = capabilities or detect_toolchain()

    def __call__(self, context: ScanJobContext) -> ScanJobResult:
        configuration = context.configuration
        workspace = context.workspace
        output = workspace.output_directory
        clock = _StageClock(context.progress)
        warnings: list[StructuredNotice] = []
        commands: list[list[str]] = []
        selected_frames: tuple[VideoFrame, ...] = ()
        video_metadata: VideoMetadata | None = None
        contact_sheet: Path | None = None

        clock.advance(JobStage.VALIDATING, 5)
        context.cancellation.raise_if_cancelled()
        if configuration.mode == InputMode.SINGLE_IMAGE:
            raise RuntimeError(
                "Single-photo mode is experimental and no local single-image reconstruction "
                "provider is configured. Upload overlapping photos or an orbit video; CadPro will "
                "not fabricate unseen geometry."
            )
        input_hashes = {path.name: sha256_file(path) for path in context.input_paths}

        capture_paths = context.input_paths
        if configuration.mode == InputMode.VIDEO:
            if len(context.input_paths) != 1:
                raise ValueError("A video scan must contain exactly one uploaded file.")
            if not self.capabilities.video_ingest:
                missing = [
                    capability.name
                    for key, capability in self.capabilities.tools.items()
                    if key in {"ffmpeg", "ffprobe"} and not capability.available
                ]
                raise RuntimeError(
                    "Video ingest dependency unavailable: "
                    + ", ".join(missing)
                    + ". Install FFmpeg and FFprobe, then restart CadPro."
                )
            clock.advance(JobStage.EXTRACTING_FRAMES, 12)
            prepared = prepare_video_frames(
                context.input_paths[0],
                workspace.working_directory / "video",
                capabilities=self.capabilities,
                settings=configuration.video,
                preset=configuration.quality_preset,
                cancellation=context.cancellation,
            )
            capture_paths = prepared.selected_paths
            selected_frames = prepared.frames
            video_metadata = prepared.metadata
            contact_sheet = prepared.contact_sheet_path
            commands.extend([list(command) for command in prepared.commands])

        clock.advance(JobStage.ANALYZING_IMAGES, 20)
        normalized, image_quality = analyze_and_normalize_images(
            capture_paths,
            workspace.working_directory / "accepted-images",
            preset=configuration.quality_preset,
            cancellation=context.cancellation,
        )
        if len(normalized) < 3:
            rejected = len(image_quality) - len(normalized)
            raise RuntimeError(
                f"Only {len(normalized)} usable, distinct images remained after quality checks "
                f"({rejected} rejected); real photogrammetry needs at least 3 and normally 20–50."
            )
        if len(normalized) < 20:
            warnings.append(
                StructuredNotice(
                    code="low_view_count",
                    message=(
                        f"Only {len(normalized)} views were accepted. Reconstruction can run, but "
                        "20–50 overlapping high/low-angle views are recommended."
                    ),
                    stage=JobStage.ANALYZING_IMAGES,
                )
            )

        clock.advance(JobStage.ESTIMATING_CAMERAS, 28)
        if configuration.reconstruction_reuse is not None:
            products = _reused_reconstruction(
                workspace.working_directory,
                configuration.reconstruction_reuse,
            )
            clock.advance(JobStage.BUILDING_MESH, 76)
        else:
            products = self.adapter.reconstruct(
                normalized,
                workspace.working_directory / "reconstruction",
                configuration=configuration,
                cancellation=context.cancellation,
                progress=clock.advance,
            )
        commands.extend([list(command) for command in products.commands])
        for message in products.warnings:
            warnings.append(
                StructuredNotice(
                    code="reconstruction_warning",
                    message=message,
                    stage=JobStage.BUILDING_MESH,
                )
            )

        clock.advance(JobStage.APPLYING_SCALE, 80)
        scale = (
            calculate_scale(configuration.scale)
            if configuration.scale is not None
            else ScaleInformation()
        )
        if not scale.calibrated:
            warnings.append(
                StructuredNotice(
                    code="scale_unknown",
                    message=scale.warning,
                    stage=JobStage.APPLYING_SCALE,
                )
            )

        clock.advance(JobStage.REPAIRING_MESH, 83)
        visual_source = products.textured_obj_path or products.mesh_path
        mesh_exports = export_mesh_products(
            visual_source,
            output,
            scale=scale,
            stem="cadpro-scan",
        )
        for message in mesh_exports.warnings:
            warnings.append(
                StructuredNotice(
                    code="mesh_repair_warning",
                    message=message,
                    stage=JobStage.REPAIRING_MESH,
                )
            )
        if products.textured_obj_path is not None and not mesh_exports.textured:
            warnings.append(
                StructuredNotice(
                    code="texture_validation_failed",
                    message=(
                        "The native texturer output could not be proven to retain linked texture "
                        "data after export; artifacts are labeled untextured."
                    ),
                    stage=JobStage.TEXTURING,
                )
            )

        clock.advance(JobStage.EXPORTING, 87)
        sparse_path = export_point_cloud(
            products.sparse_cloud_path,
            output / "cadpro-sparse-cloud.ply",
            scale=scale,
        )
        dense_path = export_point_cloud(
            products.dense_cloud_path,
            output / "cadpro-dense-cloud.ply",
            scale=scale,
        )
        dense_count = validate_point_cloud(dense_path)
        published: list[PublishedArtifact] = [
            publish_file(
                sparse_path,
                output,
                filename=sparse_path.name,
                artifact_id="sparse-ply",
                kind=ArtifactKind.SPARSE_POINT_CLOUD,
                metric_scale=scale.calibrated,
            ),
            publish_file(
                dense_path,
                output,
                filename=dense_path.name,
                artifact_id="dense-ply",
                kind=ArtifactKind.DENSE_POINT_CLOUD,
                metric_scale=scale.calibrated,
            ),
            publish_file(
                mesh_exports.cleaned_mesh_path,
                output,
                filename=mesh_exports.cleaned_mesh_path.name,
                artifact_id="mesh-ply",
                kind=ArtifactKind.TRIANGLE_MESH,
                metric_scale=scale.calibrated,
            ),
            publish_file(
                mesh_exports.glb_path,
                output,
                filename=mesh_exports.glb_path.name,
                artifact_id="visual-glb",
                kind=ArtifactKind.TEXTURED_MODEL,
                textured=mesh_exports.textured,
                metric_scale=scale.calibrated,
            ),
            publish_file(
                mesh_exports.obj_path,
                output,
                filename=mesh_exports.obj_path.name,
                artifact_id="mesh-obj",
                kind=ArtifactKind.TRIANGLE_MESH,
                textured=mesh_exports.textured,
                metric_scale=scale.calibrated,
            ),
        ]
        for index, resource in enumerate(mesh_exports.obj_resources, start=1):
            published.append(
                publish_file(
                    resource,
                    output,
                    filename=resource.name,
                    artifact_id=f"texture-{index}",
                    kind=ArtifactKind.TEXTURE_RESOURCE,
                    textured=True,
                    metric_scale=None,
                )
            )
        if mesh_exports.stl_path is not None:
            published.append(
                publish_file(
                    mesh_exports.stl_path,
                    output,
                    filename=mesh_exports.stl_path.name,
                    artifact_id="printable-stl",
                    kind=ArtifactKind.PRINTABLE_MESH,
                    metric_scale=scale.calibrated,
                )
            )
        if contact_sheet is not None:
            published.append(
                publish_file(
                    contact_sheet,
                    output,
                    filename="selected-frames-contact-sheet.jpg",
                    artifact_id="contact-sheet",
                    kind=ArtifactKind.CONTACT_SHEET,
                )
            )
        published.append(write_preview(mesh_exports.glb_path, output))

        clock.advance(JobStage.FITTING_CAD, 91)
        cad_features = []
        cad_status: Literal["generated", "skipped", "failed"] = "skipped"
        cad_explanation = "STEP was skipped because scale is unknown."
        if configuration.generate_cad and scale.calibrated:
            mesh_for_fit = load_valid_mesh(mesh_exports.cleaned_mesh_path)
            vertices = np.asarray(mesh_for_fit.vertices, dtype=np.float64)
            threshold = max(float(np.max(mesh_for_fit.extents)) * 0.005, 1e-6)
            fit = fit_supported_cad(vertices, distance_threshold=threshold)
            cad_features = list(fit.features)
            cad_explanation = fit.explanation
            if fit.selected is not None:
                try:
                    cad_export = export_fitted_step(
                        fit.selected,
                        output,
                        scale=scale,
                    )
                    published.extend(
                        [
                            publish_file(
                                cad_export.step_path,
                                output,
                                filename=cad_export.step_path.name,
                                artifact_id="fitted-step",
                                kind=ArtifactKind.ANALYTIC_CAD,
                                metric_scale=True,
                            ),
                            publish_file(
                                cad_export.script_path,
                                output,
                                filename=cad_export.script_path.name,
                                artifact_id="cad-script",
                                kind=ArtifactKind.CAD_SCRIPT,
                                metric_scale=True,
                            ),
                        ]
                    )
                    cad_status = "generated"
                except (RuntimeError, ValueError) as error:
                    cad_status = "failed"
                    cad_explanation = f"An analytic fit passed, but STEP validation failed: {error}"
                    warnings.append(
                        StructuredNotice(
                            code="cad_export_failed",
                            message=cad_explanation,
                            stage=JobStage.FITTING_CAD,
                        )
                    )
        elif not configuration.generate_cad:
            cad_explanation = "STEP fitting was disabled in this job's configuration."

        # Include sanitized, bounded native logs as downloadable diagnostics.
        log_candidates = sorted(
            {
                path.resolve()
                for path in workspace.working_directory.rglob("*.log")
                if path.is_file() and path.stat().st_size > 0
            }
        )
        for index, log_path in enumerate(log_candidates[:20], start=1):
            published.append(
                publish_file(
                    log_path,
                    output,
                    filename=f"native-tool-{index:02d}.log",
                    artifact_id=f"native-log-{index}",
                    kind=ArtifactKind.LOG,
                )
            )

        clock.advance(JobStage.VALIDATING_OUTPUTS, 96)
        registration = products.registered_cameras / len(normalized) * 100
        statistics = mesh_exports.statistics
        metrics = ReconstructionMetrics(
            uploaded_images=len(capture_paths),
            accepted_images=len(normalized),
            registered_cameras=products.registered_cameras,
            registration_percentage=min(100.0, registration),
            sparse_points=products.sparse_points,
            dense_points=dense_count,
            reprojection_error_px=products.reprojection_error_px,
            bounding_box=statistics.bounding_box,
            vertices=statistics.vertices,
            triangles=statistics.triangles,
            connected_components=statistics.connected_components,
            boundary_edges=statistics.boundary_edges,
            non_manifold_edges=statistics.non_manifold_edges,
            watertight=statistics.watertight,
            texture_resolution=_texture_resolution(mesh_exports.obj_resources),
        )
        timings = clock.finish()
        tool_versions = dict(products.tool_versions)
        tool_versions["pipeline_adapter"] = (
            "immutable-artifact-reuse"
            if configuration.reconstruction_reuse is not None
            else self.adapter.name
        )
        report = ReconstructionReport(
            job_id=str(context.job_id),
            mode=configuration.mode,
            status=JobStatus.COMPLETED,
            quality_class=_quality_class(metrics),
            configuration=configuration.model_dump(mode="json"),
            capabilities=redacted_toolchain(self.capabilities),
            image_quality=list(image_quality),
            video=video_metadata,
            selected_frames=list(selected_frames),
            metrics=metrics,
            scale=scale,
            cad_features=cad_features,
            cad_status=cad_status,
            cad_explanation=cad_explanation,
            warnings=warnings,
            errors=[],
            timings=list(timings),
            tool_versions=tool_versions,
            artifacts=[artifact.metadata for artifact in published],
            inferred_surfaces=False,
        )
        report_artifact = write_report(report, output)
        published.append(report_artifact)
        manifest_artifact = write_manifest(
            job_id=str(context.job_id),
            input_sha256=input_hashes,
            configuration=configuration.model_dump(mode="json"),
            commands=commands,
            tool_versions=tool_versions,
            artifacts=[artifact.metadata for artifact in published],
            output_directory=output,
        )
        published.append(manifest_artifact)
        bundle = write_bundle(published, output)
        published.append(bundle)
        return ScanJobResult(
            artifacts=tuple(artifact.metadata for artifact in published),
            report=report.model_dump(mode="json"),
            tool_versions=tool_versions,
            warnings=tuple(warnings),
        )


def _quality_class(
    metrics: ReconstructionMetrics,
) -> Literal["excellent", "usable", "weak", "failed"]:
    reprojection = metrics.reprojection_error_px
    if (
        metrics.accepted_images >= 20
        and metrics.registration_percentage >= 90
        and (reprojection is None or reprojection <= 0.8)
        and metrics.dense_points >= 500_000
        and metrics.connected_components == 1
    ):
        return "excellent"
    if (
        metrics.accepted_images >= 12
        and metrics.registration_percentage >= 70
        and (reprojection is None or reprojection <= 1.5)
        and metrics.dense_points >= 100_000
    ):
        return "usable"
    return "weak"


def _texture_resolution(resources: tuple[Path, ...]) -> tuple[int, int] | None:
    from PIL import Image

    for path in resources:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        try:
            with Image.open(path) as image:
                return image.width, image.height
        except OSError:
            continue
    return None


def _reused_reconstruction(
    working_directory: Path,
    reuse: ReconstructionReuse,
) -> PhotogrammetryProducts:
    root = (working_directory / "reused-reconstruction").resolve(strict=True)

    def required(name: str) -> Path:
        candidate = root / name
        if candidate.is_symlink():
            raise RuntimeError("A calibration source artifact cannot be a symbolic link.")
        path = candidate.resolve(strict=True)
        if path.parent != root or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("A calibration source artifact is unsafe, missing, or empty.")
        return path

    sparse = required("sparse.ply")
    dense = required("dense.ply")
    mesh = required("mesh.ply")
    textured_obj: Path | None = None
    resources: tuple[Path, ...] = ()
    if reuse.source_textured:
        textured_obj = required("model.obj")
        resources = tuple(
            path
            for path in sorted(root.iterdir())
            if path.is_file()
            and path.name not in {"sparse.ply", "dense.ply", "mesh.ply", "model.obj"}
        )
    versions = dict(reuse.tool_versions)
    versions["reconstruction_reused_from_job"] = str(reuse.source_job_id)
    return PhotogrammetryProducts(
        sparse_cloud_path=sparse,
        dense_cloud_path=dense,
        mesh_path=mesh,
        textured_glb_path=None,
        textured_obj_path=textured_obj,
        texture_resources=resources,
        registered_cameras=reuse.registered_cameras,
        sparse_points=reuse.sparse_points,
        reprojection_error_px=reuse.reprojection_error_px,
        commands=(),
        tool_versions=versions,
        log_paths=(),
        warnings=(
            "Calibration reused the completed job's immutable point clouds and mesh; native "
            "camera estimation and dense reconstruction were not rerun.",
            *reuse.reconstruction_warnings,
        ),
    )

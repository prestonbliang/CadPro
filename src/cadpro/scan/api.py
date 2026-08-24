"""Versioned FastAPI routes for persistent truthful scan jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from cadpro.scan.capabilities import redacted_toolchain
from cadpro.scan.jobs import JobStore, PersistentScanService
from cadpro.scan.models import (
    InputMode,
    JobStage,
    QualityPreset,
    ReconstructionReport,
    ReconstructionReuse,
    ScaleMeasurement,
    ScanConfiguration,
    StructuredNotice,
    ToolchainCapabilities,
    VideoSelectionSettings,
)


SCAN_MIN_PHOTOS = 3
SCAN_RECOMMENDED_PHOTOS = 20
SCAN_MAX_PHOTOS = 100
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_PHOTO_SET_BYTES = 500 * 1024 * 1024
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

_IMAGE_SUFFIXES = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
_IMAGE_MIME = {
    "jpeg": {"image/jpeg", "image/jpg"},
    "png": {"image/png"},
    "webp": {"image/webp"},
}
_VIDEO_SUFFIXES = {
    ".avi": "avi",
    ".m4v": "iso-media",
    ".mkv": "ebml",
    ".mov": "iso-media",
    ".mp4": "iso-media",
    ".webm": "ebml",
}
_VIDEO_MIME = {
    ".avi": {"video/x-msvideo", "video/avi"},
    ".m4v": {"video/x-m4v", "video/mp4"},
    ".mkv": {"video/x-matroska", "video/mkv"},
    ".mov": {"video/quicktime"},
    ".mp4": {"video/mp4"},
    ".webm": {"video/webm"},
}
_GENERIC_MIME = {"", "application/octet-stream"}


class ScanApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = dict(details or {})


def create_scan_router(capabilities: ToolchainCapabilities) -> APIRouter:
    router = APIRouter(prefix="/api/v2")

    @router.get("/capabilities")
    async def scan_capabilities() -> dict[str, object]:
        return {
            "version": "2",
            "capabilities": redacted_toolchain(capabilities).model_dump(mode="json"),
            "capture_limits": {
                "photos": {
                    "minimum": SCAN_MIN_PHOTOS,
                    "recommended_minimum": SCAN_RECOMMENDED_PHOTOS,
                    "maximum": SCAN_MAX_PHOTOS,
                },
                "image_maximum_bytes": MAX_IMAGE_BYTES,
                "photo_set_maximum_bytes": MAX_PHOTO_SET_BYTES,
                "video_maximum_bytes": MAX_VIDEO_BYTES,
                "video_maximum_duration_seconds": 300,
            },
            "single_image": {
                "available": False,
                "experimental": True,
                "metric": False,
                "reason": "No local single-image reconstruction provider is configured.",
            },
            "standard_pipeline_uses_paid_cloud": False,
        }

    @router.post("/jobs/photos", status_code=202)
    async def create_photo_scan(
        request: Request,
        files: Annotated[list[UploadFile], File(description="Overlapping photos of one object.")],
        quality_preset: Annotated[QualityPreset, Form()] = QualityPreset.BALANCED,
        feature_matcher: Annotated[Literal["exhaustive", "sequential"], Form()] = "exhaustive",
        mesher: Annotated[Literal["poisson", "delaunay"], Form()] = "poisson",
        use_gpu: Annotated[bool, Form()] = True,
        generate_cad: Annotated[bool, Form()] = True,
        scale_json: Annotated[str | None, Form(max_length=2_000)] = None,
    ) -> JSONResponse:
        if not SCAN_MIN_PHOTOS <= len(files) <= SCAN_MAX_PHOTOS:
            await _close_uploads(files)
            raise ScanApiError(
                422,
                "invalid_photo_count",
                f"Upload {SCAN_MIN_PHOTOS}–{SCAN_MAX_PHOTOS} overlapping photos; 20–50 is recommended.",
                details={"received": len(files)},
            )
        _require_native(
            request,
            mode=InputMode.PHOTOS,
            capabilities=capabilities,
            use_gpu=use_gpu,
        )
        scale = _parse_scale(scale_json)
        configuration = ScanConfiguration(
            mode=InputMode.PHOTOS,
            quality_preset=quality_preset,
            feature_matcher=feature_matcher,
            mesher=mesher,
            use_gpu=use_gpu,
            generate_cad=generate_cad,
            scale=scale,
        )
        return await _stage_upload_job(
            request,
            mode=InputMode.PHOTOS,
            uploads=files,
            configuration=configuration,
            media_kind="image",
            per_file_limit=MAX_IMAGE_BYTES,
            total_limit=MAX_PHOTO_SET_BYTES,
        )

    @router.post("/jobs/video", status_code=202)
    async def create_video_scan(
        request: Request,
        file: Annotated[UploadFile, File(description="One orbit video around one object.")],
        quality_preset: Annotated[QualityPreset, Form()] = QualityPreset.BALANCED,
        target_frames: Annotated[int, Form(ge=8, le=200)] = 40,
        maximum_duration_seconds: Annotated[float, Form(gt=0, le=3_600)] = 300,
        feature_matcher: Annotated[Literal["exhaustive", "sequential"], Form()] = "exhaustive",
        mesher: Annotated[Literal["poisson", "delaunay"], Form()] = "poisson",
        use_gpu: Annotated[bool, Form()] = True,
        generate_cad: Annotated[bool, Form()] = True,
        scale_json: Annotated[str | None, Form(max_length=2_000)] = None,
    ) -> JSONResponse:
        _require_native(
            request,
            mode=InputMode.VIDEO,
            capabilities=capabilities,
            use_gpu=use_gpu,
        )
        configuration = ScanConfiguration(
            mode=InputMode.VIDEO,
            quality_preset=quality_preset,
            feature_matcher=feature_matcher,
            mesher=mesher,
            use_gpu=use_gpu,
            generate_cad=generate_cad,
            scale=_parse_scale(scale_json),
            video=VideoSelectionSettings(
                target_frames=target_frames,
                maximum_duration_seconds=maximum_duration_seconds,
            ),
        )
        return await _stage_upload_job(
            request,
            mode=InputMode.VIDEO,
            uploads=[file],
            configuration=configuration,
            media_kind="video",
            per_file_limit=MAX_VIDEO_BYTES,
            total_limit=MAX_VIDEO_BYTES,
        )

    @router.post("/jobs/single-image", status_code=409)
    async def create_single_image_scan(file: Annotated[UploadFile, File()]) -> None:
        await file.close()
        raise ScanApiError(
            409,
            "single_image_provider_unavailable",
            (
                "Single-photo reconstruction is experimental, non-metric, and unavailable because "
                "no local provider is configured. Upload overlapping photos or an orbit video."
            ),
        )

    @router.get("/jobs/{job_id}")
    async def get_scan_job(request: Request, job_id: UUID) -> dict[str, object]:
        try:
            return _store(request).snapshot(job_id)
        except KeyError as error:
            raise ScanApiError(404, "job_not_found", "No scan job exists with that ID.") from error

    @router.post("/jobs/{job_id}/cancel")
    async def cancel_scan_job(request: Request, job_id: UUID) -> dict[str, object]:
        try:
            return _service(request).cancel(job_id)
        except KeyError as error:
            raise ScanApiError(404, "job_not_found", "No scan job exists with that ID.") from error

    @router.post("/jobs/{job_id}/calibration", status_code=202)
    async def calibrate_scan_job(
        request: Request,
        job_id: UUID,
        measurement: ScaleMeasurement,
    ) -> JSONResponse:
        store = _store(request)
        service = _service(request)
        try:
            source_snapshot = store.snapshot(job_id)
            source_workspace, source_inputs, source_configuration = store.context_values(job_id)
        except KeyError as error:
            raise ScanApiError(404, "job_not_found", "No scan job exists with that ID.") from error
        if source_snapshot["status"] != "completed":
            raise ScanApiError(
                409,
                "job_not_complete",
                "Calibration creates a new export revision only after reconstruction completes.",
            )
        try:
            source_report = ReconstructionReport.model_validate(source_snapshot["report"])
        except ValidationError as error:
            raise ScanApiError(
                409,
                "source_report_invalid",
                "The source job no longer has a valid reconstruction report.",
            ) from error
        if source_report.scale.calibrated:
            raise ScanApiError(
                409,
                "already_calibrated",
                "This reconstruction is already scaled; calibrate from its original unscaled job.",
            )
        del source_workspace
        workspace = store.allocate_workspace()
        copied: list[Path] = []
        try:
            for index, source in enumerate(source_inputs, start=1):
                destination = workspace.input_directory / f"input-{index:04d}{source.suffix.lower()}"
                try:
                    destination.hardlink_to(source)
                except OSError:
                    shutil.copyfile(source, destination)
                copied.append(destination)
            reused = workspace.working_directory / "reused-reconstruction"
            reused.mkdir(parents=True, exist_ok=False)
            for artifact_id, destination_name in (
                ("sparse-ply", "sparse.ply"),
                ("dense-ply", "dense.ply"),
                ("mesh-ply", "mesh.ply"),
            ):
                source_artifact, _metadata = store.artifact_path(job_id, artifact_id)
                _link_or_copy(source_artifact, reused / destination_name)
            raw_artifacts = source_snapshot.get("artifacts", [])
            artifacts = raw_artifacts if isinstance(raw_artifacts, list) else []
            mesh_obj = next(
                (
                    item
                    for item in artifacts
                    if isinstance(item, dict) and item.get("artifact_id") == "mesh-obj"
                ),
                None,
            )
            source_textured = bool(isinstance(mesh_obj, dict) and mesh_obj.get("textured"))
            if source_textured:
                source_obj, _metadata = store.artifact_path(job_id, "mesh-obj")
                _link_or_copy(source_obj, reused / "model.obj")
                for item in artifacts:
                    if not isinstance(item, dict) or item.get("kind") != "texture_resource":
                        continue
                    resource_id = item.get("artifact_id")
                    if not isinstance(resource_id, str):
                        continue
                    resource, resource_metadata = store.artifact_path(job_id, resource_id)
                    filename = resource_metadata.get("filename")
                    if not isinstance(filename, str) or Path(filename).name != filename:
                        raise ValueError("The source texture resource has an unsafe filename.")
                    _link_or_copy(resource, reused / filename)
            raw_tool_versions = source_snapshot.get("tool_versions", {})
            tool_version_items = (
                raw_tool_versions.items() if isinstance(raw_tool_versions, dict) else ()
            )
            tool_versions = {str(key): str(value) for key, value in tool_version_items}
            reuse = ReconstructionReuse(
                source_job_id=job_id,
                registered_cameras=source_report.metrics.registered_cameras,
                sparse_points=source_report.metrics.sparse_points,
                reprojection_error_px=source_report.metrics.reprojection_error_px,
                source_textured=source_textured,
                tool_versions=tool_versions,
                reconstruction_warnings=tuple(
                    warning.message
                    for warning in source_report.warnings
                    if warning.code == "reconstruction_warning"
                ),
            )
            configuration = source_configuration.model_copy(
                update={"scale": measurement, "reconstruction_reuse": reuse}
            )
            raw_metadata = source_snapshot.get("input_metadata", [])
            metadata = list(raw_metadata) if isinstance(raw_metadata, list) else []
            metadata.append({"calibration_revision_of": str(job_id)})
            store.create(
                workspace,
                mode=configuration.mode,
                input_paths=copied,
                input_metadata=metadata,
                configuration=configuration,
            )
        except (KeyError, FileNotFoundError, OSError, RuntimeError, ValueError, ValidationError) as error:
            store.discard_workspace(workspace)
            raise ScanApiError(
                409,
                "source_artifacts_unavailable",
                "The immutable source reconstruction artifacts are unavailable or invalid.",
            ) from error
        except Exception:
            store.discard_workspace(workspace)
            raise
        try:
            service.submit(workspace.job_id)
        except RuntimeError as error:
            store.fail(
                workspace.job_id,
                StructuredNotice(
                    code="job_queue_full",
                    message=str(error),
                    stage=JobStage.QUEUED,
                ),
            )
            raise ScanApiError(503, "job_queue_full", str(error)) from error
        return _accepted(store.snapshot(workspace.job_id))

    @router.get("/jobs/{job_id}/artifacts/{artifact_id}")
    async def download_scan_artifact(
        request: Request, job_id: UUID, artifact_id: str
    ) -> FileResponse:
        try:
            path, metadata = _store(request).artifact_path(job_id, artifact_id)
        except KeyError as error:
            raise ScanApiError(404, "artifact_not_found", "That artifact is not part of this job.") from error
        except RuntimeError as error:
            raise ScanApiError(409, "job_not_complete", str(error)) from error
        except FileNotFoundError as error:
            raise ScanApiError(410, "artifact_gone", "The artifact is no longer available.") from error
        inline = path.suffix.lower() == ".html"
        return FileResponse(
            path,
            filename=str(metadata["filename"]),
            media_type=str(metadata["media_type"]),
            content_disposition_type="inline" if inline else "attachment",
            headers={"Cache-Control": "private, no-store", "X-Frame-Options": "SAMEORIGIN"},
        )

    return router


async def _stage_upload_job(
    request: Request,
    *,
    mode: InputMode,
    uploads: Sequence[UploadFile],
    configuration: ScanConfiguration,
    media_kind: Literal["image", "video"],
    per_file_limit: int,
    total_limit: int,
) -> JSONResponse:
    store = _store(request)
    service = _service(request)
    workspace = store.allocate_workspace()
    saved: list[Path] = []
    metadata: list[dict[str, object]] = []
    total = 0
    try:
        for index, upload in enumerate(uploads, start=1):
            suffix = _validated_suffix(upload, media_kind=media_kind, field=f"files[{index - 1}]")
            destination = workspace.input_directory / f"input-{index:04d}{suffix}"
            size, header = await _save_upload(upload, destination, per_file_limit)
            _validate_signature(header, suffix, media_kind=media_kind)
            total += size
            if total > total_limit:
                raise ScanApiError(
                    413,
                    "upload_set_too_large",
                    f"The complete upload may not exceed {total_limit} bytes.",
                    details={"maximum_bytes": total_limit},
                )
            saved.append(destination)
            metadata.append(
                {
                    "source_name": _display_name(upload.filename),
                    "stored_name": destination.name,
                    "size_bytes": size,
                    "content_type": upload.content_type or "application/octet-stream",
                }
            )
        store.create(
            workspace,
            mode=mode,
            input_paths=saved,
            input_metadata=metadata,
            configuration=configuration,
        )
        try:
            service.submit(workspace.job_id)
        except RuntimeError as error:
            store.fail(
                workspace.job_id,
                StructuredNotice(
                    code="job_queue_full",
                    message=str(error),
                    stage=JobStage.QUEUED,
                ),
            )
            raise ScanApiError(503, "job_queue_full", str(error)) from error
    except BaseException:
        try:
            store.snapshot(workspace.job_id)
        except KeyError:
            store.discard_workspace(workspace)
        raise
    finally:
        await _close_uploads(uploads)
    return _accepted(store.snapshot(workspace.job_id))


def _require_native(
    request: Request,
    *,
    mode: InputMode,
    capabilities: ToolchainCapabilities,
    use_gpu: bool,
) -> None:
    if bool(getattr(request.app.state, "scan_test_adapter", False)):
        return
    missing: list[dict[str, object]] = []
    required = ["colmap"]
    if mode == InputMode.VIDEO:
        required.extend(["ffmpeg", "ffprobe"])
    if not use_gpu:
        required.extend(
            [
                "interface_colmap",
                "densify_point_cloud",
                "reconstruct_mesh",
                "refine_mesh",
            ]
        )
    for key in required:
        capability = capabilities.tools[key]
        if not capability.available:
            missing.append(
                {
                    "tool": capability.name,
                    "reason": capability.reason,
                    "install_hint": capability.install_hint,
                }
            )
    if missing:
        raise ScanApiError(
            409,
            "dependency_unavailable",
            "Real local reconstruction cannot start until the listed native dependencies are installed.",
            details={"missing": missing},
        )


def _parse_scale(payload: str | None) -> ScaleMeasurement | None:
    if payload is None or not payload.strip():
        return None
    try:
        return ScaleMeasurement.model_validate(json.loads(payload))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ScanApiError(
            422,
            "invalid_scale",
            "scale_json must contain two finite 3D points, a positive distance, and a supported unit.",
        ) from error


def _validated_suffix(upload: UploadFile, *, media_kind: str, field: str) -> str:
    filename = Path(upload.filename or "").name
    suffix = Path(filename).suffix.lower()
    if media_kind == "image":
        kind = _IMAGE_SUFFIXES.get(suffix)
        allowed = _IMAGE_MIME.get(kind or "", set())
    else:
        kind = _VIDEO_SUFFIXES.get(suffix)
        allowed = _VIDEO_MIME.get(suffix, set())
    if kind is None:
        raise ScanApiError(415, "unsupported_media", f"{field} has an unsupported file extension.")
    content_type = (upload.content_type or "").lower().split(";", 1)[0].strip()
    if content_type not in allowed and content_type not in _GENERIC_MIME:
        raise ScanApiError(
            415,
            "unsupported_media",
            f"{field} has content type {content_type!r}, which does not match {suffix}.",
        )
    return suffix


async def _save_upload(
    upload: UploadFile, destination: Path, maximum_bytes: int
) -> tuple[int, bytes]:
    size = 0
    header = bytearray()
    try:
        with destination.open("xb") as stream:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > maximum_bytes:
                    raise ScanApiError(
                        413,
                        "file_too_large",
                        f"{_display_name(upload.filename)} exceeds the {maximum_bytes}-byte limit.",
                    )
                if len(header) < 64:
                    header.extend(chunk[: 64 - len(header)])
                stream.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if size <= 0:
        destination.unlink(missing_ok=True)
        raise ScanApiError(422, "empty_file", "Uploaded files cannot be empty.")
    return size, bytes(header)


def _validate_signature(header: bytes, suffix: str, *, media_kind: str) -> None:
    if media_kind == "image":
        detected = _detect_image(header)
        expected = _IMAGE_SUFFIXES[suffix]
    else:
        detected = _detect_video(header)
        expected = _VIDEO_SUFFIXES[suffix]
    if detected != expected:
        raise ScanApiError(
            415,
            "signature_mismatch",
            f"The file bytes do not match the {suffix} extension.",
        )


def _detect_image(header: bytes) -> str | None:
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    return None


def _detect_video(header: bytes) -> str | None:
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"AVI ":
        return "avi"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "iso-media"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "ebml"
    return None


async def _close_uploads(uploads: Sequence[UploadFile]) -> None:
    await asyncio.gather(*(upload.close() for upload in uploads))


def _display_name(value: str | None) -> str:
    name = (value or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(character for character in name if character.isprintable())[:128] or "upload"


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copyfile(source, destination)


def _accepted(snapshot: Mapping[str, object]) -> JSONResponse:
    return JSONResponse(
        dict(snapshot),
        status_code=202,
        headers={"Location": str(snapshot["status_url"]), "Retry-After": "1"},
    )


def _store(request: Request) -> JobStore:
    store = getattr(request.app.state, "scan_store", None)
    if not isinstance(store, JobStore):
        raise ScanApiError(503, "service_unavailable", "The persistent scan store is unavailable.")
    return store


def _service(request: Request) -> PersistentScanService:
    service = getattr(request.app.state, "scan_service", None)
    if not isinstance(service, PersistentScanService):
        raise ScanApiError(503, "service_unavailable", "The scan worker is unavailable.")
    return service

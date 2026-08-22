"""Local web application and asynchronous reconstruction job service.

The browser never receives filesystem paths.  Uploads and generated artifacts live
in a private, per-process temporary directory and artifacts can only be downloaded
through the opaque IDs registered after a successful reconstruction.
"""

from __future__ import annotations

import asyncio
import html
import math
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from cadpro import __version__
from cadpro.media import MAX_IMAGE_EDGE, MAX_IMAGE_PIXELS, validated_image_size


DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_JOB_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOB_SWEEP_SECONDS = 60.0
DEFAULT_MAX_PENDING_JOBS = 2
DEFAULT_PUBLIC_ORIGIN = "http://127.0.0.1:8000"
DEFAULT_REQUEST_OVERHEAD_BYTES = 2 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_WIDTH_MM = 1_000_000.0

ASSET_DIR = Path(__file__).with_name("web_assets")
INDEX_FILE = ASSET_DIR / "index.html"

_IMAGE_SUFFIX_KIND = {
    ".bmp": "bmp",
    ".jpeg": "jpeg",
    ".jpg": "jpeg",
    ".png": "png",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".webp": "webp",
}
_IMAGE_CONTENT_TYPES = {
    "bmp": {"image/bmp", "image/x-bmp"},
    "jpeg": {"image/jpeg", "image/jpg"},
    "png": {"image/png"},
    "tiff": {"image/tiff"},
    "webp": {"image/webp"},
}
_GENERIC_CONTENT_TYPES = {"", "application/octet-stream"}
_ARTIFACT_SUFFIXES = {
    ".glb",
    ".gltf",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".obj",
    ".png",
    ".step",
    ".stl",
    ".stp",
    ".zip",
}
_TERMINAL_STATUSES = {"completed", "failed"}
_SAFE_ARTIFACT_ID = re.compile(r"[^a-z0-9]+")
_STAGE_ORDER = {
    "queued": 0,
    "upload": 1,
    "segment": 2,
    "reconstruct": 3,
    "export": 4,
    "complete": 5,
}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class ApiError(Exception):
    """A client-facing API failure with a stable machine-readable code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = dict(details or {})


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    filename: str
    path: Path
    media_type: str
    size_bytes: int


@dataclass
class Job:
    job_id: UUID
    kind: Literal["image"]
    root: Path
    input_dir: Path
    output_dir: Path
    input_paths: tuple[Path, ...]
    parameters: dict[str, Any]
    created_at: datetime
    created_monotonic: float
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    stage: Literal["queued", "upload", "segment", "reconstruct", "export", "complete"] = "queued"
    progress: int = 10
    started_at: datetime | None = None
    finished_at: datetime | None = None
    finished_monotonic: float | None = None
    artifacts: dict[str, Artifact] | None = None
    metrics: dict[str, Any] | None = None
    input_diagnostics: list[dict[str, Any]] | None = None
    error: dict[str, Any] | None = None
    future: Future[None] | None = None
    _state_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def advance(
        self,
        stage: Literal["queued", "upload", "segment", "reconstruct", "export", "complete"],
        progress: int,
    ) -> bool:
        """Advance visible work state without ever allowing a concurrent regression."""
        if not 0 <= progress <= 100:
            raise ValueError("progress must be between 0 and 100")
        with self._state_lock:
            if progress < self.progress or _STAGE_ORDER[stage] < _STAGE_ORDER[self.stage]:
                return False
            self.stage = stage
            self.progress = progress
            return True


class _RequestBodyTooLarge(Exception):
    """Internal signal used to stop a chunked request before multipart parsing finishes."""


@dataclass(frozen=True)
class StagedJob:
    job_id: UUID
    root: Path
    input_dir: Path
    output_dir: Path


ReconstructionRunner = Callable[[Job], object]


class JobManager:
    """Own upload storage and execute exactly one reconstruction at a time."""

    def __init__(
        self,
        *,
        storage_parent: str | Path | None,
        runner: ReconstructionRunner,
        retention_seconds: float,
        sweep_interval_seconds: float,
        max_pending_jobs: int,
    ) -> None:
        if retention_seconds < 0:
            raise ValueError("retention_seconds cannot be negative")
        if sweep_interval_seconds <= 0:
            raise ValueError("sweep_interval_seconds must be positive")
        if max_pending_jobs <= 0:
            raise ValueError("max_pending_jobs must be positive")
        self._lock = threading.RLock()
        self._jobs: dict[UUID, Job] = {}
        self._staged: set[UUID] = set()
        self._accepting = True
        self._runner = runner
        self._retention_seconds = retention_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._max_pending_jobs = max_pending_jobs
        self._sweep_stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cadpro-reconstruct")
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if storage_parent is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="cadpro-web-")
            self.root = Path(self._temporary.name).resolve()
        else:
            parent = Path(storage_parent).resolve()
            parent.mkdir(parents=True, exist_ok=True)
            self.root = parent / f".cadpro-web-{uuid4()}"
            self.root.mkdir(parents=False, exist_ok=False)
        self._sweeper = threading.Thread(
            target=self._sweep_loop,
            name="cadpro-expiry-sweeper",
            daemon=True,
        )
        self._sweeper.start()

    def has_capacity(self) -> bool:
        """Return whether another upload can be admitted without growing an unbounded queue."""
        with self._lock:
            self._prune_locked()
            return self._accepting and self._pending_count_locked() < self._max_pending_jobs

    def stage(self) -> StagedJob:
        with self._lock:
            if not self._accepting:
                raise ApiError(503, "service_stopping", "The reconstruction service is stopping.")
            self._prune_locked()
            if self._pending_count_locked() >= self._max_pending_jobs:
                raise ApiError(
                    503,
                    "job_queue_full",
                    "CadPro is already processing the maximum number of jobs. Try again shortly.",
                    details={"maximum_pending_jobs": self._max_pending_jobs},
                )
            job_id = uuid4()
            self._staged.add(job_id)
            root = self.root / str(job_id)
            input_dir = root / "inputs"
            output_dir = root / "artifacts"
            try:
                input_dir.mkdir(parents=True, exist_ok=False)
                output_dir.mkdir(parents=False, exist_ok=False)
            except Exception:
                self._staged.discard(job_id)
                shutil.rmtree(root, ignore_errors=True)
                raise
            return StagedJob(job_id, root, input_dir, output_dir)

    def discard(self, stage: StagedJob) -> None:
        with self._lock:
            self._staged.discard(stage.job_id)
        shutil.rmtree(stage.root, ignore_errors=True)

    def submit(
        self,
        stage: StagedJob,
        *,
        kind: Literal["image"],
        input_paths: Sequence[Path],
        parameters: Mapping[str, Any],
    ) -> Job:
        now = datetime.now(timezone.utc)
        job = Job(
            job_id=stage.job_id,
            kind=kind,
            root=stage.root,
            input_dir=stage.input_dir,
            output_dir=stage.output_dir,
            input_paths=tuple(input_paths),
            parameters=dict(parameters),
            created_at=now,
            created_monotonic=time.monotonic(),
        )
        with self._lock:
            if not self._accepting:
                self.discard(stage)
                raise ApiError(503, "service_stopping", "The reconstruction service is stopping.")
            if stage.job_id not in self._staged:
                self.discard(stage)
                raise ApiError(409, "invalid_job_stage", "The upload staging area is no longer active.")
            self._staged.remove(stage.job_id)
            self._jobs[job.job_id] = job
            try:
                job.future = self._executor.submit(self._execute, job.job_id)
            except Exception:
                self._jobs.pop(job.job_id, None)
                self.discard(stage)
                raise
        return job

    def get(self, job_id: UUID) -> Job:
        with self._lock:
            self._prune_locked()
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, "job_not_found", "No reconstruction job exists with that ID.")
            return job

    def snapshot(self, job_id: UUID) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            with job._state_lock:
                result: dict[str, Any] | None = None
                if job.status == "completed":
                    result = {
                        "artifacts": [
                            {
                                "id": artifact.artifact_id,
                                "filename": artifact.filename,
                                "media_type": artifact.media_type,
                                "size_bytes": artifact.size_bytes,
                                "download_url": (
                                    f"/api/jobs/{job.job_id}/artifacts/{artifact.artifact_id}"
                                ),
                            }
                            for artifact in (job.artifacts or {}).values()
                        ],
                        "metrics": dict(job.metrics or {}),
                        "input_diagnostics": list(job.input_diagnostics or []),
                    }
                return {
                    "id": str(job.job_id),
                    "kind": job.kind,
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                    "created_at": _iso(job.created_at),
                    "started_at": _iso(job.started_at),
                    "finished_at": _iso(job.finished_at),
                    "input_count": len(job.input_paths),
                    "parameters": dict(job.parameters),
                    "result": result,
                    "error": dict(job.error) if job.error else None,
                    "status_url": f"/api/jobs/{job.job_id}",
                }

    def artifact(self, job_id: UUID, artifact_id: str) -> Artifact:
        with self._lock:
            job = self.get(job_id)
            with job._state_lock:
                if job.status != "completed":
                    raise ApiError(
                        409,
                        "job_not_complete",
                        "Artifacts are available only after reconstruction completes.",
                        details={"status": job.status},
                    )
                artifact = (job.artifacts or {}).get(artifact_id)
            if artifact is None:
                raise ApiError(404, "artifact_not_found", "That artifact is not part of this job.")
            # Recheck after registration so a replaced file or symlink cannot escape the job.
            if artifact.path.is_symlink() or not artifact.path.is_file():
                raise ApiError(410, "artifact_gone", "The requested artifact is no longer available.")
            try:
                artifact.path.resolve(strict=True).relative_to(job.output_dir.resolve(strict=True))
            except (OSError, ValueError):
                raise ApiError(410, "artifact_gone", "The requested artifact is no longer available.")
            return artifact

    def close(self) -> None:
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            self._sweep_stop.set()
            futures = [job.future for job in self._jobs.values() if job.future is not None]
            for future in futures:
                if not future.running():
                    future.cancel()
        self._sweeper.join()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._jobs.clear()
            self._staged.clear()
        shutil.rmtree(self.root, ignore_errors=True)
        if self._temporary is not None:
            self._temporary.cleanup()

    def _execute(self, job_id: UUID) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            with job._state_lock:
                job.status = "running"
                job.started_at = datetime.now(timezone.utc)
            job.advance("segment", 30)
        try:
            manifest = self._runner(job)
            job.advance("export", 90)
            artifacts = _register_artifacts(manifest, job.output_dir)
            metrics, input_diagnostics = _result_metadata(manifest)
            if not artifacts:
                raise RuntimeError("Reconstruction did not produce any downloadable artifacts.")
        except Exception as error:  # The worker must always reach a terminal state.
            with self._lock:
                with job._state_lock:
                    job.status = "failed"
                    job.advance("complete", 100)
                    job.error = {
                        "code": "reconstruction_failed",
                        "message": _safe_worker_error(error, job.root),
                    }
                    job.finished_at = datetime.now(timezone.utc)
                    job.finished_monotonic = time.monotonic()
            return
        with self._lock:
            with job._state_lock:
                job.artifacts = artifacts
                job.metrics = metrics
                job.input_diagnostics = input_diagnostics
                job.status = "completed"
                job.advance("complete", 100)
                job.finished_at = datetime.now(timezone.utc)
                job.finished_monotonic = time.monotonic()

    def _pending_count_locked(self) -> int:
        return len(self._staged) + sum(
            job.status not in _TERMINAL_STATUSES for job in self._jobs.values()
        )

    def _sweep_loop(self) -> None:
        while not self._sweep_stop.wait(self._sweep_interval_seconds):
            with self._lock:
                self._prune_locked()

    def _prune_locked(self) -> None:
        if self._retention_seconds == 0:
            threshold = time.monotonic()
        else:
            threshold = time.monotonic() - self._retention_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in _TERMINAL_STATUSES
            and job.finished_monotonic is not None
            and job.finished_monotonic <= threshold
        ]
        for job_id in expired:
            job = self._jobs.pop(job_id)
            shutil.rmtree(job.root, ignore_errors=True)


def _run_reconstruction(job: Job) -> object:
    """The one integration seam between the web queue and reconstruction engine."""
    from cadpro.artifacts import export_artifacts
    from cadpro.reconstruct import reconstruct_single_image

    job.advance("segment", 30)
    reconstruction = reconstruct_single_image(
        job.input_paths[0],
        width_mm=job.parameters["width_mm"],
        depth_mm=job.parameters["depth_mm"],
        on_profile_ready=lambda: job.advance("reconstruct", 65),
    )
    job.advance("export", 85)
    return export_artifacts(reconstruction, job.output_dir, stem="cadpro-model")


class RequestGuardMiddleware:
    """Reject untrusted browser requests and oversized uploads before endpoint parsing."""

    def __init__(
        self,
        app: Any,
        *,
        public_origin: str,
        trusted_hosts: Sequence[str],
        image_request_bytes: int,
    ) -> None:
        self.app = app
        self._public_origin = _canonical_origin(public_origin)
        self._trusted_hosts = frozenset(host.lower().rstrip(".") for host in trusted_hosts)
        self._request_limits = {"/api/jobs/image": image_request_bytes}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        host_values = _scope_header_values(scope, b"host")
        request_host = _parse_host_header(host_values[0]) if len(host_values) == 1 else None
        if request_host is None or request_host[0] not in self._trusted_hosts:
            await _guard_error(
                400,
                "untrusted_host",
                "The request Host is not allowed by this CadPro server.",
            )(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        request_limit = self._request_limits.get(path) if method == "POST" else None
        if request_limit is None:
            await self.app(scope, receive, send)
            return

        if not _origin_is_allowed(scope, request_host, self._public_origin):
            await _guard_error(
                403,
                "untrusted_origin",
                "Cross-origin reconstruction uploads are not allowed.",
            )(scope, receive, send)
            return

        content_lengths = _scope_header_values(scope, b"content-length")
        content_length_value = content_lengths[0].strip() if len(content_lengths) == 1 else ""
        if len(content_lengths) > 1 or (
            content_lengths
            and (
                len(content_length_value) > 20
                or re.fullmatch(r"[0-9]+", content_length_value) is None
            )
        ):
            await _guard_error(
                400,
                "invalid_content_length",
                "Content-Length must be one non-negative decimal value.",
            )(scope, receive, send)
            return
        if content_lengths and int(content_length_value) > request_limit:
            await _guard_error(
                413,
                "request_too_large",
                f"The multipart request may not exceed {_format_bytes(request_limit)}.",
                details={"maximum_bytes": request_limit},
            )(scope, receive, send)
            return

        manager = getattr(scope.get("app", object()), "state", None)
        manager = getattr(manager, "job_manager", None)
        if manager is not None and not manager.has_capacity():
            await _guard_error(
                503,
                "job_queue_full",
                "CadPro is already processing the maximum number of jobs. Try again shortly.",
                headers={"Retry-After": "5"},
            )(scope, receive, send)
            return

        received_bytes = 0
        body_too_large = False
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal body_too_large, received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > request_limit:
                    body_too_large = True
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            # Starlette normalizes arbitrary multipart read failures to a 400. Suppress
            # that response once our receiver has identified the precise 413 condition.
            if body_too_large:
                return
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await _guard_error(
                413,
                "request_too_large",
                f"The multipart request may not exceed {_format_bytes(request_limit)}.",
                details={"maximum_bytes": request_limit},
            )(scope, receive, send)
            return
        if body_too_large:
            await _guard_error(
                413,
                "request_too_large",
                f"The multipart request may not exceed {_format_bytes(request_limit)}.",
                details={"maximum_bytes": request_limit},
            )(scope, receive, send)


def create_app(
    *,
    storage_parent: str | Path | None = None,
    asset_dir: str | Path | None = None,
    reconstruction_runner: ReconstructionRunner | None = None,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    job_retention_seconds: float = DEFAULT_JOB_TTL_SECONDS,
    job_sweep_interval_seconds: float = DEFAULT_JOB_SWEEP_SECONDS,
    max_pending_jobs: int = DEFAULT_MAX_PENDING_JOBS,
    request_overhead_bytes: int = DEFAULT_REQUEST_OVERHEAD_BYTES,
    trusted_hosts: Sequence[str] | None = None,
) -> FastAPI:
    """Build the application; injectable limits and runner keep API tests lightweight."""
    if max_image_bytes <= 0:
        raise ValueError("max_image_bytes must be positive")
    if job_retention_seconds < 0:
        raise ValueError("job_retention_seconds cannot be negative")
    if job_sweep_interval_seconds <= 0:
        raise ValueError("job_sweep_interval_seconds must be positive")
    if max_pending_jobs <= 0:
        raise ValueError("max_pending_jobs must be positive")
    if request_overhead_bytes < 0:
        raise ValueError("request_overhead_bytes cannot be negative")

    assets = Path(asset_dir).resolve() if asset_dir is not None else ASSET_DIR.resolve()
    runner = reconstruction_runner or _run_reconstruction
    public_origin = _configured_public_origin()
    allowed_hosts = _configured_trusted_hosts(public_origin, trusted_hosts)
    image_request_bytes = max_image_bytes + request_overhead_bytes

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        manager = JobManager(
            storage_parent=storage_parent,
            runner=runner,
            retention_seconds=job_retention_seconds,
            sweep_interval_seconds=job_sweep_interval_seconds,
            max_pending_jobs=max_pending_jobs,
        )
        application.state.job_manager = manager
        try:
            yield
        finally:
            await asyncio.to_thread(manager.close)

    application = FastAPI(
        title="CadPro",
        version=__version__,
        description="Turn one object image into downloadable CAD artifacts.",
        lifespan=lifespan,
    )

    @application.exception_handler(ApiError)
    async def api_error_handler(_request: Request, error: ApiError) -> JSONResponse:
        payload: dict[str, Any] = {
            "error": {"code": error.code, "message": error.message}
        }
        if error.details:
            payload["error"]["details"] = error.details
        headers = {"Retry-After": "5"} if error.code == "job_queue_full" else None
        return JSONResponse(payload, status_code=error.status_code, headers=headers)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        issues = []
        for item in error.errors():
            location = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            issues.append({"field": location or "request", "message": item.get("msg", "Invalid value")})
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "One or more request fields are invalid.",
                    "details": {"issues": issues},
                }
            },
            status_code=422,
        )

    @application.middleware("http")
    async def api_cache_headers(request: Request, call_next: Callable[..., Any]):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    application.add_middleware(
        RequestGuardMiddleware,
        public_origin=public_origin,
        trusted_hosts=allowed_hosts,
        image_request_bytes=image_request_bytes,
    )

    @application.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        index_file = assets / "index.html"
        if not index_file.is_file():
            raise ApiError(503, "frontend_unavailable", "The CadPro web interface is not installed.")
        try:
            document = index_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ApiError(
                503, "frontend_unavailable", "The CadPro web interface could not be loaded."
            ) from error
        document = document.replace("__CADPRO_ORIGIN__", html.escape(public_origin, quote=True))
        return HTMLResponse(document)

    @application.get("/api/health", include_in_schema=False)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "capture_limits": {
                "images": {"minimum": 1, "maximum": 1},
                "image_file": {
                    "maximum_bytes": max_image_bytes,
                    "maximum_pixels": MAX_IMAGE_PIXELS,
                    "maximum_edge_pixels": MAX_IMAGE_EDGE,
                },
                "dimension_mm": {"maximum": MAX_WIDTH_MM},
            },
        }

    @application.post("/api/jobs/image", status_code=202)
    async def create_image_job(
        request: Request,
        file: Annotated[
            list[UploadFile],
            File(
                description="Exactly one object image.",
                json_schema_extra={"minItems": 1, "maxItems": 1},
            ),
        ],
        width_mm: Annotated[
            float,
            Form(gt=0, le=MAX_WIDTH_MM, description="Known maximum object width in millimeters."),
        ],
        depth_mm: Annotated[
            float,
            Form(gt=0, le=MAX_WIDTH_MM, description="Desired extrusion depth in millimeters."),
        ],
    ) -> JSONResponse:
        if len(file) != 1:
            for upload in file:
                await upload.close()
            raise ApiError(
                422,
                "invalid_image_count",
                "Upload exactly one image.",
                details={"received": len(file), "minimum": 1, "maximum": 1},
            )
        upload = file[0]
        _validate_dimension(width_mm, field="width_mm")
        _validate_dimension(depth_mm, field="depth_mm")
        manager = _manager(request)
        try:
            stage = manager.stage()
        except Exception:
            await upload.close()
            raise
        try:
            suffix = _validated_upload_suffix(upload, field="file")
            destination = stage.input_dir / f"image{suffix}"
            _size, header = await _save_upload(upload, destination, max_image_bytes)
            _validate_file_signature(header, suffix, field="file")
            try:
                await asyncio.to_thread(validated_image_size, destination)
            except ValueError as error:
                raise ApiError(422, "invalid_image", str(error)) from error
            job = manager.submit(
                stage,
                kind="image",
                input_paths=[destination],
                parameters={"width_mm": width_mm, "depth_mm": depth_mm},
            )
        except Exception:
            manager.discard(stage)
            raise
        finally:
            await upload.close()
        return _accepted_job(manager.snapshot(job.job_id))

    @application.get("/api/jobs/{job_id}")
    async def get_job(request: Request, job_id: UUID) -> dict[str, Any]:
        return _manager(request).snapshot(job_id)

    @application.get("/api/jobs/{job_id}/artifacts/{artifact_id}")
    async def download_artifact(request: Request, job_id: UUID, artifact_id: str) -> FileResponse:
        artifact = _manager(request).artifact(job_id, artifact_id)
        inline = artifact.path.suffix.lower() == ".html"
        return FileResponse(
            artifact.path,
            media_type=artifact.media_type,
            filename=artifact.filename,
            content_disposition_type="inline" if inline else "attachment",
            headers={
                "Cache-Control": "private, no-store",
                "X-Frame-Options": "SAMEORIGIN",
            },
        )

    application.mount(
        "/static",
        StaticFiles(directory=str(assets), check_dir=False, follow_symlink=False),
        name="static",
    )
    return application


def _manager(request: Request) -> JobManager:
    manager = getattr(request.app.state, "job_manager", None)
    if manager is None:
        raise ApiError(503, "service_unavailable", "The reconstruction service is not running.")
    return manager


def _accepted_job(snapshot: Mapping[str, Any]) -> JSONResponse:
    return JSONResponse(
        dict(snapshot),
        status_code=202,
        headers={"Location": str(snapshot["status_url"]), "Retry-After": "1"},
    )


def _validate_dimension(value: float, *, field: str) -> None:
    if not math.isfinite(value) or not 0 < value <= MAX_WIDTH_MM:
        raise ApiError(
            422,
            f"invalid_{field.removesuffix('_mm')}",
            f"{field} must be a finite number greater than 0 and at most {MAX_WIDTH_MM:g}.",
        )


def _validated_upload_suffix(
    upload: UploadFile,
    *,
    field: str,
) -> str:
    filename = upload.filename or ""
    if not filename or "\x00" in filename:
        raise ApiError(422, "invalid_filename", f"{field} must have a valid filename.")
    # Only the suffix is retained; path components and the user-controlled stem are discarded.
    suffix = Path(filename.replace("\\", "/").rsplit("/", 1)[-1]).suffix.lower()
    if suffix not in _IMAGE_SUFFIX_KIND:
        supported = ", ".join(sorted(_IMAGE_SUFFIX_KIND))
        raise ApiError(
            415,
            "unsupported_image_type",
            f"{field} must use one of these file extensions: {supported}.",
            details={"filename": _display_filename(filename)},
        )
    received_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    expected_types = _IMAGE_CONTENT_TYPES[_IMAGE_SUFFIX_KIND[suffix]]
    if received_type not in expected_types | _GENERIC_CONTENT_TYPES:
        raise ApiError(
            415,
            "content_type_mismatch",
            f"{field} has content type '{received_type}', which does not match {suffix}.",
            details={"filename": _display_filename(filename)},
        )
    return suffix


async def _save_upload(upload: UploadFile, destination: Path, maximum_bytes: int) -> tuple[int, bytes]:
    size = 0
    header = bytearray()
    try:
        with destination.open("xb") as output:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_bytes:
                    raise ApiError(
                        413,
                        "file_too_large",
                        (
                            f"'{_display_filename(upload.filename or '')}' exceeds the "
                            f"{_format_bytes(maximum_bytes)} limit."
                        ),
                        details={"maximum_bytes": maximum_bytes},
                    )
                if len(header) < 64:
                    header.extend(chunk[: 64 - len(header)])
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if size == 0:
        destination.unlink(missing_ok=True)
        raise ApiError(422, "empty_upload", f"'{_display_filename(upload.filename or '')}' is empty.")
    return size, bytes(header)


def _validate_file_signature(
    header: bytes,
    suffix: str,
    *,
    field: str,
) -> None:
    expected = _IMAGE_SUFFIX_KIND[suffix]
    detected = _detect_image(header)
    if detected != expected:
        raise ApiError(
            415,
            "file_signature_mismatch",
            f"{field} contents do not match the {suffix} file extension.",
        )


def _detect_image(header: bytes) -> str | None:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"BM"):
        return "bmp"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    return None


def _register_artifacts(manifest: object, output_dir: Path) -> dict[str, Artifact]:
    output_root = output_dir.resolve(strict=True)
    paths: dict[Path, None] = {}
    for candidate in _manifest_paths(manifest):
        path = candidate if candidate.is_absolute() else output_root / candidate
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in _ARTIFACT_SUFFIXES:
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(output_root)
        except (OSError, ValueError):
            continue
        paths[resolved] = None

    artifacts: dict[str, Artifact] = {}
    for path in sorted(paths, key=lambda item: item.name.lower()):
        base_id = _artifact_base_id(path)
        artifact_id = base_id
        counter = 2
        while artifact_id in artifacts:
            artifact_id = f"{base_id}-{counter}"
            counter += 1
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() in {".step", ".stp"}:
            media_type = "model/step"
        artifacts[artifact_id] = Artifact(
            artifact_id=artifact_id,
            filename=path.name,
            path=path,
            media_type=media_type,
            size_bytes=path.stat().st_size,
        )
    return artifacts


def _result_metadata(manifest: object) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy non-path export metadata into the public completion result."""
    metrics_value = (
        manifest.get("metrics")
        if isinstance(manifest, Mapping)
        else getattr(manifest, "metrics", None)
    )
    diagnostics_value = (
        manifest.get("input_diagnostics", ())
        if isinstance(manifest, Mapping)
        else getattr(manifest, "input_diagnostics", ())
    )
    metrics = _json_metadata(metrics_value)
    if not isinstance(metrics, dict):
        metrics = {}
    diagnostics: list[dict[str, Any]] = []
    if isinstance(diagnostics_value, Sequence) and not isinstance(
        diagnostics_value, (str, bytes, bytearray)
    ):
        for item in diagnostics_value:
            converted = _json_metadata(item)
            if isinstance(converted, dict):
                diagnostics.append(converted)
    return metrics, diagnostics


def _json_metadata(value: object, *, _depth: int = 0) -> Any:
    if _depth > 6 or value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.name
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_metadata(getattr(value, field.name), _depth=_depth + 1)
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_metadata(item, _depth=_depth + 1)
            for key, item in value.items()
            if isinstance(key, (str, int, float, bool))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_metadata(item, _depth=_depth + 1) for item in value]
    return None


def _manifest_paths(value: object, *, _seen: set[int] | None = None, _depth: int = 0) -> Iterable[Path]:
    """Yield path-like values explicitly present in an export manifest."""
    if value is None or _depth > 8:
        return
    if isinstance(value, Path):
        yield value
        return
    if isinstance(value, str):
        if Path(value).suffix.lower() in _ARTIFACT_SUFFIXES:
            yield Path(value)
        return
    if isinstance(value, (bytes, bytearray)):
        return

    seen = _seen if _seen is not None else set()
    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)

    children: Iterable[object]
    if isinstance(value, Mapping):
        children = value.values()
    elif is_dataclass(value) and not isinstance(value, type):
        children = (getattr(value, field.name) for field in fields(value))
    elif isinstance(value, Sequence):
        children = value
    elif hasattr(value, "model_dump") and callable(value.model_dump):
        children = value.model_dump().values()
    elif value.__class__.__module__.startswith("cadpro.") and hasattr(value, "__dict__"):
        children = vars(value).values()
    else:
        return
    for child in children:
        yield from _manifest_paths(child, _seen=seen, _depth=_depth + 1)


def _artifact_base_id(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"step", "stp"}:
        return "step"
    stem = _SAFE_ARTIFACT_ID.sub("-", path.stem.lower()).strip("-")
    return f"{stem or 'artifact'}-{suffix or 'file'}"


def _safe_worker_error(error: Exception, job_root: Path) -> str:
    if isinstance(error, (FileNotFoundError, RuntimeError, ValueError)):
        message = str(error).replace(str(job_root), "the uploaded job")
        message = " ".join(message.split())
        if message:
            return message[:500]
    return "Reconstruction failed unexpectedly. Check the capture guidance and try again."


def _display_filename(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(character for character in name if character.isprintable())[:128] or "upload"


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024 * 1024):g} GiB"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):g} MiB"
    if value >= 1024:
        return f"{value / 1024:g} KiB"
    return f"{value} bytes"


def _scope_header_values(scope: Mapping[str, Any], name: bytes) -> list[str]:
    return [
        value.decode("latin-1")
        for key, value in scope.get("headers", ())
        if key.lower() == name
    ]


def _parse_host_header(value: str) -> tuple[str, int | None] | None:
    if not value or any(character.isspace() for character in value):
        return None
    parsed = urlsplit(f"//{value}")
    try:
        port = parsed.port
    except ValueError:
        return None
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed.hostname.lower().rstrip("."), port


def _canonical_origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value.strip())
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("origin contains an invalid port") from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("origin must be an http(s) origin without credentials, path, query or fragment")
    scheme = parsed.scheme.lower()
    return scheme, parsed.hostname.lower().rstrip("."), port or (443 if scheme == "https" else 80)


def _configured_trusted_hosts(
    public_origin: str,
    explicit_hosts: Sequence[str] | None,
) -> tuple[str, ...]:
    hosts = set(_LOOPBACK_HOSTS)
    hosts.add(_canonical_origin(public_origin)[1])
    configured = explicit_hosts
    if configured is None:
        configured = tuple(
            item.strip()
            for item in os.environ.get("CADPRO_TRUSTED_HOSTS", "").split(",")
            if item.strip()
        )
    for value in configured:
        parsed = _parse_host_header(value)
        if parsed is None:
            raise ValueError(
                "trusted hosts must be plain hostnames or IP addresses, optionally with a port"
            )
        hosts.add(parsed[0])
    return tuple(sorted(hosts))


def _origin_is_allowed(
    scope: Mapping[str, Any],
    request_host: tuple[str, int | None],
    public_origin: tuple[str, str, int],
) -> bool:
    fetch_sites = [item.strip().lower() for item in _scope_header_values(scope, b"sec-fetch-site")]
    if len(fetch_sites) > 1 or fetch_sites == ["cross-site"]:
        return False
    origins = _scope_header_values(scope, b"origin")
    if not origins:
        return True
    if len(origins) != 1 or origins[0].strip().lower() == "null":
        return False
    try:
        origin = _canonical_origin(origins[0])
    except ValueError:
        return False
    if origin == public_origin:
        return True
    scheme = str(scope.get("scheme", "http")).lower()
    if scheme not in {"http", "https"}:
        return False
    request_origin = (
        scheme,
        request_host[0],
        request_host[1] or (443 if scheme == "https" else 80),
    )
    return request_host[0] in _LOOPBACK_HOSTS and origin == request_origin


def _guard_error(
    status_code: int,
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = dict(details)
    response_headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        **dict(headers or {}),
    }
    return JSONResponse({"error": error}, status_code=status_code, headers=response_headers)


def _configured_public_origin() -> str:
    origin = os.environ.get("CADPRO_PUBLIC_ORIGIN", DEFAULT_PUBLIC_ORIGIN).strip().rstrip("/")
    try:
        _canonical_origin(origin)
    except ValueError as error:
        raise ValueError(
            "CADPRO_PUBLIC_ORIGIN must be an http(s) origin without credentials, path, query or fragment"
        ) from error
    return origin


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


app = create_app(storage_parent=os.getenv("CADPRO_STORAGE_DIR"))

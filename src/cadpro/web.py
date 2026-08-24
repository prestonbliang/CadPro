"""Local web application and asynchronous reconstruction job service.

The browser never receives filesystem paths.  Uploads and generated artifacts live
in a private, per-process temporary directory and artifacts can only be downloaded
through the opaque IDs registered after a successful reconstruction.
"""

from __future__ import annotations

import asyncio
import html
import json
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
from dataclasses import dataclass, field, fields, is_dataclass, replace
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
from cadpro.enrichment import EnrichmentConfig, sanitize_query
from cadpro.media import (
    MAX_IMAGE_EDGE,
    MAX_IMAGE_PIXELS,
    validated_frame_dimensions,
    validated_image_dimensions,
    validated_image_size,
)
from cadpro.ml_mesh import ConceptMeshConfig
from cadpro.meshy import MeshyConfig, MeshyOptions
from cadpro.neural import NeuralCheckpointError, NeuralConfig, NeuralDepthModel


MIN_PHOTOS = 20
MAX_PHOTOS = 50
MIN_VIDEO_VIEWS = 20
MAX_VIDEO_VIEWS = 50
DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_PHOTO_SET_BYTES = 500 * 1024 * 1024
DEFAULT_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_JOB_TTL_SECONDS = 24 * 60 * 60
DEFAULT_JOB_SWEEP_SECONDS = 60.0
DEFAULT_MAX_PENDING_JOBS = 2
DEFAULT_PUBLIC_ORIGIN = "http://127.0.0.1:8000"
DEFAULT_REQUEST_OVERHEAD_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_TEXT_REQUEST_BYTES = 64 * 1024
RESEARCH_FRAME_MAX_EDGE = 1_024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_WIDTH_MM = 1_000_000.0
MAX_PROMPT_CHARS = 600

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
_VIDEO_SUFFIX_KIND = {
    ".avi": "avi",
    ".m4v": "iso-media",
    ".mkv": "ebml",
    ".mov": "iso-media",
    ".mp4": "iso-media",
    ".webm": "ebml",
}
_VIDEO_CONTENT_TYPES = {
    ".avi": {"video/x-msvideo", "video/avi"},
    ".m4v": {"video/x-m4v", "video/mp4"},
    ".mkv": {"video/x-matroska", "video/mkv"},
    ".mov": {"video/quicktime"},
    ".mp4": {"video/mp4"},
    ".webm": {"video/webm"},
}
_GENERIC_CONTENT_TYPES = {"", "application/octet-stream"}
_ARTIFACT_SUFFIXES = {
    ".glb",
    ".gltf",
    ".3mf",
    ".fbx",
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
    "research": 2,
    "segment": 3,
    "reconstruct": 4,
    "export": 5,
    "complete": 6,
}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_UPLOAD_RESERVATION_SCOPE_KEY = "cadpro.upload_reservation"


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
    kind: Literal["image", "photos", "video", "text"]
    root: Path
    input_dir: Path
    output_dir: Path
    input_paths: tuple[Path, ...]
    parameters: dict[str, Any]
    created_at: datetime
    created_monotonic: float
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    stage: Literal[
        "queued", "upload", "research", "segment", "reconstruct", "export", "complete"
    ] = "queued"
    progress: int = 10
    started_at: datetime | None = None
    finished_at: datetime | None = None
    finished_monotonic: float | None = None
    artifacts: dict[str, Artifact] | None = None
    metrics: dict[str, Any] | None = None
    input_diagnostics: list[dict[str, Any]] | None = None
    enrichment: dict[str, Any] | None = None
    neural_prediction: dict[str, Any] | None = None
    concept_mesh: dict[str, Any] | None = None
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
        stage: Literal[
            "queued", "upload", "research", "segment", "reconstruct", "export", "complete"
        ],
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


@dataclass(frozen=True)
class JobManifest:
    """Combine validated CAD exports with optional supplemental job artifacts."""

    cad: object | None
    concept_mesh_path: Path | None = None
    concept_mesh: Mapping[str, Any] | None = None
    visual_asset: object | None = None


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
        self._upload_reservations: set[UUID] = set()
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
            return self._accepting and self._admitted_count_locked() < self._max_pending_jobs

    def reserve_upload(self) -> UUID | None:
        """Reserve queue capacity before FastAPI parses and spools a multipart body."""
        with self._lock:
            self._prune_locked()
            if not self._accepting or self._admitted_count_locked() >= self._max_pending_jobs:
                return None
            reservation = uuid4()
            self._upload_reservations.add(reservation)
            return reservation

    def release_upload(self, reservation: UUID) -> None:
        """Release an unconsumed upload reservation; consuming twice is harmless."""
        with self._lock:
            self._upload_reservations.discard(reservation)

    def stage(self, *, upload_reservation: UUID | None = None) -> StagedJob:
        with self._lock:
            if not self._accepting:
                raise ApiError(503, "service_stopping", "The reconstruction service is stopping.")
            self._prune_locked()
            if upload_reservation is not None:
                if upload_reservation not in self._upload_reservations:
                    raise ApiError(
                        409,
                        "invalid_upload_reservation",
                        "The upload admission reservation is no longer active.",
                    )
                self._upload_reservations.remove(upload_reservation)
            elif self._admitted_count_locked() >= self._max_pending_jobs:
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
        kind: Literal["image", "photos", "video", "text"],
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
                        "enrichment": dict(job.enrichment) if job.enrichment else None,
                        "neural_prediction": (
                            dict(job.neural_prediction) if job.neural_prediction else None
                        ),
                        "concept_mesh": dict(job.concept_mesh) if job.concept_mesh else None,
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
            self._upload_reservations.clear()
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
            job.advance("upload", 15)
        try:
            manifest = self._runner(job)
            job.advance("export", 90)
            artifacts = _register_artifacts(manifest, job.output_dir)
            (
                metrics,
                input_diagnostics,
                enrichment,
                neural_prediction,
                concept_mesh,
            ) = _result_metadata(manifest)
            if not artifacts:
                raise RuntimeError("Reconstruction did not produce any downloadable artifacts.")
        except Exception as error:  # The worker must always reach a terminal state.
            with self._lock:
                with job._state_lock:
                    job.status = "failed"
                    job.advance("complete", 100)
                    job.error = {
                        "code": (
                            "generation_failed" if job.kind == "text" else "reconstruction_failed"
                        ),
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
                job.enrichment = enrichment
                job.neural_prediction = neural_prediction
                job.concept_mesh = concept_mesh
                job.status = "completed"
                job.advance("complete", 100)
                job.finished_at = datetime.now(timezone.utc)
                job.finished_monotonic = time.monotonic()

    def _pending_count_locked(self) -> int:
        return len(self._staged) + sum(
            job.status not in _TERMINAL_STATUSES for job in self._jobs.values()
        )

    def _admitted_count_locked(self) -> int:
        return self._pending_count_locked() + len(self._upload_reservations)

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


def _run_reconstruction(
    job: Job,
    *,
    neural_model: NeuralDepthModel | None = None,
    concept_mesh_config: ConceptMeshConfig | None = None,
    meshy_config: MeshyConfig | None = None,
) -> object:
    """The one integration seam between the web queue and reconstruction engine."""
    from cadpro.artifacts import export_artifacts
    from cadpro.enrichment import (
        EnrichmentConfig,
        EnrichmentError,
        EnrichmentReport,
        enrich_references,
        sanitize_query,
    )
    from cadpro.ml_mesh import (
        ConceptMeshConfig,
        ConceptMeshError,
        generate_concept_mesh,
    )
    from cadpro.meshy import MeshyError, generate_meshy_asset
    from cadpro.reconstruct import (
        reconstruct_photo_set,
        reconstruct_single_image,
        reconstruct_turntable_video,
    )

    active_meshy_config = meshy_config or MeshyConfig.from_env()
    active_concept_config = concept_mesh_config or ConceptMeshConfig.from_env()

    if job.kind == "text":
        if not active_meshy_config.available:
            raise RuntimeError("Text-to-3D generation is unavailable on this server.")
        job.advance("research", 24)
        try:
            generated = generate_meshy_asset(
                job.output_dir,
                prompt=str(job.parameters["prompt"]),
                config=active_meshy_config,
                options=_meshy_options(job.parameters),
                stem="cadpro-ai-asset",
            )
        except MeshyError:
            raise
        metadata = {
            "status": "completed",
            **dict(generated.metadata),
            "input_strategy": "text_prompt",
            "source_mode": "text",
            "warnings": list(generated.warnings),
        }
        job.advance("reconstruct", 82)
        job.advance("export", 94)
        return JobManifest(
            cad=None,
            concept_mesh_path=generated.glb_path,
            concept_mesh=metadata,
            visual_asset=generated,
        )

    enrichment: dict[str, Any] | None = None
    neural_prediction: dict[str, Any] | None = None
    concept_mesh: dict[str, Any] | None = None
    concept_mesh_path: Path | None = None
    optional_requested = bool(
        job.parameters.get("ai_enhance") or job.parameters.get("concept_mesh")
    )
    reference_images: tuple[Path, ...] | None = None
    reference_error: str | None = None
    if optional_requested:
        job.advance("research", 22)

        try:
            reference_images = _representative_image_paths(job)
        except Exception:
            reference_error = "Representative views could not be prepared for optional AI processing."

    if job.parameters.get("ai_enhance"):
        query = sanitize_query(str(job.parameters.get("object_hint", "")))
        if reference_images is None:
            enrichment = EnrichmentReport.failed(
                reference_error or "AI reference views were unavailable.",
                query,
            ).to_dict()
        else:
            try:
                enrichment = enrich_references(
                    reference_images,
                    query,
                    config=EnrichmentConfig.from_env(),
                ).to_dict()
            except EnrichmentError as error:
                enrichment = EnrichmentReport.failed(str(error), query).to_dict()
            except Exception:
                enrichment = EnrichmentReport.failed(
                    "AI/web reference enrichment failed; local reconstruction continued.",
                    query,
                ).to_dict()

    job.advance("segment", 35)
    if job.kind == "image":
        depth_mm = job.parameters.get("depth_mm")
        if job.parameters.get("neural_predict"):
            if neural_model is None:
                raise RuntimeError("The configured neural depth model is unavailable.")
            prediction = neural_model.predict(
                job.input_paths[0],
                measured_width_mm=job.parameters["width_mm"],
            )
            depth_mm = prediction.depth_mm
            neural_prediction = prediction.to_dict()
        if not isinstance(depth_mm, (int, float)):
            raise RuntimeError("One-photo reconstruction needs a depth value or neural prediction.")
        reconstruction = reconstruct_single_image(
            job.input_paths[0],
            width_mm=job.parameters["width_mm"],
            depth_mm=float(depth_mm),
            on_profile_ready=lambda: job.advance("reconstruct", 68),
        )
    elif job.kind == "photos":
        reconstruction = reconstruct_photo_set(
            job.input_paths,
            width_mm=job.parameters["width_mm"],
            clockwise=job.parameters["clockwise"],
        )
    else:
        reconstruction = reconstruct_turntable_video(
            job.input_paths[0],
            width_mm=job.parameters["width_mm"],
            views=job.parameters["views"],
            start_frame=job.parameters["start_frame"],
            end_frame=job.parameters["end_frame"],
            clockwise=job.parameters["clockwise"],
        )
    if enrichment is not None:
        reconstruction = replace(reconstruction, enrichment=enrichment)
    if neural_prediction is not None:
        reconstruction = replace(reconstruction, neural_prediction=neural_prediction)
    job.advance("reconstruct", 68)
    job.advance("export", 86)
    cad_manifest = export_artifacts(reconstruction, job.output_dir, stem="cadpro-model")

    visual_asset: object | None = None
    if job.parameters.get("concept_mesh"):
        job.advance("export", 94)
        if reference_images is None:
            concept_mesh = _failed_concept_mesh_metadata(
                reference_error or "A representative image was unavailable.",
                provider=("meshy" if active_meshy_config.available else None),
            )
        elif active_meshy_config.available:
            selected_images = _select_meshy_reference_images(reference_images)
            try:
                generated = generate_meshy_asset(
                    job.output_dir,
                    prompt=str(job.parameters.get("object_hint", "")).strip() or None,
                    image_paths=selected_images,
                    config=active_meshy_config,
                    options=_meshy_options(job.parameters),
                    stem="cadpro-ai-concept",
                )
                visual_asset = generated
                concept_mesh_path = generated.glb_path
                concept_mesh = {
                    "status": "completed",
                    **dict(generated.metadata),
                    "input_strategy": (
                        "single_reference"
                        if len(selected_images) == 1
                        else "evenly_spaced_multi_view"
                    ),
                    "source_mode": job.kind,
                    "source_view_count": len(selected_images),
                    "warnings": list(generated.warnings),
                }
            except MeshyError as error:
                concept_mesh = _failed_concept_mesh_metadata(
                    str(error), provider="meshy"
                )
            except Exception:
                concept_mesh = _failed_concept_mesh_metadata(
                    "AI visual-asset generation failed; validated CAD exports are still available.",
                    provider="meshy",
                )
        else:
            try:
                generated = generate_concept_mesh(
                    reference_images[0],
                    job.output_dir,
                    config=active_concept_config,
                )
                concept_mesh_path = generated.glb_path
                concept_mesh = {
                    "status": generated.status,
                    **dict(generated.metadata),
                    "input_strategy": "representative_view",
                    "source_name": job.input_paths[0].name,
                    "warnings": list(generated.warnings),
                }
            except ConceptMeshError as error:
                concept_mesh = _failed_concept_mesh_metadata(str(error))
            except Exception:
                concept_mesh = _failed_concept_mesh_metadata(
                    "Concept-mesh generation failed; validated CAD exports are still available."
                )

        try:
            _append_concept_mesh_report(
                cad_manifest.report_path,
                concept_mesh,
                concept_mesh_path,
            )
        except Exception:
            concept_mesh.setdefault("warnings", []).append(
                "Concept-mesh status could not be appended to the downloadable JSON report."
            )

    return JobManifest(
        cad=cad_manifest,
        concept_mesh_path=concept_mesh_path,
        concept_mesh=concept_mesh,
        visual_asset=visual_asset,
    )


def _failed_concept_mesh_metadata(
    warning: str,
    *,
    provider: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "provider": provider or "hunyuan-compatible-worker",
        "artifact_kind": "ai_concept_mesh",
        "format": "glb",
        "metric_scale": False,
        "manufacturing_cad": False,
        "derived_from_step": False,
        "input_strategy": "representative_view",
        "warnings": [warning],
    }


def _append_concept_mesh_report(
    report_path: Path,
    metadata: Mapping[str, Any],
    artifact_path: Path | None,
) -> None:
    document = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("CAD report root must be an object")
    document["concept_mesh"] = dict(metadata)
    if artifact_path is not None:
        artifacts = document.setdefault("artifacts", {})
        if isinstance(artifacts, dict):
            artifacts["concept_mesh"] = {
                "file": artifact_path.name,
                "bytes": artifact_path.stat().st_size,
            }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".cadpro-report-",
        suffix=".json",
        dir=report_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)


def _representative_image_paths(job: Job, maximum: int = 6) -> tuple[Path, ...]:
    """Return bounded still images for optional AI work, sampling video if needed."""
    if job.kind != "video":
        return job.input_paths

    import cv2

    source = job.input_paths[0]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("The video could not be decoded for optional AI processing.")
    try:
        try:
            reported_size = validated_image_dimensions(
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        except (OverflowError, ValueError) as error:
            raise ValueError(
                f"Video frame metadata is outside the image safety limits: {error}"
            ) from error
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start = int(job.parameters.get("start_frame", 0))
        configured_end = job.parameters.get("end_frame")
        stop = frame_count if configured_end is None else int(configured_end)
        if frame_count <= 0 or start < 0 or stop <= start or stop > frame_count:
            raise ValueError("The selected video range is invalid for optional AI analysis.")
        sample_count = min(maximum, stop - start)
        indices = tuple(start + (index * (stop - start)) // sample_count for index in range(sample_count))
        destination_dir = job.input_dir / "research-frames"
        destination_dir.mkdir(parents=False, exist_ok=False)
        paths: list[Path] = []
        for order, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("A representative video frame could not be decoded for AI analysis.")
            try:
                decoded_size = validated_frame_dimensions(frame)
            except ValueError as error:
                raise ValueError(
                    f"Video frame {frame_index} is outside the image safety limits: {error}"
                ) from error
            if decoded_size != reported_size:
                raise ValueError(
                    f"Video frame {frame_index} dimensions changed from "
                    f"{reported_size[0]} x {reported_size[1]} to "
                    f"{decoded_size[0]} x {decoded_size[1]}"
                )
            maximum_edge = max(decoded_size)
            if maximum_edge > RESEARCH_FRAME_MAX_EDGE:
                scale = RESEARCH_FRAME_MAX_EDGE / maximum_edge
                frame = cv2.resize(
                    frame,
                    (
                        max(1, round(decoded_size[0] * scale)),
                        max(1, round(decoded_size[1] * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            destination = destination_dir / f"frame-{order:02d}.jpg"
            if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]):
                raise RuntimeError("A representative video frame could not be prepared for AI analysis.")
            paths.append(destination)
        return tuple(paths)
    finally:
        capture.release()


def _select_meshy_reference_images(
    paths: Sequence[Path],
    maximum: int = 4,
) -> tuple[Path, ...]:
    """Choose stable, evenly spaced views within Meshy's documented 1-4 image limit."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if not paths:
        raise ValueError("At least one representative image is required.")
    if len(paths) <= maximum:
        return tuple(paths)
    indices = tuple((index * len(paths)) // maximum for index in range(maximum))
    return tuple(paths[index] for index in indices)


class RequestGuardMiddleware:
    """Reject untrusted browser requests and oversized uploads before endpoint parsing."""

    def __init__(
        self,
        app: Any,
        *,
        public_origin: str,
        trusted_hosts: Sequence[str],
        image_request_bytes: int,
        photo_request_bytes: int,
        video_request_bytes: int,
        text_request_bytes: int = DEFAULT_MAX_TEXT_REQUEST_BYTES,
    ) -> None:
        self.app = app
        self._public_origin = _canonical_origin(public_origin)
        self._trusted_hosts = frozenset(host.lower().rstrip(".") for host in trusted_hosts)
        self._request_limits = {
            "/api/jobs/image": image_request_bytes,
            "/api/jobs/photos": photo_request_bytes,
            "/api/jobs/video": video_request_bytes,
            "/api/jobs/text": text_request_bytes,
        }

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
        reservation: UUID | None = None
        if manager is not None:
            reservation = manager.reserve_upload()
            if reservation is None:
                await _guard_error(
                    503,
                    "job_queue_full",
                    "CadPro is already processing the maximum number of jobs. Try again shortly.",
                    headers={"Retry-After": "5"},
                )(scope, receive, send)
                return
            scope[_UPLOAD_RESERVATION_SCOPE_KEY] = reservation

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
        finally:
            if manager is not None and reservation is not None:
                manager.release_upload(reservation)
                scope.pop(_UPLOAD_RESERVATION_SCOPE_KEY, None)


def create_app(
    *,
    storage_parent: str | Path | None = None,
    asset_dir: str | Path | None = None,
    reconstruction_runner: ReconstructionRunner | None = None,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_photo_set_bytes: int = DEFAULT_MAX_PHOTO_SET_BYTES,
    max_video_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
    job_retention_seconds: float = DEFAULT_JOB_TTL_SECONDS,
    job_sweep_interval_seconds: float = DEFAULT_JOB_SWEEP_SECONDS,
    max_pending_jobs: int = DEFAULT_MAX_PENDING_JOBS,
    request_overhead_bytes: int = DEFAULT_REQUEST_OVERHEAD_BYTES,
    trusted_hosts: Sequence[str] | None = None,
) -> FastAPI:
    """Build the application; injectable limits and runner keep API tests lightweight."""
    for name, value in (
        ("max_image_bytes", max_image_bytes),
        ("max_photo_set_bytes", max_photo_set_bytes),
        ("max_video_bytes", max_video_bytes),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if job_retention_seconds < 0:
        raise ValueError("job_retention_seconds cannot be negative")
    if job_sweep_interval_seconds <= 0:
        raise ValueError("job_sweep_interval_seconds must be positive")
    if max_pending_jobs <= 0:
        raise ValueError("max_pending_jobs must be positive")
    if request_overhead_bytes < 0:
        raise ValueError("request_overhead_bytes cannot be negative")

    assets = Path(asset_dir).resolve() if asset_dir is not None else ASSET_DIR.resolve()
    public_origin = _configured_public_origin()
    allowed_hosts = _configured_trusted_hosts(public_origin, trusted_hosts)
    intelligence_config = EnrichmentConfig.from_env()
    concept_mesh_config = ConceptMeshConfig.from_env()
    meshy_config = MeshyConfig.from_env()
    neural_config = NeuralConfig.from_env()
    neural_model: NeuralDepthModel | None = None
    neural_checkpoint_valid = False
    if neural_config.available and neural_config.checkpoint is not None:
        try:
            neural_model = NeuralDepthModel.load(neural_config.checkpoint)
            neural_checkpoint_valid = True
        except NeuralCheckpointError:
            neural_model = None
    runner = (
        reconstruction_runner
        if reconstruction_runner is not None
        else lambda job: _run_reconstruction(
            job,
            neural_model=neural_model,
            concept_mesh_config=concept_mesh_config,
            meshy_config=meshy_config,
        )
    )
    image_request_bytes = max_image_bytes + request_overhead_bytes
    photo_request_bytes = min(
        max_photo_set_bytes, max_image_bytes * MAX_PHOTOS
    ) + request_overhead_bytes
    video_request_bytes = max_video_bytes + request_overhead_bytes
    text_request_bytes = DEFAULT_MAX_TEXT_REQUEST_BYTES

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
        description=(
            "Turn measured object captures into validated CAD, or generate an explicitly "
            "non-metric visual 3D asset from text or reference images."
        ),
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
        photo_request_bytes=photo_request_bytes,
        video_request_bytes=video_request_bytes,
        text_request_bytes=text_request_bytes,
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
                # Keep the v1.1 key for clients that already read it.
                "images": {"minimum": 1, "maximum": 1},
                "single_image": {"minimum": 1, "maximum": 1},
                "photos": {"minimum": MIN_PHOTOS, "maximum": MAX_PHOTOS},
                "video_views": {"minimum": MIN_VIDEO_VIEWS, "maximum": MAX_VIDEO_VIEWS},
                "image_file": {
                    "maximum_bytes": max_image_bytes,
                    "maximum_pixels": MAX_IMAGE_PIXELS,
                    "maximum_edge_pixels": MAX_IMAGE_EDGE,
                },
                "photo_set": {"maximum_bytes": max_photo_set_bytes},
                "video_file": {"maximum_bytes": max_video_bytes},
                "dimension_mm": {"maximum": MAX_WIDTH_MM},
            },
            "intelligence": {
                "available": intelligence_config.available,
                "provider": "openai",
                "model": intelligence_config.model if intelligence_config.available else None,
                "vision": intelligence_config.available,
                "web_search": intelligence_config.available,
                "geometry_mutation": False,
            },
            "neural_prediction": {
                "available": neural_model is not None,
                "enabled": neural_config.enabled,
                "checkpoint_valid": neural_checkpoint_valid,
                "model_type": "numpy_mlp_depth_regressor",
                "predicts": "depth_to_width_ratio",
                "changes_geometry": True,
                "requires_measured_width": True,
                "trained_examples": (
                    neural_model.trained_examples if neural_model is not None else None
                ),
                "validation_examples": (
                    neural_model.validation_examples if neural_model is not None else None
                ),
            },
            "concept_mesh": {
                "available": meshy_config.available or concept_mesh_config.available,
                "provider": (
                    "meshy"
                    if meshy_config.available
                    else "hunyuan-compatible-worker"
                ),
                "format": "glb",
                "metric_scale": False,
                "manufacturing_cad": False,
                "replaces_step": False,
                "multi_view_maximum": 4 if meshy_config.available else 1,
                "textured": meshy_config.available,
                "pbr": meshy_config.available,
                "rigging": meshy_config.available,
            },
            "generative_mesh": {
                "available": meshy_config.available,
                "provider": "meshy",
                "text_to_3d": meshy_config.available,
                "image_to_3d": meshy_config.available,
                "multi_image_to_3d": meshy_config.available,
                "multi_image_maximum": 4,
                "video_strategy": "four_evenly_spaced_frames",
                "textures": meshy_config.available,
                "pbr": meshy_config.available,
                "remesh": meshy_config.available,
                "rigging": {
                    "available": meshy_config.available,
                    "scope": "textured_standard_humanoids_only",
                },
                "formats": ["glb", "stl"],
                "metric_scale": False,
                "manufacturing_cad": False,
                "creates_step": False,
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
            float | None,
            Form(gt=0, le=MAX_WIDTH_MM, description="Desired extrusion depth in millimeters."),
        ] = None,
        neural_predict: Annotated[
            bool,
            Form(description="Predict extrusion depth with the configured trained neural model."),
        ] = False,
        ai_enhance: Annotated[
            bool,
            Form(description="Run optional vision analysis and cited web-reference research."),
        ] = False,
        object_hint: Annotated[
            str,
            Form(max_length=320, description="Optional object identity or research hint."),
        ] = "",
        concept_mesh: Annotated[
            bool,
            Form(description="Generate an optional non-metric concept GLB."),
        ] = False,
        mesh_texture: Annotated[bool, Form(description="Texture the optional AI mesh.")] = True,
        mesh_pbr: Annotated[bool, Form(description="Generate PBR maps for the AI mesh.")] = True,
        mesh_topology: Annotated[Literal["triangle", "quad"], Form()] = "triangle",
        mesh_target_faces: Annotated[int, Form(ge=100, le=300_000)] = 30_000,
        mesh_rig: Annotated[
            bool,
            Form(description="Attempt Meshy rigging for a standard textured humanoid."),
        ] = False,
        mesh_height_m: Annotated[float, Form(gt=0, le=10)] = 1.7,
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
        try:
            neural_parameters = _neural_parameters(
                neural_predict,
                depth_mm,
                neural_model,
            )
            intelligence_parameters = _intelligence_parameters(
                ai_enhance,
                object_hint,
                intelligence_config,
            )
            concept_mesh_parameters = _concept_mesh_parameters(
                concept_mesh,
                concept_mesh_config,
                meshy_config,
                texture=mesh_texture,
                pbr=mesh_pbr,
                topology=mesh_topology,
                target_faces=mesh_target_faces,
                rig_humanoid=mesh_rig,
                height_meters=mesh_height_m,
                object_hint=object_hint,
            )
        except Exception:
            await upload.close()
            raise
        manager = _manager(request)
        try:
            stage = manager.stage(upload_reservation=_upload_reservation(request))
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
                parameters={
                    "width_mm": width_mm,
                    **neural_parameters,
                    **intelligence_parameters,
                    **concept_mesh_parameters,
                },
            )
        except BaseException:
            manager.discard(stage)
            raise
        finally:
            await upload.close()
        return _accepted_job(manager.snapshot(job.job_id))

    @application.post("/api/jobs/photos", status_code=202)
    async def create_photo_job(
        request: Request,
        files: Annotated[
            list[UploadFile],
            File(
                description="20 to 50 object photos, repeated in rotation order.",
                json_schema_extra={"minItems": MIN_PHOTOS, "maxItems": MAX_PHOTOS},
            ),
        ],
        width_mm: Annotated[
            float,
            Form(gt=0, le=MAX_WIDTH_MM, description="Known maximum object width in millimeters."),
        ],
        clockwise: Annotated[
            bool,
            Form(description="Whether photo order rotates clockwise when viewed from above."),
        ] = False,
        ai_enhance: Annotated[
            bool,
            Form(description="Run optional vision analysis and cited web-reference research."),
        ] = False,
        object_hint: Annotated[
            str,
            Form(max_length=320, description="Optional object identity or research hint."),
        ] = "",
        concept_mesh: Annotated[
            bool,
            Form(description="Generate an optional non-metric concept GLB."),
        ] = False,
        mesh_texture: Annotated[bool, Form(description="Texture the optional AI mesh.")] = True,
        mesh_pbr: Annotated[bool, Form(description="Generate PBR maps for the AI mesh.")] = True,
        mesh_topology: Annotated[Literal["triangle", "quad"], Form()] = "triangle",
        mesh_target_faces: Annotated[int, Form(ge=100, le=300_000)] = 30_000,
        mesh_rig: Annotated[
            bool,
            Form(description="Attempt Meshy rigging for a standard textured humanoid."),
        ] = False,
        mesh_height_m: Annotated[float, Form(gt=0, le=10)] = 1.7,
    ) -> JSONResponse:
        if not MIN_PHOTOS <= len(files) <= MAX_PHOTOS:
            await _close_uploads(files)
            raise ApiError(
                422,
                "invalid_photo_count",
                f"Upload between {MIN_PHOTOS} and {MAX_PHOTOS} photos in rotation order.",
                details={"received": len(files), "minimum": MIN_PHOTOS, "maximum": MAX_PHOTOS},
            )
        _validate_dimension(width_mm, field="width_mm")
        try:
            intelligence_parameters = _intelligence_parameters(
                ai_enhance,
                object_hint,
                intelligence_config,
            )
            concept_mesh_parameters = _concept_mesh_parameters(
                concept_mesh,
                concept_mesh_config,
                meshy_config,
                texture=mesh_texture,
                pbr=mesh_pbr,
                topology=mesh_topology,
                target_faces=mesh_target_faces,
                rig_humanoid=mesh_rig,
                height_meters=mesh_height_m,
                object_hint=object_hint,
            )
        except Exception:
            await _close_uploads(files)
            raise
        manager = _manager(request)
        try:
            stage = manager.stage(upload_reservation=_upload_reservation(request))
        except Exception:
            await _close_uploads(files)
            raise
        saved: list[Path] = []
        total_bytes = 0
        try:
            for index, upload in enumerate(files, start=1):
                field = f"files[{index - 1}]"
                suffix = _validated_upload_suffix(upload, field=field, media_kind="image")
                destination = stage.input_dir / f"view-{index:03d}{suffix}"
                size, header = await _save_upload(upload, destination, max_image_bytes)
                _validate_file_signature(header, suffix, field=field, media_kind="image")
                try:
                    await asyncio.to_thread(validated_image_size, destination)
                except ValueError as error:
                    raise ApiError(422, "invalid_image", f"{field}: {error}") from error
                total_bytes += size
                if total_bytes > max_photo_set_bytes:
                    raise ApiError(
                        413,
                        "photo_set_too_large",
                        f"The complete photo set may not exceed {_format_bytes(max_photo_set_bytes)}.",
                        details={"maximum_bytes": max_photo_set_bytes},
                    )
                saved.append(destination)
            job = manager.submit(
                stage,
                kind="photos",
                input_paths=saved,
                parameters={
                    "width_mm": width_mm,
                    "clockwise": clockwise,
                    **intelligence_parameters,
                    **concept_mesh_parameters,
                },
            )
        except BaseException:
            manager.discard(stage)
            raise
        finally:
            await _close_uploads(files)
        return _accepted_job(manager.snapshot(job.job_id))

    @application.post("/api/jobs/video", status_code=202)
    async def create_video_job(
        request: Request,
        file: Annotated[UploadFile, File(description="One complete turntable video.")],
        width_mm: Annotated[
            float,
            Form(gt=0, le=MAX_WIDTH_MM, description="Known maximum object width in millimeters."),
        ],
        views: Annotated[int, Form(ge=MIN_VIDEO_VIEWS, le=MAX_VIDEO_VIEWS)] = 24,
        start_frame: Annotated[int, Form(ge=0)] = 0,
        end_frame: Annotated[int | None, Form(ge=1)] = None,
        clockwise: Annotated[bool, Form()] = False,
        ai_enhance: Annotated[
            bool,
            Form(description="Run optional vision analysis and cited web-reference research."),
        ] = False,
        object_hint: Annotated[
            str,
            Form(max_length=320, description="Optional object identity or research hint."),
        ] = "",
        concept_mesh: Annotated[
            bool,
            Form(description="Generate an optional non-metric concept GLB."),
        ] = False,
        mesh_texture: Annotated[bool, Form(description="Texture the optional AI mesh.")] = True,
        mesh_pbr: Annotated[bool, Form(description="Generate PBR maps for the AI mesh.")] = True,
        mesh_topology: Annotated[Literal["triangle", "quad"], Form()] = "triangle",
        mesh_target_faces: Annotated[int, Form(ge=100, le=300_000)] = 30_000,
        mesh_rig: Annotated[
            bool,
            Form(description="Attempt Meshy rigging for a standard textured humanoid."),
        ] = False,
        mesh_height_m: Annotated[float, Form(gt=0, le=10)] = 1.7,
    ) -> JSONResponse:
        _validate_dimension(width_mm, field="width_mm")
        try:
            intelligence_parameters = _intelligence_parameters(
                ai_enhance,
                object_hint,
                intelligence_config,
            )
            concept_mesh_parameters = _concept_mesh_parameters(
                concept_mesh,
                concept_mesh_config,
                meshy_config,
                texture=mesh_texture,
                pbr=mesh_pbr,
                topology=mesh_topology,
                target_faces=mesh_target_faces,
                rig_humanoid=mesh_rig,
                height_meters=mesh_height_m,
                object_hint=object_hint,
            )
        except Exception:
            await file.close()
            raise
        if end_frame is not None and end_frame <= start_frame:
            await file.close()
            raise ApiError(
                422,
                "invalid_frame_range",
                "end_frame must be greater than start_frame.",
                details={"start_frame": start_frame, "end_frame": end_frame},
            )
        manager = _manager(request)
        try:
            stage = manager.stage(upload_reservation=_upload_reservation(request))
        except Exception:
            await file.close()
            raise
        try:
            suffix = _validated_upload_suffix(file, field="file", media_kind="video")
            destination = stage.input_dir / f"turntable{suffix}"
            _size, header = await _save_upload(file, destination, max_video_bytes)
            _validate_file_signature(header, suffix, field="file", media_kind="video")
            job = manager.submit(
                stage,
                kind="video",
                input_paths=[destination],
                parameters={
                    "width_mm": width_mm,
                    "views": views,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "clockwise": clockwise,
                    **intelligence_parameters,
                    **concept_mesh_parameters,
                },
            )
        except BaseException:
            manager.discard(stage)
            raise
        finally:
            await file.close()
        return _accepted_job(manager.snapshot(job.job_id))

    @application.post("/api/jobs/text", status_code=202)
    async def create_text_job(
        request: Request,
        prompt: Annotated[
            str,
            Form(
                min_length=1,
                max_length=MAX_PROMPT_CHARS,
                description="A description of the visual 3D asset to generate.",
            ),
        ],
        mesh_texture: Annotated[bool, Form(description="Texture the generated mesh.")] = True,
        mesh_pbr: Annotated[bool, Form(description="Generate PBR maps for the mesh.")] = True,
        mesh_topology: Annotated[Literal["triangle", "quad"], Form()] = "triangle",
        mesh_target_faces: Annotated[int, Form(ge=100, le=300_000)] = 30_000,
        mesh_rig: Annotated[
            bool,
            Form(description="Attempt rigging for a standard textured humanoid only."),
        ] = False,
        mesh_height_m: Annotated[float, Form(gt=0, le=10)] = 1.7,
    ) -> JSONResponse:
        cleaned_prompt = " ".join(prompt.split())
        if not cleaned_prompt:
            raise ApiError(422, "invalid_prompt", "Enter a text description to generate a mesh.")
        if not meshy_config.available:
            raise ApiError(
                409,
                "generative_mesh_unavailable",
                "Text-to-3D generation is not configured on this server.",
            )
        mesh_parameters = _meshy_parameters(
            texture=mesh_texture,
            pbr=mesh_pbr,
            topology=mesh_topology,
            target_faces=mesh_target_faces,
            rig_humanoid=mesh_rig,
            height_meters=mesh_height_m,
        )
        manager = _manager(request)
        stage = manager.stage(upload_reservation=_upload_reservation(request))
        try:
            job = manager.submit(
                stage,
                kind="text",
                input_paths=(),
                parameters={
                    "prompt": cleaned_prompt,
                    "concept_mesh": True,
                    **mesh_parameters,
                },
            )
        except BaseException:
            manager.discard(stage)
            raise
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


def _upload_reservation(request: Request) -> UUID | None:
    value = request.scope.get(_UPLOAD_RESERVATION_SCOPE_KEY)
    return value if isinstance(value, UUID) else None


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


def _neural_parameters(
    requested: bool,
    depth_mm: float | None,
    model: NeuralDepthModel | None,
) -> dict[str, Any]:
    if requested:
        if model is None:
            raise ApiError(
                409,
                "neural_model_unavailable",
                (
                    "A trained neural checkpoint is not configured on this server. "
                    "Enter a measured extrusion depth or train and enable a checkpoint."
                ),
            )
        return {"neural_predict": True}
    if depth_mm is None:
        raise ApiError(
            422,
            "invalid_request",
            "Enter extrusion depth or request the configured neural depth prediction.",
        )
    _validate_dimension(depth_mm, field="depth_mm")
    return {"depth_mm": depth_mm}


def _intelligence_parameters(
    requested: bool,
    object_hint: str,
    config: EnrichmentConfig,
) -> dict[str, Any]:
    hint = sanitize_query(object_hint)
    if not requested:
        return {}
    if not config.available:
        raise ApiError(
            409,
            "intelligence_unavailable",
            (
                "AI/web research is not configured on this server. "
                "Local CAD reconstruction remains available."
            ),
        )
    return {"ai_enhance": True, "object_hint": hint}


def _concept_mesh_parameters(
    requested: bool,
    config: ConceptMeshConfig,
    meshy_config: MeshyConfig,
    *,
    texture: bool,
    pbr: bool,
    topology: Literal["triangle", "quad"],
    target_faces: int,
    rig_humanoid: bool,
    height_meters: float,
    object_hint: str,
) -> dict[str, Any]:
    if not requested:
        return {}
    if not meshy_config.available and not config.available:
        raise ApiError(
            409,
            "concept_mesh_unavailable",
            (
                "An optional AI mesh provider is not configured on this server. "
                "Validated STEP reconstruction remains available."
            ),
        )
    if not meshy_config.available:
        if rig_humanoid:
            raise ApiError(
                409,
                "mesh_rigging_unavailable",
                "Humanoid rigging requires the configured Meshy provider.",
            )
        return {"concept_mesh": True}
    parameters = {
        "concept_mesh": True,
        **_meshy_parameters(
            texture=texture,
            pbr=pbr,
            topology=topology,
            target_faces=target_faces,
            rig_humanoid=rig_humanoid,
            height_meters=(height_meters if rig_humanoid else None),
        ),
    }
    hint = sanitize_query(object_hint)
    if hint:
        parameters["object_hint"] = hint
    return parameters


def _meshy_parameters(
    *,
    texture: bool,
    pbr: bool,
    topology: Literal["triangle", "quad"],
    target_faces: int,
    rig_humanoid: bool,
    height_meters: float | None,
) -> dict[str, Any]:
    try:
        options = MeshyOptions(
            texture=texture,
            pbr=(pbr if texture else False),
            topology=topology,
            target_faces=target_faces,
            rig_humanoid=rig_humanoid,
            height_meters=(height_meters if rig_humanoid else None),
        )
    except (TypeError, ValueError) as error:
        raise ApiError(422, "invalid_mesh_options", str(error)) from None
    return {
        "mesh_texture": options.texture,
        "mesh_pbr": options.pbr,
        "mesh_topology": options.topology,
        "mesh_target_faces": options.target_faces,
        "mesh_rig": options.rig_humanoid,
        "mesh_height_m": options.height_meters,
    }


def _meshy_options(parameters: Mapping[str, Any]) -> MeshyOptions:
    return MeshyOptions(
        texture=bool(parameters.get("mesh_texture", True)),
        pbr=bool(parameters.get("mesh_pbr", True)),
        topology=str(parameters.get("mesh_topology", "triangle")),
        target_faces=int(parameters.get("mesh_target_faces", 30_000)),
        rig_humanoid=bool(parameters.get("mesh_rig", False)),
        height_meters=(
            float(parameters.get("mesh_height_m", 1.7))
            if bool(parameters.get("mesh_rig", False))
            else None
        ),
    )


def _validated_upload_suffix(
    upload: UploadFile,
    *,
    field: str,
    media_kind: Literal["image", "video"] = "image",
) -> str:
    filename = upload.filename or ""
    if not filename or "\x00" in filename:
        raise ApiError(422, "invalid_filename", f"{field} must have a valid filename.")
    # Only the suffix is retained; path components and the user-controlled stem are discarded.
    suffix = Path(filename.replace("\\", "/").rsplit("/", 1)[-1]).suffix.lower()
    allowed = _IMAGE_SUFFIX_KIND if media_kind == "image" else _VIDEO_SUFFIX_KIND
    if suffix not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ApiError(
            415,
            f"unsupported_{media_kind}_type",
            f"{field} must use one of these file extensions: {supported}.",
            details={"filename": _display_filename(filename)},
        )
    received_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    expected_types = (
        _IMAGE_CONTENT_TYPES[_IMAGE_SUFFIX_KIND[suffix]]
        if media_kind == "image"
        else _VIDEO_CONTENT_TYPES[suffix]
    )
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
    media_kind: Literal["image", "video"] = "image",
) -> None:
    expected = (_IMAGE_SUFFIX_KIND if media_kind == "image" else _VIDEO_SUFFIX_KIND)[suffix]
    detected = _detect_image(header) if media_kind == "image" else _detect_video(header)
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


def _detect_video(header: bytes) -> str | None:
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"AVI ":
        return "avi"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "iso-media"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "ebml"
    return None


async def _close_uploads(uploads: Iterable[UploadFile]) -> None:
    for upload in uploads:
        await upload.close()


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


def _result_metadata(
    manifest: object,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Copy non-path export metadata into the public completion result."""
    metadata_source = (
        manifest.get("cad", manifest)
        if isinstance(manifest, Mapping)
        else getattr(manifest, "cad", manifest)
    )
    metrics_value = (
        metadata_source.get("metrics")
        if isinstance(metadata_source, Mapping)
        else getattr(metadata_source, "metrics", None)
    )
    diagnostics_value = (
        metadata_source.get("input_diagnostics", ())
        if isinstance(metadata_source, Mapping)
        else getattr(metadata_source, "input_diagnostics", ())
    )
    enrichment_value = (
        metadata_source.get("enrichment")
        if isinstance(metadata_source, Mapping)
        else getattr(metadata_source, "enrichment", None)
    )
    neural_prediction_value = (
        metadata_source.get("neural_prediction")
        if isinstance(metadata_source, Mapping)
        else getattr(metadata_source, "neural_prediction", None)
    )
    concept_mesh_value = (
        manifest.get("concept_mesh")
        if isinstance(manifest, Mapping)
        else getattr(manifest, "concept_mesh", None)
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
    enrichment = _json_metadata(enrichment_value)
    if not isinstance(enrichment, dict):
        enrichment = None
    neural_prediction = _json_metadata(neural_prediction_value)
    if not isinstance(neural_prediction, dict):
        neural_prediction = None
    concept_mesh = _json_metadata(concept_mesh_value)
    if not isinstance(concept_mesh, dict):
        concept_mesh = None
    return metrics, diagnostics, enrichment, neural_prediction, concept_mesh


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

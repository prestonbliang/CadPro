"""SQLite-backed scan jobs, legal transitions, recovery, and cancellation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
import shutil
import sqlite3
import threading
from typing import Any, Protocol
from uuid import UUID, uuid4

from cadpro.scan.models import (
    ACTIVE_STAGES,
    ArtifactMetadata,
    InputMode,
    JobStage,
    JobStatus,
    ScanConfiguration,
    StructuredNotice,
)
from cadpro.scan.process import CancellationToken, ProcessCancelled


_STAGE_ORDER = {stage: index for index, stage in enumerate(JobStage)}
_TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


@dataclass(frozen=True)
class ScanWorkspace:
    job_id: UUID
    root: Path
    input_directory: Path
    working_directory: Path
    output_directory: Path


@dataclass(frozen=True)
class ScanJobContext:
    job_id: UUID
    workspace: ScanWorkspace
    input_paths: tuple[Path, ...]
    configuration: ScanConfiguration
    cancellation: CancellationToken
    progress: Callable[[JobStage, int], None]


@dataclass(frozen=True)
class ScanJobResult:
    artifacts: tuple[ArtifactMetadata, ...]
    report: dict[str, object]
    tool_versions: dict[str, str]
    warnings: tuple[StructuredNotice, ...] = ()


class ScanRunner(Protocol):
    def __call__(self, context: ScanJobContext) -> ScanJobResult: ...


class JobStore:
    """Own a stable job root and serialize all metadata changes through SQLite."""

    def __init__(self, storage_directory: str | Path) -> None:
        self.root = Path(storage_directory).expanduser().resolve()
        self.jobs_directory = self.root / "jobs"
        self.database_path = self.root / "jobs.sqlite3"
        self.jobs_directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL CHECK(progress BETWEEN 0 AND 100),
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL,
                    input_files_json TEXT NOT NULL,
                    input_metadata_json TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    tool_versions_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    report_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0, 1))
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_status_index ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS transitions_job_index ON transitions(job_id, sequence);
                """
            )

    def allocate_workspace(self) -> ScanWorkspace:
        job_id = uuid4()
        root = self.jobs_directory / str(job_id)
        inputs = root / "inputs"
        work = root / "work"
        outputs = root / "artifacts"
        inputs.mkdir(parents=True, exist_ok=False)
        work.mkdir(parents=False, exist_ok=False)
        outputs.mkdir(parents=False, exist_ok=False)
        return ScanWorkspace(job_id, root, inputs, work, outputs)

    def discard_workspace(self, workspace: ScanWorkspace) -> None:
        if self._safe_job_root(workspace.root):
            shutil.rmtree(workspace.root, ignore_errors=True)

    def create(
        self,
        workspace: ScanWorkspace,
        *,
        mode: InputMode,
        input_paths: Sequence[Path],
        input_metadata: Sequence[Mapping[str, object]],
        configuration: ScanConfiguration,
    ) -> dict[str, object]:
        canonical_root = self.jobs_directory / str(workspace.job_id)
        expected = ScanWorkspace(
            workspace.job_id,
            canonical_root,
            canonical_root / "inputs",
            canonical_root / "work",
            canonical_root / "artifacts",
        )
        supplied_paths = (
            workspace.root,
            workspace.input_directory,
            workspace.working_directory,
            workspace.output_directory,
        )
        expected_paths = (
            expected.root,
            expected.input_directory,
            expected.working_directory,
            expected.output_directory,
        )
        if any(
            supplied.resolve() != wanted.resolve()
            for supplied, wanted in zip(supplied_paths, expected_paths, strict=True)
        ):
            raise ValueError("The workspace is not managed by this job store.")
        if configuration.mode != mode:
            raise ValueError("configuration mode must match the job mode")
        relative_inputs: list[str] = []
        for input_path in input_paths:
            resolved = input_path.resolve(strict=True)
            try:
                relative = resolved.relative_to(workspace.input_directory.resolve(strict=True))
            except ValueError as error:
                raise ValueError("Every job input must remain inside its isolated input directory.") from error
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe input path.")
            relative_inputs.append(relative.as_posix())
        now = _utc_now()
        values = (
            str(workspace.job_id),
            mode.value,
            JobStatus.QUEUED.value,
            JobStage.QUEUED.value,
            0,
            _iso(now),
            _iso(now),
            _json(relative_inputs),
            _json(list(input_metadata)),
            configuration.model_dump_json(),
            "[]",
            "[]",
            "{}",
            "[]",
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, mode, status, stage, progress, created_at, updated_at,
                    input_files_json, input_metadata_json, configuration_json,
                    warnings_json, errors_json, tool_versions_json, artifacts_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self._insert_transition(
                connection,
                workspace.job_id,
                JobStatus.QUEUED,
                JobStage.QUEUED,
                0,
                now,
            )
        return self.snapshot(workspace.job_id)

    def snapshot(self, job_id: UUID | str) -> dict[str, object]:
        row = self._row(job_id)
        artifacts = _load_json(row["artifacts_json"], [])
        for artifact in artifacts:
            if isinstance(artifact, dict):
                artifact_id = artifact.get("artifact_id")
                if isinstance(artifact_id, str):
                    artifact["download_url"] = f"/api/v2/jobs/{row['id']}/artifacts/{artifact_id}"
        return {
            "id": row["id"],
            "mode": row["mode"],
            "status": row["status"],
            "stage": row["stage"],
            "progress": row["progress"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "updated_at": row["updated_at"],
            "input_metadata": _load_json(row["input_metadata_json"], []),
            "configuration": _load_json(row["configuration_json"], {}),
            "warnings": _load_json(row["warnings_json"], []),
            "errors": _load_json(row["errors_json"], []),
            "tool_versions": _load_json(row["tool_versions_json"], {}),
            "artifacts": artifacts,
            "report": _load_json(row["report_json"], None),
            "cancel_requested": bool(row["cancel_requested"]),
            "status_url": f"/api/v2/jobs/{row['id']}",
            "cancel_url": f"/api/v2/jobs/{row['id']}/cancel",
        }

    def context_values(
        self, job_id: UUID | str
    ) -> tuple[ScanWorkspace, tuple[Path, ...], ScanConfiguration]:
        row = self._row(job_id)
        identifier = UUID(str(row["id"]))
        root = self.jobs_directory / str(identifier)
        workspace = ScanWorkspace(
            identifier,
            root,
            root / "inputs",
            root / "work",
            root / "artifacts",
        )
        input_names = _load_json(row["input_files_json"], [])
        if not isinstance(input_names, list):
            raise RuntimeError("Stored job inputs are invalid.")
        input_paths: list[Path] = []
        input_root = workspace.input_directory.resolve(strict=True)
        for name in input_names:
            if not isinstance(name, str):
                raise RuntimeError("Stored job input name is invalid.")
            path = (input_root / Path(name)).resolve(strict=True)
            try:
                path.relative_to(input_root)
            except ValueError as error:
                raise RuntimeError("Stored job input escaped its workspace.") from error
            input_paths.append(path)
        configuration = ScanConfiguration.model_validate_json(row["configuration_json"])
        return workspace, tuple(input_paths), configuration

    def queued_ids(self) -> tuple[UUID, ...]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at",
                (JobStatus.QUEUED.value,),
            ).fetchall()
        return tuple(UUID(row["id"]) for row in rows)

    def transition(self, job_id: UUID | str, stage: JobStage, progress: int) -> None:
        if stage not in ACTIVE_STAGES:
            raise ValueError("Use a terminal method for a terminal stage.")
        if not 0 <= progress < 100:
            raise ValueError("Active progress must be between 0 and 99.")
        now = _utc_now()
        with self._lock, self._connect() as connection:
            row = self._select_row(connection, job_id)
            status = JobStatus(row["status"])
            current_stage = JobStage(row["stage"])
            current_progress = int(row["progress"])
            if status in _TERMINAL:
                raise RuntimeError("A terminal job cannot transition.")
            if _STAGE_ORDER[stage] < _STAGE_ORDER[current_stage] or progress < current_progress:
                raise ValueError("Job stage and progress cannot regress.")
            started_at = row["started_at"] or _iso(now)
            connection.execute(
                """
                UPDATE jobs SET status=?, stage=?, progress=?, started_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    JobStatus.RUNNING.value,
                    stage.value,
                    progress,
                    started_at,
                    _iso(now),
                    str(job_id),
                ),
            )
            self._insert_transition(
                connection, UUID(str(job_id)), JobStatus.RUNNING, stage, progress, now
            )

    def request_cancel(self, job_id: UUID | str) -> dict[str, object]:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            row = self._select_row(connection, job_id)
            status = JobStatus(row["status"])
            if status in _TERMINAL:
                return self.snapshot(job_id)
            if status == JobStatus.QUEUED:
                notice = StructuredNotice(
                    code="cancelled_by_user",
                    message="The queued reconstruction was cancelled by the user.",
                    stage=JobStage.QUEUED,
                )
                connection.execute(
                    """
                    UPDATE jobs SET status=?, stage=?, progress=100, finished_at=?, updated_at=?,
                    cancel_requested=1, errors_json=? WHERE id=?
                    """,
                    (
                        JobStatus.CANCELLED.value,
                        JobStage.CANCELLED.value,
                        _iso(now),
                        _iso(now),
                        _json([notice.model_dump(mode="json")]),
                        str(job_id),
                    ),
                )
                self._insert_transition(
                    connection,
                    UUID(str(job_id)),
                    JobStatus.CANCELLED,
                    JobStage.CANCELLED,
                    100,
                    now,
                )
            else:
                connection.execute(
                    "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?",
                    (_iso(now), str(job_id)),
                )
        return self.snapshot(job_id)

    def cancel_requested(self, job_id: UUID | str) -> bool:
        row = self._row(job_id)
        return bool(row["cancel_requested"])

    def complete(self, job_id: UUID | str, result: ScanJobResult) -> None:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            row = self._select_row(connection, job_id)
            if JobStatus(row["status"]) in _TERMINAL:
                raise RuntimeError("A terminal job cannot complete again.")
            warnings = [warning.model_dump(mode="json") for warning in result.warnings]
            connection.execute(
                """
                UPDATE jobs SET status=?, stage=?, progress=100, finished_at=?, updated_at=?,
                    warnings_json=?, tool_versions_json=?, artifacts_json=?, report_json=?
                WHERE id=?
                """,
                (
                    JobStatus.COMPLETED.value,
                    JobStage.COMPLETED.value,
                    _iso(now),
                    _iso(now),
                    _json(warnings),
                    _json(result.tool_versions),
                    _json([artifact.model_dump(mode="json") for artifact in result.artifacts]),
                    _json(result.report),
                    str(job_id),
                ),
            )
            self._insert_transition(
                connection,
                UUID(str(job_id)),
                JobStatus.COMPLETED,
                JobStage.COMPLETED,
                100,
                now,
            )

    def fail(
        self,
        job_id: UUID | str,
        notice: StructuredNotice,
        *,
        cancelled: bool = False,
    ) -> None:
        now = _utc_now()
        target_status = JobStatus.CANCELLED if cancelled else JobStatus.FAILED
        target_stage = JobStage.CANCELLED if cancelled else JobStage.FAILED
        with self._lock, self._connect() as connection:
            row = self._select_row(connection, job_id)
            if JobStatus(row["status"]) in _TERMINAL:
                return
            errors = _load_json(row["errors_json"], [])
            if not isinstance(errors, list):
                errors = []
            errors.append(notice.model_dump(mode="json"))
            connection.execute(
                """
                UPDATE jobs SET status=?, stage=?, progress=100, finished_at=?, updated_at=?,
                    errors_json=? WHERE id=?
                """,
                (
                    target_status.value,
                    target_stage.value,
                    _iso(now),
                    _iso(now),
                    _json(errors),
                    str(job_id),
                ),
            )
            self._insert_transition(
                connection,
                UUID(str(job_id)),
                target_status,
                target_stage,
                100,
                now,
            )

    def recover_interrupted(self) -> int:
        """Fail previously running jobs so a crashed worker never leaves false progress."""

        now = _utc_now()
        recovered = 0
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, errors_json, stage FROM jobs WHERE status=?",
                (JobStatus.RUNNING.value,),
            ).fetchall()
            for row in rows:
                notice = StructuredNotice(
                    code="worker_interrupted",
                    message=(
                        "The CadPro process stopped during reconstruction. The job was marked "
                        "failed instead of being left permanently running; submit it again."
                    ),
                    stage=JobStage(row["stage"]),
                )
                errors = _load_json(row["errors_json"], [])
                errors = errors if isinstance(errors, list) else []
                errors.append(notice.model_dump(mode="json"))
                connection.execute(
                    """
                    UPDATE jobs SET status=?, stage=?, progress=100, finished_at=?, updated_at=?,
                        errors_json=? WHERE id=?
                    """,
                    (
                        JobStatus.FAILED.value,
                        JobStage.FAILED.value,
                        _iso(now),
                        _iso(now),
                        _json(errors),
                        row["id"],
                    ),
                )
                self._insert_transition(
                    connection,
                    UUID(row["id"]),
                    JobStatus.FAILED,
                    JobStage.FAILED,
                    100,
                    now,
                )
                recovered += 1
        return recovered

    def transitions(self, job_id: UUID | str) -> list[dict[str, object]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, stage, progress, occurred_at FROM transitions
                WHERE job_id=? ORDER BY sequence
                """,
                (str(job_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def artifact_path(self, job_id: UUID | str, artifact_id: str) -> tuple[Path, dict[str, object]]:
        if not artifact_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in artifact_id):
            raise KeyError("artifact_not_found")
        row = self._row(job_id)
        if JobStatus(row["status"]) != JobStatus.COMPLETED:
            raise RuntimeError("Artifacts are available only for completed jobs.")
        artifacts = _load_json(row["artifacts_json"], [])
        metadata = next(
            (
                item
                for item in artifacts
                if isinstance(item, dict) and item.get("artifact_id") == artifact_id
            ),
            None,
        )
        if not isinstance(metadata, dict) or not isinstance(metadata.get("filename"), str):
            raise KeyError("artifact_not_found")
        output_root = (self.jobs_directory / str(UUID(str(job_id))) / "artifacts").resolve(strict=True)
        path = (output_root / metadata["filename"]).resolve(strict=True)
        try:
            path.relative_to(output_root)
        except ValueError as error:
            raise KeyError("artifact_not_found") from error
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("artifact_gone")
        return path, metadata

    def cleanup_expired(self, retention: timedelta) -> int:
        if retention.total_seconds() < 0:
            raise ValueError("retention cannot be negative")
        threshold = _iso(_utc_now() - retention)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM jobs WHERE status IN (?, ?, ?) AND finished_at <= ?
                """,
                (
                    JobStatus.COMPLETED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                    threshold,
                ),
            ).fetchall()
            identifiers = [UUID(row["id"]) for row in rows]
            connection.executemany("DELETE FROM jobs WHERE id=?", [(str(item),) for item in identifiers])
        for identifier in identifiers:
            root = self.jobs_directory / str(identifier)
            if self._safe_job_root(root):
                shutil.rmtree(root, ignore_errors=True)
        return len(identifiers)

    def _row(self, job_id: UUID | str) -> sqlite3.Row:
        with self._lock, self._connect() as connection:
            return self._select_row(connection, job_id)

    @staticmethod
    def _select_row(connection: sqlite3.Connection, job_id: UUID | str) -> sqlite3.Row:
        try:
            identifier = str(UUID(str(job_id)))
        except ValueError as error:
            raise KeyError("job_not_found") from error
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError("job_not_found")
        return row

    @staticmethod
    def _insert_transition(
        connection: sqlite3.Connection,
        job_id: UUID,
        status: JobStatus,
        stage: JobStage,
        progress: int,
        occurred_at: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO transitions(job_id, status, stage, progress, occurred_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(job_id), status.value, stage.value, progress, _iso(occurred_at)),
        )

    def _safe_job_root(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            resolved.relative_to(self.jobs_directory.resolve(strict=True))
            return resolved.parent == self.jobs_directory.resolve(strict=True) and bool(
                UUID(resolved.name)
            )
        except (OSError, ValueError):
            return False


class PersistentScanService:
    """One bounded local worker around the persistent store."""

    def __init__(
        self,
        store: JobStore,
        runner: ScanRunner,
        *,
        maximum_queued: int = 4,
        retention_seconds: float = 24 * 60 * 60,
        sweep_interval_seconds: float = 60,
    ) -> None:
        if maximum_queued <= 0:
            raise ValueError("maximum_queued must be positive")
        if retention_seconds < 0:
            raise ValueError("retention_seconds cannot be negative")
        if sweep_interval_seconds <= 0:
            raise ValueError("sweep_interval_seconds must be positive")
        self.store = store
        self.runner = runner
        self._retention_seconds = retention_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._queue: queue.Queue[UUID | None] = queue.Queue(maxsize=maximum_queued)
        self._stopping = threading.Event()
        self._active_job: UUID | None = None
        self._active_lock = threading.Lock()
        self.store.recover_interrupted()
        for job_id in self.store.queued_ids():
            try:
                self._queue.put_nowait(job_id)
            except queue.Full:
                break
        self._thread = threading.Thread(target=self._worker, name="cadpro-scan-worker", daemon=True)
        self._thread.start()
        self._sweeper = threading.Thread(
            target=self._sweep_loop,
            name="cadpro-scan-expiry-sweeper",
            daemon=True,
        )
        self._sweeper.start()

    def submit(self, job_id: UUID) -> None:
        if self._stopping.is_set():
            raise RuntimeError("The scan service is stopping.")
        try:
            self._queue.put_nowait(job_id)
        except queue.Full as error:
            raise RuntimeError("The persistent scan queue is full.") from error

    def cancel(self, job_id: UUID) -> dict[str, object]:
        return self.store.request_cancel(job_id)

    def close(self) -> None:
        self._stopping.set()
        with self._active_lock:
            active = self._active_job
        if active is not None:
            self.store.request_cancel(active)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=30)
        self._sweeper.join(timeout=min(30, self._sweep_interval_seconds + 1))

    def _sweep_loop(self) -> None:
        while not self._stopping.wait(self._sweep_interval_seconds):
            if self._retention_seconds > 0:
                self.store.cleanup_expired(timedelta(seconds=self._retention_seconds))

    def _worker(self) -> None:
        while not self._stopping.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job_id is None:
                return
            try:
                snapshot = self.store.snapshot(job_id)
                if snapshot["status"] != JobStatus.QUEUED.value:
                    continue
                with self._active_lock:
                    self._active_job = job_id

                def cancellation_probe(job_id: UUID = job_id) -> bool:
                    return self._stopping.is_set() or self.store.cancel_requested(job_id)

                def update_progress(
                    stage: JobStage, progress: int, job_id: UUID = job_id
                ) -> None:
                    self.store.transition(job_id, stage, progress)

                workspace, inputs, configuration = self.store.context_values(job_id)
                token = CancellationToken(probe=cancellation_probe)
                context = ScanJobContext(
                    job_id=job_id,
                    workspace=workspace,
                    input_paths=inputs,
                    configuration=configuration,
                    cancellation=token,
                    progress=update_progress,
                )
                result = self.runner(context)
                token.raise_if_cancelled()
                self.store.complete(job_id, result)
            except ProcessCancelled as error:
                self.store.fail(
                    job_id,
                    StructuredNotice(
                        code="cancelled_by_user",
                        message=str(error),
                        stage=_current_stage(self.store, job_id),
                    ),
                    cancelled=True,
                )
            except Exception as error:
                self.store.fail(
                    job_id,
                    StructuredNotice(
                        code="scan_pipeline_failed",
                        message=_safe_error(error),
                        stage=_current_stage(self.store, job_id),
                    ),
                )
            finally:
                with self._active_lock:
                    self._active_job = None
                self._queue.task_done()


def _current_stage(store: JobStore, job_id: UUID) -> JobStage:
    try:
        return JobStage(str(store.snapshot(job_id)["stage"]))
    except (KeyError, ValueError):
        return JobStage.FAILED


def _safe_error(error: Exception) -> str:
    if isinstance(error, (RuntimeError, ValueError, FileNotFoundError)):
        message = " ".join(str(error).split())
        if message:
            return message[:1000]
    return "The scan pipeline failed unexpectedly. Review the job log and capture guidance."


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _load_json(payload: str | None, fallback: Any) -> Any:
    if payload is None:
        return fallback
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return fallback

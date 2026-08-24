from __future__ import annotations

from datetime import timedelta
import hashlib
from pathlib import Path
import threading
import time
from uuid import uuid4

import pytest

from cadpro.scan.jobs import (
    JobStore,
    PersistentScanService,
    ScanJobContext,
    ScanJobResult,
    ScanWorkspace,
)
from cadpro.scan.models import (
    ArtifactKind,
    ArtifactMetadata,
    InputMode,
    JobStage,
    JobStatus,
    ScanConfiguration,
    StructuredNotice,
)


_TERMINAL_STATUSES = {
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
}


def _create_job(
    store: JobStore,
    *,
    filename: str = "capture.jpg",
    mode: InputMode = InputMode.PHOTOS,
) -> tuple[ScanWorkspace, Path]:
    workspace = store.allocate_workspace()
    source = workspace.input_directory / filename
    source.write_bytes(b"bounded scan fixture")
    store.create(
        workspace,
        mode=mode,
        input_paths=[source],
        input_metadata=[{"source_name": filename, "size_bytes": source.stat().st_size}],
        configuration=ScanConfiguration(mode=mode, use_gpu=False),
    )
    return workspace, source


def _empty_result(*, report: dict[str, object] | None = None) -> ScanJobResult:
    return ScanJobResult(
        artifacts=(),
        report=report or {"fixture": True},
        tool_versions={"fixture-runner": "1.0"},
    )


def _artifact_result(
    workspace: ScanWorkspace,
    *,
    artifact_id: str = "textured-model",
    filename: str = "model.glb",
    payload: bytes = b"tiny glb fixture",
) -> ScanJobResult:
    destination = workspace.output_directory / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    artifact = ArtifactMetadata(
        artifact_id=artifact_id,
        kind=ArtifactKind.TEXTURED_MODEL,
        filename=filename,
        media_type="model/gltf-binary",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        textured=True,
        metric_scale=False,
    )
    return ScanJobResult(
        artifacts=(artifact,),
        report={"quality_class": "usable"},
        tool_versions={"fixture-runner": "1.0"},
    )


def _wait_for_status(
    store: JobStore,
    job_id,
    expected: JobStatus,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = store.snapshot(job_id)
        status = snapshot["status"]
        if status == expected.value:
            return snapshot
        if status in _TERMINAL_STATUSES:
            raise AssertionError(f"job reached {status!r}, expected {expected.value!r}")
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {expected.value!r} before timeout")


def test_workspace_allocation_and_inputs_are_confined(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    workspace = store.allocate_workspace()
    source = workspace.input_directory / "inside.jpg"
    source.write_bytes(b"inside")

    assert workspace.root.parent == store.jobs_directory
    assert workspace.input_directory.parent == workspace.root
    assert workspace.working_directory.parent == workspace.root
    assert workspace.output_directory.parent == workspace.root

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    configuration = ScanConfiguration(mode=InputMode.PHOTOS)
    with pytest.raises(ValueError, match="isolated input directory"):
        store.create(
            workspace,
            mode=InputMode.PHOTOS,
            input_paths=[outside],
            input_metadata=[],
            configuration=configuration,
        )

    store.create(
        workspace,
        mode=InputMode.PHOTOS,
        input_paths=[source],
        input_metadata=[],
        configuration=configuration,
    )
    restored_workspace, restored_inputs, _ = store.context_values(workspace.job_id)
    assert restored_workspace == workspace
    assert restored_inputs == (source.resolve(),)


def test_store_rejects_a_workspace_outside_its_job_root(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    outside_root = tmp_path / "unmanaged-workspace"
    outside_inputs = outside_root / "inputs"
    outside_work = outside_root / "work"
    outside_outputs = outside_root / "artifacts"
    outside_inputs.mkdir(parents=True)
    outside_work.mkdir()
    outside_outputs.mkdir()
    source = outside_inputs / "capture.jpg"
    source.write_bytes(b"outside")
    workspace = ScanWorkspace(
        job_id=uuid4(),
        root=outside_root,
        input_directory=outside_inputs,
        working_directory=outside_work,
        output_directory=outside_outputs,
    )

    with pytest.raises(ValueError, match="workspace"):
        store.create(
            workspace,
            mode=InputMode.PHOTOS,
            input_paths=[source],
            input_metadata=[],
            configuration=ScanConfiguration(mode=InputMode.PHOTOS),
        )

    sentinel = outside_root / "keep.txt"
    sentinel.write_text("must remain", encoding="utf-8")
    store.discard_workspace(workspace)
    assert sentinel.read_text(encoding="utf-8") == "must remain"


def test_transitions_are_legal_ordered_and_non_regressing(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    workspace, _ = _create_job(store)

    store.transition(workspace.job_id, JobStage.VALIDATING, 5)
    store.transition(workspace.job_id, JobStage.ANALYZING_IMAGES, 20)
    store.transition(workspace.job_id, JobStage.ANALYZING_IMAGES, 30)

    snapshot = store.snapshot(workspace.job_id)
    assert snapshot["status"] == JobStatus.RUNNING.value
    assert snapshot["stage"] == JobStage.ANALYZING_IMAGES.value
    assert snapshot["progress"] == 30
    assert snapshot["started_at"] is not None
    with pytest.raises(ValueError, match="cannot regress"):
        store.transition(workspace.job_id, JobStage.VALIDATING, 31)
    with pytest.raises(ValueError, match="cannot regress"):
        store.transition(workspace.job_id, JobStage.BUILDING_MESH, 29)
    with pytest.raises(ValueError, match="terminal method"):
        store.transition(workspace.job_id, JobStage.COMPLETED, 99)
    with pytest.raises(ValueError, match="between 0 and 99"):
        store.transition(workspace.job_id, JobStage.BUILDING_MESH, 100)

    transitions = store.transitions(workspace.job_id)
    assert [(item["stage"], item["progress"]) for item in transitions] == [
        (JobStage.QUEUED.value, 0),
        (JobStage.VALIDATING.value, 5),
        (JobStage.ANALYZING_IMAGES.value, 20),
        (JobStage.ANALYZING_IMAGES.value, 30),
    ]


def test_queued_service_cancellation_is_terminal_and_runner_never_receives_job(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    first_started = threading.Event()
    release_first = threading.Event()
    seen: list[object] = []

    def runner(context: ScanJobContext) -> ScanJobResult:
        seen.append(context.job_id)
        first_started.set()
        while not release_first.wait(0.01):
            context.cancellation.raise_if_cancelled()
        return _empty_result()

    service = PersistentScanService(store, runner, maximum_queued=2)
    first, _ = _create_job(store, filename="first.jpg")
    second, _ = _create_job(store, filename="second.jpg")
    try:
        service.submit(first.job_id)
        assert first_started.wait(2)
        service.submit(second.job_id)

        cancelled = service.cancel(second.job_id)

        assert cancelled["status"] == JobStatus.CANCELLED.value
        assert cancelled["stage"] == JobStage.CANCELLED.value
        assert cancelled["progress"] == 100
        assert cancelled["cancel_requested"] is True
        assert cancelled["errors"][0]["code"] == "cancelled_by_user"
        release_first.set()
        _wait_for_status(store, first.job_id, JobStatus.COMPLETED)
        assert second.job_id not in seen
    finally:
        release_first.set()
        service.close()


def test_running_service_cancellation_reaches_cancelled_terminal_state(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    runner_started = threading.Event()

    def runner(context: ScanJobContext) -> ScanJobResult:
        context.progress(JobStage.VALIDATING, 10)
        runner_started.set()
        while True:
            context.cancellation.raise_if_cancelled()
            time.sleep(0.01)

    service = PersistentScanService(store, runner)
    workspace, _ = _create_job(store)
    try:
        service.submit(workspace.job_id)
        assert runner_started.wait(2)

        requested = service.cancel(workspace.job_id)
        snapshot = _wait_for_status(store, workspace.job_id, JobStatus.CANCELLED)

        assert requested["cancel_requested"] is True
        assert snapshot["stage"] == JobStage.CANCELLED.value
        assert snapshot["progress"] == 100
        assert snapshot["errors"][-1]["code"] == "cancelled_by_user"
        assert snapshot["finished_at"] is not None
    finally:
        service.close()


def test_terminal_job_is_immutable(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    workspace, _ = _create_job(store)
    store.complete(workspace.job_id, _empty_result())
    completed = store.snapshot(workspace.job_id)
    transition_count = len(store.transitions(workspace.job_id))

    with pytest.raises(RuntimeError, match="terminal job"):
        store.transition(workspace.job_id, JobStage.EXPORTING, 90)
    with pytest.raises(RuntimeError, match="cannot complete again"):
        store.complete(workspace.job_id, _empty_result(report={"changed": True}))
    store.fail(
        workspace.job_id,
        StructuredNotice(code="late_failure", message="must not replace success"),
    )
    after_failure = store.snapshot(workspace.job_id)
    after_cancel = store.request_cancel(workspace.job_id)

    assert after_failure == completed
    assert after_cancel == completed
    assert len(store.transitions(workspace.job_id)) == transition_count


def test_service_startup_recovers_interrupted_running_jobs(tmp_path):
    root = tmp_path / "scan-store"
    first_store = JobStore(root)
    workspace, _ = _create_job(first_store)
    first_store.transition(workspace.job_id, JobStage.BUILDING_DENSE_CLOUD, 45)
    runner_called = threading.Event()

    def runner(_context: ScanJobContext) -> ScanJobResult:
        runner_called.set()
        return _empty_result()

    restarted_store = JobStore(root)
    service = PersistentScanService(restarted_store, runner)
    try:
        snapshot = restarted_store.snapshot(workspace.job_id)
        assert snapshot["status"] == JobStatus.FAILED.value
        assert snapshot["stage"] == JobStage.FAILED.value
        assert snapshot["progress"] == 100
        assert snapshot["errors"][-1]["code"] == "worker_interrupted"
        assert snapshot["errors"][-1]["stage"] == JobStage.BUILDING_DENSE_CLOUD.value
        assert runner_called.is_set() is False
    finally:
        service.close()


def test_job_state_persists_across_store_instances(tmp_path):
    root = tmp_path / "scan-store"
    first_store = JobStore(root)
    workspace, source = _create_job(first_store, mode=InputMode.VIDEO, filename="orbit.mp4")
    first_store.transition(workspace.job_id, JobStage.EXTRACTING_FRAMES, 12)

    second_store = JobStore(root)
    snapshot = second_store.snapshot(workspace.job_id)
    restored_workspace, restored_inputs, configuration = second_store.context_values(
        workspace.job_id
    )

    assert snapshot["mode"] == InputMode.VIDEO.value
    assert snapshot["status"] == JobStatus.RUNNING.value
    assert snapshot["stage"] == JobStage.EXTRACTING_FRAMES.value
    assert snapshot["progress"] == 12
    assert restored_workspace == workspace
    assert restored_inputs == (source.resolve(),)
    assert configuration.mode == InputMode.VIDEO
    second_store.transition(workspace.job_id, JobStage.ANALYZING_IMAGES, 20)
    assert first_store.snapshot(workspace.job_id)["progress"] == 20


def test_artifacts_reject_traversal_and_cross_job_access(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    first, _ = _create_job(store, filename="first.jpg")
    first_result = _artifact_result(first, artifact_id="first-model", filename="first.glb")
    store.complete(first.job_id, first_result)
    path, metadata = store.artifact_path(first.job_id, "first-model")
    assert path == (first.output_directory / "first.glb").resolve()
    assert metadata["artifact_id"] == "first-model"

    second, _ = _create_job(store, filename="second.jpg")
    cross_job_filename = f"../../{first.job_id}/artifacts/first.glb"
    stolen = ArtifactMetadata(
        artifact_id="stolen-model",
        kind=ArtifactKind.TEXTURED_MODEL,
        filename=cross_job_filename,
        media_type="model/gltf-binary",
        size_bytes=(first.output_directory / "first.glb").stat().st_size,
        sha256=first_result.artifacts[0].sha256,
    )
    store.complete(
        second.job_id,
        ScanJobResult(artifacts=(stolen,), report={}, tool_versions={}),
    )

    with pytest.raises(KeyError, match="artifact_not_found"):
        store.artifact_path(first.job_id, "../first-model")
    with pytest.raises(KeyError, match="artifact_not_found"):
        store.artifact_path(second.job_id, "first-model")
    with pytest.raises(KeyError, match="artifact_not_found"):
        store.artifact_path(second.job_id, "stolen-model")


def test_cleanup_removes_only_expired_terminal_jobs_and_workspaces(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    completed, _ = _create_job(store, filename="completed.jpg")
    failed, _ = _create_job(store, filename="failed.jpg")
    cancelled, _ = _create_job(store, filename="cancelled.jpg")
    queued, _ = _create_job(store, filename="queued.jpg")
    store.complete(completed.job_id, _empty_result())
    store.fail(
        failed.job_id,
        StructuredNotice(code="fixture_failed", message="expected fixture failure"),
    )
    store.request_cancel(cancelled.job_id)

    removed = store.cleanup_expired(timedelta(0))

    assert removed == 3
    for workspace in (completed, failed, cancelled):
        assert workspace.root.exists() is False
        with pytest.raises(KeyError, match="job_not_found"):
            store.snapshot(workspace.job_id)
    assert queued.root.is_dir()
    assert store.snapshot(queued.job_id)["status"] == JobStatus.QUEUED.value
    with pytest.raises(ValueError, match="cannot be negative"):
        store.cleanup_expired(timedelta(seconds=-1))


def test_persistent_service_sweeps_expired_terminal_jobs(tmp_path):
    store = JobStore(tmp_path / "scan-store")
    workspace, _ = _create_job(store)
    store.fail(
        workspace.job_id,
        StructuredNotice(code="fixture_failed", message="expected fixture failure"),
    )
    service = PersistentScanService(
        store,
        lambda _context: _empty_result(),
        retention_seconds=0.001,
        sweep_interval_seconds=0.01,
    )
    try:
        deadline = time.monotonic() + 2
        while workspace.root.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert workspace.root.exists() is False
        with pytest.raises(KeyError, match="job_not_found"):
            store.snapshot(workspace.job_id)
    finally:
        service.close()


def test_persistent_service_completes_with_tiny_injected_runner(tmp_path):
    store = JobStore(tmp_path / "scan-store")

    def runner(context: ScanJobContext) -> ScanJobResult:
        assert context.input_paths[0].read_bytes() == b"bounded scan fixture"
        context.progress(JobStage.VALIDATING, 10)
        context.progress(JobStage.EXPORTING, 85)
        return _artifact_result(context.workspace)

    service = PersistentScanService(store, runner)
    workspace, _ = _create_job(store)
    try:
        service.submit(workspace.job_id)
        snapshot = _wait_for_status(store, workspace.job_id, JobStatus.COMPLETED)

        assert snapshot["progress"] == 100
        assert snapshot["report"] == {"quality_class": "usable"}
        assert snapshot["tool_versions"] == {"fixture-runner": "1.0"}
        assert snapshot["artifacts"][0]["artifact_id"] == "textured-model"
        assert snapshot["artifacts"][0]["download_url"].endswith(
            "/artifacts/textured-model"
        )
        artifact_path, _ = store.artifact_path(workspace.job_id, "textured-model")
        assert artifact_path.read_bytes() == b"tiny glb fixture"
    finally:
        service.close()


def test_persistent_service_records_safe_runner_failure(tmp_path):
    store = JobStore(tmp_path / "scan-store")

    def runner(context: ScanJobContext) -> ScanJobResult:
        context.progress(JobStage.VALIDATING, 10)
        raise ValueError("fixture reconstruction failed")

    service = PersistentScanService(store, runner)
    workspace, _ = _create_job(store)
    try:
        service.submit(workspace.job_id)
        snapshot = _wait_for_status(store, workspace.job_id, JobStatus.FAILED)

        assert snapshot["stage"] == JobStage.FAILED.value
        assert snapshot["progress"] == 100
        assert snapshot["errors"][-1] == {
            "code": "scan_pipeline_failed",
            "details": {},
            "message": "fixture reconstruction failed",
            "stage": JobStage.VALIDATING.value,
        }
        assert snapshot["finished_at"] is not None
    finally:
        service.close()

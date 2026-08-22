from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from cadpro.web import create_app


PNG = b"\x89PNG\r\n\x1a\n" + b"test-image"
MP4 = b"\x00\x00\x00\x18ftypisom" + b"test-video"


def _assets(tmp_path: Path) -> Path:
    assets = tmp_path / "web-assets"
    assets.mkdir(parents=True)
    (assets / "index.html").write_text(
        '<meta property="og:url" content="__CADPRO_ORIGIN__/"><h1>CadPro</h1>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("window.cadpro = true;", encoding="utf-8")
    return assets


def _photo_files(count: int = 20, *, body: bytes = PNG, name: str = "photo.png"):
    return [("files", (f"{index:02d}-{name}", body, "image/png")) for index in range(count)]


def _wait_for_terminal(client: TestClient, job_id: str) -> dict:
    for _ in range(200):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.005)
    raise AssertionError("reconstruction job did not reach a terminal state")


def _artifact_runner(seen: list | None = None, outside: Path | None = None):
    def run(job):
        if seen is not None:
            seen.append(job)
        step = job.output_dir / "cadpro-model.step"
        step.write_bytes(b"ISO-10303-21;\nEND-ISO-10303-21;")
        values = {
            "step_path": step,
            "metrics": {
                "dimensions_mm": (10.0, 20.0, 30.0),
                "volume_mm3": 6000.0,
                "is_valid": True,
            },
            "input_diagnostics": [
                {"order": 0, "source_name": job.input_paths[0].name, "outline_points": 12}
            ],
        }
        if outside is not None:
            values["outside"] = outside
        return values

    return run


def test_index_replaces_configured_origin_and_static_assets_are_served(tmp_path, monkeypatch):
    assets = _assets(tmp_path)
    monkeypatch.setenv("CADPRO_PUBLIC_ORIGIN", "https://cad.example")
    app = create_app(
        storage_parent=tmp_path / "jobs",
        asset_dir=assets,
        reconstruction_runner=_artifact_runner(),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        rejected = client.get("/", headers={"host": "attacker.invalid"})
        index = client.get("/", headers={"host": "cad.example"})
        static = client.get("/static/app.js")

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "untrusted_host"
    assert index.status_code == 200
    assert "https://cad.example/" in index.text
    assert "__CADPRO_ORIGIN__" not in index.text
    assert "attacker.invalid" not in index.text
    assert static.status_code == 200
    assert static.text == "window.cadpro = true;"


def test_photo_job_preserves_order_uses_safe_names_and_downloads_registered_artifact(tmp_path):
    seen = []
    storage = tmp_path / "storage"
    outside = tmp_path / "existing-secret.step"
    outside.write_bytes(b"do not expose")
    app = create_app(
        storage_parent=storage,
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(seen, outside),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/api/jobs/photos",
            data={"width_mm": "125.5", "clockwise": "true"},
            files=_photo_files(name="..\\..\\part.png"),
        )
        assert response.status_code == 202
        assert response.headers["location"] == response.json()["status_url"]
        job = _wait_for_terminal(client, response.json()["id"])
        assert job["status"] == "completed"
        assert job["stage"] == "complete"
        assert job["progress"] == 100
        assert job["input_count"] == 20
        assert job["parameters"] == {"width_mm": 125.5, "clockwise": True}
        assert len(job["result"]["artifacts"]) == 1
        assert job["result"]["metrics"]["dimensions_mm"] == [10.0, 20.0, 30.0]
        assert job["result"]["input_diagnostics"][0]["source_name"] == "view-001.png"

        artifact = job["result"]["artifacts"][0]
        download = client.get(artifact["download_url"])
        missing = client.get(f"/api/jobs/{job['id']}/artifacts/existing-secret-step")

        assert download.status_code == 200
        assert download.content.startswith(b"ISO-10303-21")
        assert "attachment" in download.headers["content-disposition"]
        assert missing.status_code == 404
        assert outside.read_bytes() == b"do not expose"
        assert [path.name for path in seen[0].input_paths] == [
            f"view-{index:03d}.png" for index in range(1, 21)
        ]
        session_root = app.state.job_manager.root
        assert session_root.is_dir()

    assert storage.is_dir()
    assert not session_root.exists()
    assert outside.exists()


def test_photo_count_content_and_size_errors_are_actionable_and_clean_staging(tmp_path):
    storage = tmp_path / "storage"
    app = create_app(
        storage_parent=storage,
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
        max_image_bytes=len(PNG) - 1,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        too_few = client.post(
            "/api/jobs/photos",
            data={"width_mm": "80"},
            files=_photo_files(19),
        )
        bad_magic = client.post(
            "/api/jobs/photos",
            data={"width_mm": "80"},
            files=_photo_files(body=b"not a png"),
        )
        too_large = client.post(
            "/api/jobs/photos",
            data={"width_mm": "80"},
            files=_photo_files(),
        )

        assert too_few.status_code == 422
        assert too_few.json()["error"]["code"] == "invalid_photo_count"
        # The invalid body is below the size limit and reaches signature validation.
        assert bad_magic.status_code == 415
        assert bad_magic.json()["error"]["code"] == "file_signature_mismatch"
        assert too_large.status_code == 413
        assert too_large.json()["error"]["code"] == "file_too_large"
        assert list(app.state.job_manager.root.iterdir()) == []


def test_video_job_validates_fields_and_passes_safe_path_and_parameters(tmp_path):
    seen = []
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(seen),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        invalid_range = client.post(
            "/api/jobs/video",
            data={"width_mm": "90", "views": "24", "start_frame": "10", "end_frame": "10"},
            files={"file": ("clip.mp4", MP4, "video/mp4")},
        )
        invalid_views = client.post(
            "/api/jobs/video",
            data={"width_mm": "90", "views": "8"},
            files={"file": ("clip.mp4", MP4, "video/mp4")},
        )
        accepted = client.post(
            "/api/jobs/video",
            data={
                "width_mm": "90",
                "views": "30",
                "start_frame": "5",
                "end_frame": "105",
                "clockwise": "true",
            },
            files={"file": ("..\\..\\private.mp4", MP4, "video/mp4")},
        )
        job = _wait_for_terminal(client, accepted.json()["id"])

    assert invalid_range.status_code == 422
    assert invalid_range.json()["error"]["code"] == "invalid_frame_range"
    assert invalid_views.status_code == 422
    assert invalid_views.json()["error"]["code"] == "invalid_request"
    assert accepted.status_code == 202
    assert job["status"] == "completed"
    assert seen[0].input_paths[0].name == "turntable.mp4"
    assert seen[0].parameters == {
        "width_mm": 90.0,
        "views": 30,
        "start_frame": 5,
        "end_frame": 105,
        "clockwise": True,
    }


def test_jobs_run_one_at_a_time_and_failures_reach_terminal_state(tmp_path):
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def runner(job):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)
            if job.parameters["width_mm"] == 2:
                raise ValueError("No consistent foreground was found; retake the capture.")
            output = job.output_dir / "model.step"
            output.write_bytes(b"STEP")
            return output
        finally:
            with lock:
                active -= 1

    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=runner,
    )
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        first = client.post(
            "/api/jobs/photos", data={"width_mm": "1"}, files=_photo_files()
        ).json()
        second = client.post(
            "/api/jobs/photos", data={"width_mm": "2"}, files=_photo_files()
        ).json()
        first_result = _wait_for_terminal(client, first["id"])
        second_result = _wait_for_terminal(client, second["id"])

    assert maximum_active == 1
    assert first_result["status"] == "completed"
    assert second_result["status"] == "failed"
    assert second_result["error"] == {
        "code": "reconstruction_failed",
        "message": "No consistent foreground was found; retake the capture.",
    }


def test_invalid_job_ids_and_unknown_artifacts_never_resolve_filesystem_paths(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
    )
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        invalid = client.get("/api/jobs/not-a-uuid")
        missing = client.get("/api/jobs/00000000-0000-0000-0000-000000000000")

    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_request"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "job_not_found"


def test_uploads_reject_cross_origin_requests_and_allow_same_loopback_origin(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        hostile = client.post(
            "/api/jobs/photos",
            headers={"origin": "https://attacker.invalid"},
            data={"width_mm": "80"},
            files=_photo_files(),
        )
        browser_cross_site = client.post(
            "/api/jobs/photos",
            headers={"sec-fetch-site": "cross-site"},
            data={"width_mm": "80"},
            files=_photo_files(),
        )
        accepted = client.post(
            "/api/jobs/photos",
            headers={"origin": "http://127.0.0.1:8000"},
            data={"width_mm": "80"},
            files=_photo_files(),
        )

        assert hostile.status_code == 403
        assert hostile.json()["error"]["code"] == "untrusted_origin"
        assert browser_cross_site.status_code == 403
        assert browser_cross_site.json()["error"]["code"] == "untrusted_origin"
        assert accepted.status_code == 202
        assert _wait_for_terminal(client, accepted.json()["id"])["status"] == "completed"


def test_request_limit_rejects_declared_and_chunked_bodies_before_multipart_parsing(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
        max_video_bytes=32,
        request_overhead_bytes=8,
    )

    multipart = (
        b'--cadpro\r\nContent-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
        b"Content-Type: video/mp4\r\n\r\n"
        + MP4
        + b"x" * 64
        + b"\r\n--cadpro--\r\n"
    )

    def chunks():
        for offset in range(0, len(multipart), 20):
            yield multipart[offset : offset + 20]

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        declared = client.post(
            "/api/jobs/video",
            content=b"x" * 41,
            headers={"content-type": "multipart/form-data; boundary=cadpro"},
        )
        chunked = client.post(
            "/api/jobs/video",
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=cadpro"},
        )

    assert declared.status_code == 413
    assert declared.json()["error"]["code"] == "request_too_large"
    assert chunked.status_code == 413
    assert chunked.json()["error"]["code"] == "request_too_large"


def test_queue_admission_is_bounded_while_one_job_runs(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def runner(job):
        started.set()
        assert release.wait(timeout=5)
        output = job.output_dir / "model.step"
        output.write_bytes(b"STEP")
        return output

    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=runner,
        max_pending_jobs=2,
    )
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        try:
            first = client.post(
                "/api/jobs/photos", data={"width_mm": "1"}, files=_photo_files()
            )
            assert first.status_code == 202
            assert started.wait(timeout=2)
            second = client.post(
                "/api/jobs/photos", data={"width_mm": "2"}, files=_photo_files()
            )
            rejected = client.post(
                "/api/jobs/photos", data={"width_mm": "3"}, files=_photo_files()
            )

            assert second.status_code == 202
            assert rejected.status_code == 503
            assert rejected.headers["retry-after"] == "5"
            assert rejected.json()["error"]["code"] == "job_queue_full"
            assert len(list(app.state.job_manager.root.iterdir())) == 2
        finally:
            release.set()

        assert _wait_for_terminal(client, first.json()["id"])["status"] == "completed"
        assert _wait_for_terminal(client, second.json()["id"])["status"] == "completed"


def test_concurrent_progress_updates_cannot_regress_stage_or_percentage(tmp_path):
    updated = threading.Event()
    release = threading.Event()
    worker_state = []

    def runner(job):
        updates = [
            ("upload", 20, 0.004),
            ("segment", 30, 0.003),
            ("reconstruct", 60, 0.002),
            ("export", 85, 0.0),
        ]
        barrier = threading.Barrier(len(updates))

        def advance(stage, progress, delay):
            barrier.wait(timeout=2)
            time.sleep(delay)
            job.advance(stage, progress)

        threads = [
            threading.Thread(target=advance, args=item) for item in updates
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()
        with job._state_lock:
            worker_state.append((job.stage, job.progress))
        updated.set()
        assert release.wait(timeout=5)
        output = job.output_dir / "model.step"
        output.write_bytes(b"STEP")
        return output

    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=runner,
    )
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        try:
            accepted = client.post(
                "/api/jobs/photos", data={"width_mm": "80"}, files=_photo_files()
            )
            assert accepted.status_code == 202
            assert updated.wait(timeout=2)
            in_progress = client.get(accepted.json()["status_url"]).json()
            assert in_progress["status"] == "running"
            assert (in_progress["stage"], in_progress["progress"]) == ("export", 85)
        finally:
            release.set()

        completed = _wait_for_terminal(client, accepted.json()["id"])
        assert (completed["stage"], completed["progress"]) == ("complete", 100)

    assert worker_state == [("export", 85)]


def test_expired_job_is_swept_while_service_is_idle(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
        job_retention_seconds=0.03,
        job_sweep_interval_seconds=0.01,
    )
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/photos", data={"width_mm": "80"}, files=_photo_files()
        )
        completed = _wait_for_terminal(client, accepted.json()["id"])
        job_root = app.state.job_manager.root / completed["id"]
        assert job_root.is_dir()

        deadline = time.monotonic() + 2
        while job_root.exists() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not job_root.exists()
        missing = client.get(completed["status_url"])
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "job_not_found"

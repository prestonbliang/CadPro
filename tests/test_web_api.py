from __future__ import annotations

import asyncio
import base64
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import zlib

from fastapi.testclient import TestClient

from cadpro.web import JobManager, RequestGuardMiddleware, create_app


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


def _oversized_png() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", 50_000, 50_000, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _assets(tmp_path: Path) -> Path:
    assets = tmp_path / "web-assets"
    assets.mkdir(parents=True)
    (assets / "index.html").write_text(
        '<meta property="og:url" content="__CADPRO_ORIGIN__/"><h1>CadPro</h1>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("window.cadpro = true;", encoding="utf-8")
    return assets


def _image_file(
    *, body: bytes = PNG, name: str = "object.png", content_type: str = "image/png"
):
    return {"file": (name, body, content_type)}


def _photo_files(
    count: int = 20,
    *,
    body: bytes = PNG,
    name: str = "photo.png",
    content_type: str = "image/png",
):
    return [
        ("files", (f"{index:02d}-{name}", body, content_type))
        for index in range(count)
    ]


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


def _post_image(client: TestClient, *, width: str = "80", depth: str = "10", **kwargs):
    return client.post(
        "/api/jobs/image",
        data={"width_mm": width, "depth_mm": depth},
        files=_image_file(**kwargs),
    )


def test_index_health_and_routes_advertise_all_capture_modes(tmp_path, monkeypatch):
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
        health = client.get("/api/health")

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "untrusted_host"
    assert index.status_code == 200
    assert "https://cad.example/" in index.text
    assert "__CADPRO_ORIGIN__" not in index.text
    assert "attacker.invalid" not in index.text
    assert static.status_code == 200
    assert static.text == "window.cadpro = true;"
    assert health.json()["capture_limits"]["images"] == {"minimum": 1, "maximum": 1}
    assert health.json()["capture_limits"]["single_image"] == {"minimum": 1, "maximum": 1}
    assert health.json()["capture_limits"]["photos"] == {"minimum": 20, "maximum": 50}
    assert health.json()["capture_limits"]["video_views"] == {"minimum": 20, "maximum": 50}
    assert health.json()["intelligence"]["geometry_mutation"] is False
    assert health.json()["capture_limits"]["image_file"] == {
        "maximum_bytes": 25 * 1024 * 1024,
        "maximum_pixels": 12_500_000,
        "maximum_edge_pixels": 8_192,
    }
    post_routes = {
        route.path
        for route in app.routes
        if "POST" in (getattr(route, "methods", None) or set())
    }
    assert post_routes == {
        "/api/jobs/image",
        "/api/jobs/photos",
        "/api/jobs/video",
        "/api/jobs/text",
    }
    openapi = app.openapi()
    for path, field, minimum, maximum in (
        ("/api/jobs/image", "file", 1, 1),
        ("/api/jobs/photos", "files", 20, 50),
    ):
        body_schema = openapi["paths"][path]["post"]["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]
        body_name = body_schema["$ref"].rsplit("/", 1)[-1]
        file_schema = openapi["components"]["schemas"][body_name]["properties"][field]
        assert file_schema["minItems"] == minimum
        assert file_schema["maxItems"] == maximum


def test_image_job_uses_safe_name_dimensions_and_downloads_registered_artifact(tmp_path):
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
        response = _post_image(
            client,
            width="125.5",
            depth="18.25",
            name="..\\..\\part.png",
        )
        assert response.status_code == 202
        assert response.headers["location"] == response.json()["status_url"]
        job = _wait_for_terminal(client, response.json()["id"])
        assert job["status"] == "completed"
        assert job["stage"] == "complete"
        assert job["progress"] == 100
        assert job["kind"] == "image"
        assert job["input_count"] == 1
        assert job["parameters"] == {"width_mm": 125.5, "depth_mm": 18.25}
        assert len(job["result"]["artifacts"]) == 1
        assert job["result"]["metrics"]["dimensions_mm"] == [10.0, 20.0, 30.0]
        assert job["result"]["input_diagnostics"][0]["source_name"] == "image.png"

        artifact = job["result"]["artifacts"][0]
        download = client.get(artifact["download_url"])
        missing = client.get(f"/api/jobs/{job['id']}/artifacts/existing-secret-step")

        assert download.status_code == 200
        assert download.content.startswith(b"ISO-10303-21")
        assert "attachment" in download.headers["content-disposition"]
        assert missing.status_code == 404
        assert outside.read_bytes() == b"do not expose"
        assert [path.name for path in seen[0].input_paths] == ["image.png"]
        session_root = app.state.job_manager.root
        assert session_root.is_dir()

    assert storage.is_dir()
    assert not session_root.exists()
    assert outside.exists()


def test_photo_job_preserves_order_and_safe_names(tmp_path):
    seen = []
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(seen),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = client.post(
            "/api/jobs/photos",
            data={"width_mm": "125.5", "clockwise": "true"},
            files=_photo_files(name="..\\..\\part.png"),
        )
        assert accepted.status_code == 202
        job = _wait_for_terminal(client, accepted.json()["id"])

    assert job["status"] == "completed"
    assert job["kind"] == "photos"
    assert job["input_count"] == 20
    assert job["parameters"] == {"width_mm": 125.5, "clockwise": True}
    assert [path.name for path in seen[0].input_paths] == [
        f"view-{index:03d}.png" for index in range(1, 21)
    ]


def test_photo_endpoint_enforces_runtime_cardinality_and_media_bounds(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        too_few = client.post(
            "/api/jobs/photos", data={"width_mm": "80"}, files=_photo_files(19)
        )
        too_many = client.post(
            "/api/jobs/photos", data={"width_mm": "80"}, files=_photo_files(51)
        )
        wrong_type = client.post(
            "/api/jobs/photos",
            data={"width_mm": "80"},
            files=_photo_files(content_type="text/plain"),
        )
        bad_magic = client.post(
            "/api/jobs/photos",
            data={"width_mm": "80"},
            files=_photo_files(body=b"not a png"),
        )
        oversized = client.post(
            "/api/jobs/photos",
            data={"width_mm": "80"},
            files=_photo_files(body=_oversized_png()),
        )

        for response, received in ((too_few, 19), (too_many, 51)):
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "invalid_photo_count"
            assert response.json()["error"]["details"]["received"] == received
        assert wrong_type.status_code == 415
        assert wrong_type.json()["error"]["code"] == "content_type_mismatch"
        assert bad_magic.status_code == 415
        assert bad_magic.json()["error"]["code"] == "file_signature_mismatch"
        assert oversized.status_code == 422
        assert oversized.json()["error"]["code"] == "invalid_image"
        assert "files[0]" in oversized.json()["error"]["message"]
        assert list(app.state.job_manager.root.iterdir()) == []


def test_photo_endpoint_enforces_per_file_and_complete_set_byte_limits(tmp_path):
    per_file_app = create_app(
        storage_parent=tmp_path / "per-file",
        asset_dir=_assets(tmp_path / "per-file-assets"),
        reconstruction_runner=_artifact_runner(),
        max_image_bytes=len(PNG) - 1,
    )
    set_app = create_app(
        storage_parent=tmp_path / "set",
        asset_dir=_assets(tmp_path / "set-assets"),
        reconstruction_runner=_artifact_runner(),
        max_photo_set_bytes=len(PNG) * 20 - 1,
    )

    with TestClient(per_file_app, base_url="http://127.0.0.1:8000") as client:
        per_file = client.post(
            "/api/jobs/photos", data={"width_mm": "80"}, files=_photo_files()
        )
        assert per_file.status_code == 413
        assert per_file.json()["error"]["code"] == "file_too_large"
        assert list(per_file_app.state.job_manager.root.iterdir()) == []
    with TestClient(set_app, base_url="http://127.0.0.1:8000") as client:
        complete_set = client.post(
            "/api/jobs/photos", data={"width_mm": "80"}, files=_photo_files()
        )
        assert complete_set.status_code == 413
        assert complete_set.json()["error"]["code"] == "photo_set_too_large"
        assert list(set_app.state.job_manager.root.iterdir()) == []


def test_video_job_validates_media_fields_and_passes_safe_parameters(tmp_path):
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
        wrong_extension = client.post(
            "/api/jobs/video",
            data={"width_mm": "90"},
            files={"file": ("clip.png", PNG, "image/png")},
        )
        wrong_content_type = client.post(
            "/api/jobs/video",
            data={"width_mm": "90"},
            files={"file": ("clip.mp4", MP4, "text/plain")},
        )
        bad_magic = client.post(
            "/api/jobs/video",
            data={"width_mm": "90"},
            files={"file": ("clip.mp4", b"not mp4", "video/mp4")},
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
        assert accepted.status_code == 202
        job = _wait_for_terminal(client, accepted.json()["id"])

    assert invalid_range.status_code == 422
    assert invalid_range.json()["error"]["code"] == "invalid_frame_range"
    assert invalid_views.status_code == 422
    assert invalid_views.json()["error"]["code"] == "invalid_request"
    assert wrong_extension.status_code == 415
    assert wrong_extension.json()["error"]["code"] == "unsupported_video_type"
    assert wrong_content_type.status_code == 415
    assert wrong_content_type.json()["error"]["code"] == "content_type_mismatch"
    assert bad_magic.status_code == 415
    assert bad_magic.json()["error"]["code"] == "file_signature_mismatch"
    assert job["status"] == "completed"
    assert job["kind"] == "video"
    assert seen[0].input_paths[0].name == "turntable.mp4"
    assert seen[0].parameters == {
        "width_mm": 90.0,
        "views": 30,
        "start_frame": 5,
        "end_frame": 105,
        "clockwise": True,
    }


def test_image_fields_are_required_positive_finite_and_bounded(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        missing_depth = client.post(
            "/api/jobs/image", data={"width_mm": "80"}, files=_image_file()
        )
        bad_width = _post_image(client, width="0")
        bad_depth = _post_image(client, depth="nan")
        huge_depth = _post_image(client, depth="1000001")

    for response in (missing_depth, bad_width, bad_depth, huge_depth):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"


def test_image_endpoint_rejects_more_than_one_file(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
    )

    uploads = [
        ("file", ("front.png", PNG, "image/png")),
        ("file", ("back.png", PNG, "image/png")),
    ]
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/api/jobs/image",
            data={"width_mm": "80", "depth_mm": "10"},
            files=uploads,
        )

        assert response.status_code == 422
        assert response.json() == {
            "error": {
                "code": "invalid_image_count",
                "message": "Upload exactly one image.",
                "details": {"received": 2, "minimum": 1, "maximum": 1},
            }
        }
        assert list(app.state.job_manager.root.iterdir()) == []


def test_image_content_type_signature_and_size_errors_clean_staging(tmp_path):
    storage = tmp_path / "storage"
    app = create_app(
        storage_parent=storage,
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
        max_image_bytes=len(PNG) - 1,
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        wrong_extension = _post_image(client, name="object.mp4", content_type="video/mp4")
        wrong_type = _post_image(client, content_type="text/plain")
        bad_magic = _post_image(client, body=b"not a png")
        corrupt_image = _post_image(client, body=b"\x89PNG\r\n\x1a\ncorrupt")
        oversized_image = _post_image(client, body=_oversized_png())
        too_large = _post_image(client)

        assert wrong_extension.status_code == 415
        assert wrong_extension.json()["error"]["code"] == "unsupported_image_type"
        assert wrong_type.status_code == 415
        assert wrong_type.json()["error"]["code"] == "content_type_mismatch"
        assert bad_magic.status_code == 415
        assert bad_magic.json()["error"]["code"] == "file_signature_mismatch"
        assert corrupt_image.status_code == 422
        assert corrupt_image.json()["error"]["code"] == "invalid_image"
        assert str(app.state.job_manager.root) not in corrupt_image.text
        assert "storage" not in corrupt_image.json()["error"]["message"].lower()
        assert oversized_image.status_code == 422
        assert oversized_image.json()["error"]["code"] == "invalid_image"
        assert "12,500,000 pixels" in oversized_image.json()["error"]["message"]
        assert too_large.status_code == 413
        assert too_large.json()["error"]["code"] == "file_too_large"
        assert list(app.state.job_manager.root.iterdir()) == []


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
                raise ValueError("No consistent foreground was found; retake the image.")
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
        first = _post_image(client, width="1").json()
        second = _post_image(client, width="2").json()
        first_result = _wait_for_terminal(client, first["id"])
        second_result = _wait_for_terminal(client, second["id"])

    assert maximum_active == 1
    assert first_result["status"] == "completed"
    assert second_result["status"] == "failed"
    assert second_result["error"] == {
        "code": "reconstruction_failed",
        "message": "No consistent foreground was found; retake the image.",
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
            "/api/jobs/image",
            headers={"origin": "https://attacker.invalid"},
            data={"width_mm": "80", "depth_mm": "10"},
            files=_image_file(),
        )
        hostile_photos = client.post(
            "/api/jobs/photos",
            headers={"origin": "https://attacker.invalid"},
            data={"width_mm": "80"},
            files=_photo_files(),
        )
        hostile_video = client.post(
            "/api/jobs/video",
            headers={"origin": "https://attacker.invalid"},
            data={"width_mm": "80"},
            files={"file": ("clip.mp4", MP4, "video/mp4")},
        )
        browser_cross_site = client.post(
            "/api/jobs/image",
            headers={"sec-fetch-site": "cross-site"},
            data={"width_mm": "80", "depth_mm": "10"},
            files=_image_file(),
        )
        accepted = client.post(
            "/api/jobs/image",
            headers={"origin": "http://127.0.0.1:8000"},
            data={"width_mm": "80", "depth_mm": "10"},
            files=_image_file(),
        )

        assert hostile.status_code == 403
        assert hostile.json()["error"]["code"] == "untrusted_origin"
        assert hostile_photos.status_code == 403
        assert hostile_photos.json()["error"]["code"] == "untrusted_origin"
        assert hostile_video.status_code == 403
        assert hostile_video.json()["error"]["code"] == "untrusted_origin"
        assert browser_cross_site.status_code == 403
        assert browser_cross_site.json()["error"]["code"] == "untrusted_origin"
        assert accepted.status_code == 202
        assert _wait_for_terminal(client, accepted.json()["id"])["status"] == "completed"


def test_request_limit_rejects_declared_and_chunked_bodies_before_multipart_parsing(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
        max_image_bytes=32,
        request_overhead_bytes=8,
    )

    multipart = (
        b'--cadpro\r\nContent-Disposition: form-data; name="file"; filename="object.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
        + PNG
        + b"x" * 64
        + b"\r\n--cadpro--\r\n"
    )

    def chunks():
        for offset in range(0, len(multipart), 20):
            yield multipart[offset : offset + 20]

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        declared = client.post(
            "/api/jobs/image",
            content=b"x" * 41,
            headers={"content-type": "multipart/form-data; boundary=cadpro"},
        )
        chunked = client.post(
            "/api/jobs/image",
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=cadpro"},
        )

    assert declared.status_code == 413
    assert declared.json()["error"]["code"] == "request_too_large"
    assert chunked.status_code == 413
    assert chunked.json()["error"]["code"] == "request_too_large"


def test_upload_capacity_is_reserved_before_a_slow_multipart_body_is_spooled(tmp_path):
    manager = JobManager(
        storage_parent=tmp_path / "storage",
        runner=lambda job: None,
        retention_seconds=60,
        sweep_interval_seconds=60,
        max_pending_jobs=1,
    )
    entered_parser = threading.Event()
    release_parser = threading.Event()
    downstream_calls = []

    async def downstream(scope, receive, send):
        downstream_calls.append(scope)
        entered_parser.set()
        assert release_parser.wait(timeout=5)
        await receive()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guard = RequestGuardMiddleware(
        downstream,
        public_origin="http://127.0.0.1:8000",
        trusted_hosts=("127.0.0.1",),
        image_request_bytes=1_024,
        photo_request_bytes=1_024,
        video_request_bytes=1_024,
    )

    def scope():
        return {
            "type": "http",
            "method": "POST",
            "path": "/api/jobs/image",
            "scheme": "http",
            "headers": [(b"host", b"127.0.0.1:8000")],
            "app": SimpleNamespace(state=SimpleNamespace(job_manager=manager)),
        }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    first_messages = []
    second_messages = []

    async def send_first(message):
        first_messages.append(message)

    async def send_second(message):
        second_messages.append(message)

    worker = threading.Thread(
        target=lambda: asyncio.run(guard(scope(), receive, send_first))
    )
    worker.start()
    try:
        assert entered_parser.wait(timeout=2)
        asyncio.run(guard(scope(), receive, send_second))
        assert next(
            message["status"]
            for message in second_messages
            if message["type"] == "http.response.start"
        ) == 503
        assert len(downstream_calls) == 1
    finally:
        release_parser.set()
        worker.join(timeout=5)
        manager.close()

    assert not worker.is_alive()
    assert next(
        message["status"]
        for message in first_messages
        if message["type"] == "http.response.start"
    ) == 204


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
            first = _post_image(client, width="1")
            assert first.status_code == 202
            assert started.wait(timeout=2)
            second = _post_image(client, width="2")
            rejected = _post_image(client, width="3")

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

        threads = [threading.Thread(target=advance, args=item) for item in updates]
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
            accepted = _post_image(client)
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


def test_optional_intelligence_is_explicit_configured_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("CADPRO_AI_ENRICHMENT", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-exposed")
    monkeypatch.setenv("CADPRO_AI_MODEL", "gpt-test")
    monkeypatch.setenv("CADPRO_ML_MESH_ENABLED", "1")
    monkeypatch.setenv("CADPRO_ML_MESH_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("CADPRO_ML_MESH_ENDPOINT", "http://127.0.0.1:8099/generate")
    seen = []
    app = create_app(
        storage_parent=tmp_path / "jobs",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(seen),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        health = client.get("/api/health")
        accepted = client.post(
            "/api/jobs/image",
            data={
                "width_mm": "80",
                "depth_mm": "10",
                "ai_enhance": "true",
                "object_hint": "  catalog\n model   42  ",
                "concept_mesh": "true",
            },
            files=_image_file(),
        )
        assert accepted.status_code == 202, accepted.text
        completed = _wait_for_terminal(client, accepted.json()["id"])

    intelligence = health.json()["intelligence"]
    assert intelligence == {
        "available": True,
        "provider": "openai",
        "model": "gpt-test",
        "vision": True,
        "web_search": True,
        "geometry_mutation": False,
    }
    assert "sk-test-never-exposed" not in health.text
    assert health.json()["concept_mesh"] == {
        "available": True,
        "provider": "hunyuan-compatible-worker",
        "format": "glb",
        "metric_scale": False,
        "manufacturing_cad": False,
        "replaces_step": False,
        "multi_view_maximum": 1,
        "textured": False,
        "pbr": False,
        "rigging": False,
    }
    assert "127.0.0.1:8099" not in health.text
    assert completed["parameters"] == {
        "width_mm": 80.0,
        "depth_mm": 10.0,
        "ai_enhance": True,
        "object_hint": "catalog model 42",
        "concept_mesh": True,
    }
    assert seen[0].parameters == completed["parameters"]


def test_requesting_unconfigured_intelligence_does_not_stage_a_job(tmp_path, monkeypatch):
    monkeypatch.delenv("CADPRO_AI_ENRICHMENT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CADPRO_ML_MESH_ENABLED", raising=False)
    monkeypatch.delenv("CADPRO_ML_MESH_LICENSE_ACCEPTED", raising=False)
    monkeypatch.delenv("CADPRO_ML_MESH_ENDPOINT", raising=False)
    app = create_app(
        storage_parent=tmp_path / "jobs",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
    )

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        health = client.get("/api/health")
        rejected = client.post(
            "/api/jobs/image",
            data={"width_mm": "80", "depth_mm": "10", "ai_enhance": "true"},
            files=_image_file(),
        )
        concept_rejected = client.post(
            "/api/jobs/image",
            data={"width_mm": "80", "depth_mm": "10", "concept_mesh": "true"},
            files=_image_file(),
        )

    assert health.json()["intelligence"]["available"] is False
    assert rejected.status_code == 409
    assert rejected.json()["error"]["code"] == "intelligence_unavailable"
    assert "Local CAD reconstruction remains available" in rejected.text
    assert concept_rejected.status_code == 409
    assert concept_rejected.json()["error"]["code"] == "concept_mesh_unavailable"
    assert "Validated STEP reconstruction remains available" in concept_rejected.text


def test_expired_job_is_swept_while_service_is_idle(tmp_path):
    app = create_app(
        storage_parent=tmp_path / "storage",
        asset_dir=_assets(tmp_path),
        reconstruction_runner=_artifact_runner(),
        job_retention_seconds=0.03,
        job_sweep_interval_seconds=0.01,
    )
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        accepted = _post_image(client)
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

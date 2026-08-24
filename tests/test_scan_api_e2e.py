from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4
import zipfile

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, ImageDraw
import pytest

from cadpro.scan.models import (
    InputMode,
    JobStage,
    JobStatus,
    ReconstructionReport,
    ScanConfiguration,
    ToolCapability,
    ToolchainCapabilities,
)
from cadpro.scan.photogrammetry import ColmapOpenMvsAdapter, SyntheticTestAdapter
from cadpro.web import create_app


BASE_URL = "http://127.0.0.1:8000"
_TERMINAL = {
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
}


@pytest.fixture(autouse=True)
def local_test_origin(monkeypatch):
    monkeypatch.setenv("CADPRO_PUBLIC_ORIGIN", BASE_URL)
    monkeypatch.delenv("CADPRO_TRUSTED_HOSTS", raising=False)


def _mock_capabilities(*, colmap_available: bool = True) -> ToolchainCapabilities:
    names = {
        "ffmpeg": "FFmpeg",
        "ffprobe": "FFprobe",
        "colmap": "COLMAP",
        "interface_colmap": "OpenMVS InterfaceCOLMAP",
        "densify_point_cloud": "OpenMVS DensifyPointCloud",
        "reconstruct_mesh": "OpenMVS ReconstructMesh",
        "refine_mesh": "OpenMVS RefineMesh",
        "texture_mesh": "OpenMVS TextureMesh",
        "blender": "Blender",
        "trimesh": "trimesh",
        "ocp": "OpenCascade Python bindings",
    }
    tools: dict[str, ToolCapability] = {}
    for key, name in names.items():
        available = colmap_available if key == "colmap" else True
        tools[key] = ToolCapability(
            name=name,
            available=available,
            executable=f"mock-{key}" if available else None,
            version="mock-1.0" if available else None,
            reason=None if available else "COLMAP is intentionally absent in this API test.",
            install_hint=None if available else "Install COLMAP and restart CadPro.",
        )
    return ToolchainCapabilities(
        tools=tools,
        photo_reconstruction=colmap_available,
        video_ingest=True,
        dense_reconstruction=colmap_available,
        texture_generation=True,
        mesh_processing=True,
        analytic_cad=True,
    )


def _app(
    root: Path,
    *,
    adapter,
    capabilities: ToolchainCapabilities,
):
    return create_app(
        storage_parent=root / "legacy-jobs",
        scan_storage_parent=root / "persistent-scan",
        scan_adapter=adapter,
        scan_capabilities=capabilities,
        trusted_hosts=("127.0.0.1", "testserver"),
    )


def _sharp_png(seed: int) -> bytes:
    random = np.random.default_rng(seed)
    pixels = random.integers(18, 238, size=(224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    offset = 12 + seed * 9
    draw.rectangle((offset, 24, offset + 92, 116), outline=(255, 255, 255), width=5)
    draw.ellipse((110 - seed * 4, 80, 202 - seed * 3, 172), outline=(0, 0, 0), width=6)
    draw.line((8, 205 - seed * 7, 214, 14 + seed * 11), fill=(255, 230, 20), width=4)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _photo_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("files", (f"view-{index}.png", _sharp_png(index), "image/png"))
        for index in range(1, 4)
    ]


def _submit_photos(client: TestClient):
    return client.post(
        "/api/v2/jobs/photos",
        data={
            "quality_preset": "draft",
            "feature_matcher": "exhaustive",
            "mesher": "poisson",
            "use_gpu": "false",
            "generate_cad": "true",
        },
        files=_photo_files(),
    )


def _wait_for_terminal(
    client: TestClient,
    status_url: str,
    *,
    expected: JobStatus,
    timeout_seconds: float = 30.0,
) -> tuple[dict[str, object], list[int]]:
    deadline = time.monotonic() + timeout_seconds
    progress: list[int] = []
    while time.monotonic() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200, response.text
        payload = response.json()
        progress.append(payload["progress"])
        if payload["status"] == expected.value:
            return payload, progress
        if payload["status"] in _TERMINAL:
            raise AssertionError(
                f"job reached {payload['status']!r}, expected {expected.value!r}: "
                f"{payload.get('errors')}"
            )
        time.sleep(0.02)
    raise AssertionError(f"job did not reach {expected.value!r} before timeout")


def _job_count(client: TestClient) -> int:
    database = client.app.state.scan_store.database_path
    with sqlite3.connect(database) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])


def test_capabilities_and_single_image_are_truthful_without_creating_a_job(tmp_path):
    capabilities = _mock_capabilities()
    adapter = SyntheticTestAdapter(allow_test_only=True)
    app = _app(tmp_path, adapter=adapter, capabilities=capabilities)

    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/api/v2/capabilities")

        assert response.status_code == 200
        payload = response.json()
        assert payload["version"] == "2"
        assert payload["capture_limits"]["photos"] == {
            "minimum": 3,
            "recommended_minimum": 20,
            "maximum": 100,
        }
        assert payload["capabilities"]["photo_reconstruction"] is True
        assert payload["capabilities"]["tools"]["colmap"]["version"] == "mock-1.0"
        assert payload["capabilities"]["tools"]["colmap"]["executable"] is None
        assert payload["standard_pipeline_uses_paid_cloud"] is False
        assert payload["single_image"]["available"] is False
        assert payload["single_image"]["experimental"] is True
        assert payload["single_image"]["metric"] is False

        before = _job_count(client)
        rejected = client.post(
            "/api/v2/jobs/single-image",
            files={"file": ("one-view.png", _sharp_png(9), "image/png")},
        )

        assert rejected.status_code == 409
        error = rejected.json()["error"]
        assert error["code"] == "single_image_provider_unavailable"
        message = error["message"].lower()
        assert "experimental" in message
        assert "non-metric" in message
        assert "unavailable" in message
        assert _job_count(client) == before
        assert list(client.app.state.scan_store.jobs_directory.iterdir()) == []


def test_three_photo_mock_e2e_artifacts_calibration_and_restart_recovery(tmp_path):
    capabilities = _mock_capabilities()
    scan_root = tmp_path / "app-state"
    app = _app(
        scan_root,
        adapter=SyntheticTestAdapter(allow_test_only=True),
        capabilities=capabilities,
    )

    with TestClient(app, base_url=BASE_URL) as client:
        accepted = _submit_photos(client)
        assert accepted.status_code == 202, accepted.text
        initial = accepted.json()
        assert accepted.headers["location"] == initial["status_url"]
        assert accepted.headers["retry-after"] == "1"
        assert initial["mode"] == InputMode.PHOTOS.value
        assert initial["configuration"]["quality_preset"] == "draft"
        assert [item["source_name"] for item in initial["input_metadata"]] == [
            "view-1.png",
            "view-2.png",
            "view-3.png",
        ]

        completed, observed_progress = _wait_for_terminal(
            client,
            initial["status_url"],
            expected=JobStatus.COMPLETED,
        )
        assert observed_progress == sorted(observed_progress)
        assert completed["progress"] == 100
        assert completed["report"]["quality_class"] == "weak"
        assert completed["report"]["scale"]["calibrated"] is False
        assert completed["report"]["inferred_surfaces"] is False
        assert any(item["code"] == "scale_unknown" for item in completed["warnings"])
        artifacts = {item["artifact_id"]: item for item in completed["artifacts"]}
        assert {
            "sparse-ply",
            "dense-ply",
            "mesh-ply",
            "visual-glb",
            "mesh-obj",
            "printable-stl",
            "preview",
            "report",
            "manifest",
            "complete-bundle",
        } <= set(artifacts)
        assert "fitted-step" not in artifacts
        for artifact in artifacts.values():
            downloaded = client.get(artifact["download_url"])
            assert downloaded.status_code == 200, artifact
            assert len(downloaded.content) == artifact["size_bytes"]

        glb = client.get(artifacts["visual-glb"]["download_url"])
        assert glb.content.startswith(b"glTF")
        assert artifacts["visual-glb"]["metric_scale"] is False
        report_response = client.get(artifacts["report"]["download_url"])
        report = ReconstructionReport.model_validate_json(report_response.content)
        assert report.job_id == initial["id"]
        assert report.mode == InputMode.PHOTOS
        assert report.metrics.uploaded_images == 3
        assert report.metrics.accepted_images == 3
        assert all(tool.executable is None for tool in report.capabilities.tools.values())
        bundle_response = client.get(artifacts["complete-bundle"]["download_url"])
        with zipfile.ZipFile(BytesIO(bundle_response.content), "r") as archive:
            names = archive.namelist()
            assert archive.testzip() is None
            assert len(names) == len(set(names))
            assert all(Path(name).name == name and ".." not in Path(name).parts for name in names)
            bundled = {
                item["filename"]
                for key, item in artifacts.items()
                if key != "complete-bundle"
            }
            assert set(names) == bundled
            assert "reconstruction-report.json" in names
            assert "reproducibility-manifest.json" in names

        traversal = client.get(
            f"/api/v2/jobs/{initial['id']}/artifacts/%2E%2E"
        )
        assert traversal.status_code == 404

        calibration = client.post(
            f"/api/v2/jobs/{initial['id']}/calibration",
            json={
                "point_a": [-1.0, 0.0, 0.0],
                "point_b": [1.0, 0.0, 0.0],
                "real_distance": 200.0,
                "unit": "mm",
                "selection_uncertainty": 0.01,
            },
        )
        assert calibration.status_code == 202, calibration.text
        revision = calibration.json()
        assert revision["id"] != initial["id"]
        assert revision["configuration"]["scale"]["unit"] == "mm"
        assert revision["input_metadata"][-1] == {
            "calibration_revision_of": initial["id"]
        }

        calibrated, _ = _wait_for_terminal(
            client,
            revision["status_url"],
            expected=JobStatus.COMPLETED,
        )
        scale = calibrated["report"]["scale"]
        assert scale["calibrated"] is True
        assert scale["output_unit"] == "mm"
        assert scale["scale_factor"] == pytest.approx(100.0)
        assert scale["estimated_uncertainty"] == pytest.approx(1.0)
        assert calibrated["report"]["metrics"]["bounding_box"] == pytest.approx(
            [200.0, 100.0, 50.0]
        )
        assert calibrated["tool_versions"]["pipeline_adapter"] == "immutable-artifact-reuse"
        assert (
            calibrated["tool_versions"]["reconstruction_reused_from_job"]
            == initial["id"]
        )
        assert any(
            "were not rerun" in warning["message"]
            for warning in calibrated["warnings"]
        )
        revised_artifacts = {
            item["artifact_id"]: item for item in calibrated["artifacts"]
        }
        assert revised_artifacts["visual-glb"]["metric_scale"] is True
        assert calibrated["report"]["cad_status"] == "generated"
        assert "fitted-step" in revised_artifacts
        step = client.get(revised_artifacts["fitted-step"]["download_url"])
        assert step.status_code == 200
        assert step.content.startswith(b"ISO-10303-21")

        cross_job = client.get(
            f"/api/v2/jobs/{initial['id']}/artifacts/fitted-step"
        )
        assert cross_job.status_code == 404
        assert cross_job.json()["error"]["code"] == "artifact_not_found"
        unknown_job = client.get(
            f"/api/v2/jobs/{uuid4()}/artifacts/visual-glb"
        )
        assert unknown_job.status_code == 404

        store = client.app.state.scan_store
        interrupted = store.allocate_workspace()
        interrupted_input = interrupted.input_directory / "input-0001.png"
        interrupted_input.write_bytes(_sharp_png(21))
        store.create(
            interrupted,
            mode=InputMode.PHOTOS,
            input_paths=[interrupted_input],
            input_metadata=[{"source_name": "interrupted.png"}],
            configuration=ScanConfiguration(mode=InputMode.PHOTOS, use_gpu=False),
        )
        store.transition(interrupted.job_id, JobStage.BUILDING_MESH, 70)

    restarted = _app(
        scan_root,
        adapter=SyntheticTestAdapter(allow_test_only=True),
        capabilities=capabilities,
    )
    with TestClient(restarted, base_url=BASE_URL) as client:
        persisted = client.get(initial["status_url"])
        assert persisted.status_code == 200
        assert persisted.json()["status"] == JobStatus.COMPLETED.value
        assert client.get(artifacts["complete-bundle"]["download_url"]).status_code == 200
        persisted_revision = client.get(revision["status_url"])
        assert persisted_revision.status_code == 200
        assert persisted_revision.json()["report"]["scale"]["calibrated"] is True
        recovered = client.get(f"/api/v2/jobs/{interrupted.job_id}")
        assert recovered.status_code == 200
        recovered_payload = recovered.json()
        assert recovered_payload["status"] == JobStatus.FAILED.value
        assert recovered_payload["errors"][-1]["code"] == "worker_interrupted"


def test_running_scan_can_be_cancelled_through_api(tmp_path):
    entered_adapter = threading.Event()
    release_adapter = threading.Event()

    def delay_probe() -> None:
        entered_adapter.set()
        release_adapter.wait(10)

    app = _app(
        tmp_path,
        adapter=SyntheticTestAdapter(allow_test_only=True, delay_probe=delay_probe),
        capabilities=_mock_capabilities(),
    )
    with TestClient(app, base_url=BASE_URL) as client:
        accepted = _submit_photos(client)
        assert accepted.status_code == 202, accepted.text
        job = accepted.json()
        try:
            assert entered_adapter.wait(10)
            cancelled = client.post(job["cancel_url"])
            assert cancelled.status_code == 200
            assert cancelled.json()["cancel_requested"] is True
            release_adapter.set()
            terminal, _ = _wait_for_terminal(
                client,
                job["status_url"],
                expected=JobStatus.CANCELLED,
            )
            assert terminal["stage"] == JobStage.CANCELLED.value
            assert terminal["artifacts"] == []
            assert terminal["errors"][-1]["code"] == "cancelled_by_user"
        finally:
            release_adapter.set()


def test_missing_native_dependency_returns_409_without_creating_a_job(tmp_path):
    capabilities = _mock_capabilities(colmap_available=False)
    app = _app(
        tmp_path,
        adapter=ColmapOpenMvsAdapter(capabilities),
        capabilities=capabilities,
    )

    with TestClient(app, base_url=BASE_URL) as client:
        assert _job_count(client) == 0

        rejected = _submit_photos(client)

        assert rejected.status_code == 409
        error = rejected.json()["error"]
        assert error["code"] == "dependency_unavailable"
        assert "cannot start" in error["message"]
        assert error["details"]["missing"] == [
            {
                "tool": "COLMAP",
                "reason": "COLMAP is intentionally absent in this API test.",
                "install_hint": "Install COLMAP and restart CadPro.",
            }
        ]
        assert _job_count(client) == 0
        assert list(client.app.state.scan_store.jobs_directory.iterdir()) == []

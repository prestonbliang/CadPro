"""Opt-in smoke test for a real local photogrammetry installation.

Set ``CADPRO_REAL_SCAN_DATASET`` to a directory containing 3-100 overlapping JPG, PNG,
or WebP photographs of one object. Images must be direct children of the directory and must
meet CadPro's 25 MiB per-file and 500 MiB set limits. The test skips unless the variable,
dataset, COLMAP/dense toolchain, and Python mesh-processing capability are all available.

This test invokes native reconstruction and can take up to 30 minutes. It never uses the
synthetic adapter, cloud AI, repository binary fixtures, or an implicit fallback.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING
import zipfile

import pytest

from cadpro.scan.api import MAX_IMAGE_BYTES, MAX_PHOTO_SET_BYTES, SCAN_MAX_PHOTOS
from cadpro.scan.artifacts import validate_output, validate_zip
from cadpro.scan.capabilities import default_toolchain
from cadpro.scan.models import ArtifactKind, JobStatus, ReconstructionReport
from cadpro.web import create_app

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration

_BASE_URL = "http://127.0.0.1:8000"
_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_REQUIRED_ARTIFACTS = {
    "sparse-ply",
    "dense-ply",
    "mesh-ply",
    "visual-glb",
    "mesh-obj",
    "preview",
    "report",
    "manifest",
    "complete-bundle",
}
_TIMEOUT_SECONDS = 30 * 60


def _dataset_or_skip() -> tuple[Path, ...]:
    configured = os.environ.get("CADPRO_REAL_SCAN_DATASET", "").strip()
    if not configured:
        pytest.skip(
            "set CADPRO_REAL_SCAN_DATASET to a local directory of overlapping object photos"
        )
    directory = Path(configured).expanduser()
    if not directory.is_dir():
        pytest.skip(f"CADPRO_REAL_SCAN_DATASET is not a directory: {directory}")
    images = tuple(
        sorted(
            (
                path.resolve()
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in _MIME_TYPES
            ),
            key=lambda path: path.name.casefold(),
        )
    )
    if len(images) < 3:
        pytest.skip(
            "CADPRO_REAL_SCAN_DATASET must contain at least 3 supported overlapping images"
        )
    images = images[:SCAN_MAX_PHOTOS]
    oversized = [path.name for path in images if path.stat().st_size > MAX_IMAGE_BYTES]
    if oversized:
        pytest.skip(f"dataset images exceed CadPro's per-file limit: {oversized}")
    total_bytes = sum(path.stat().st_size for path in images)
    if total_bytes > MAX_PHOTO_SET_BYTES:
        pytest.skip("CADPRO_REAL_SCAN_DATASET exceeds CadPro's photo-set byte limit")
    return images


def _require_native_capabilities() -> None:
    capabilities = default_toolchain()
    missing: list[str] = []
    if not capabilities.photo_reconstruction:
        missing.append("COLMAP photo reconstruction")
    if not capabilities.dense_reconstruction:
        missing.append("dense reconstruction")
    if not capabilities.mesh_processing:
        missing.append("Python mesh processing")
    if missing:
        pytest.skip("native scan capabilities are unavailable: " + ", ".join(missing))


def _await_terminal(client: TestClient, snapshot: dict[str, object]) -> dict[str, object]:
    status_url = str(snapshot["status_url"])
    cancel_url = str(snapshot["cancel_url"])
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    observed_progress: list[int] = []
    while time.monotonic() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200, response.text
        current = response.json()
        observed_progress.append(int(current["progress"]))
        if current["status"] in {
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }:
            assert observed_progress == sorted(observed_progress)
            return current
        time.sleep(1)
    client.post(cancel_url)
    pytest.fail(f"real reconstruction exceeded the {_TIMEOUT_SECONDS}-second test timeout")


def _download_artifact(
    client: TestClient,
    artifact: dict[str, object],
    output_directory: Path,
) -> Path:
    filename = str(artifact["filename"])
    assert Path(filename).name == filename
    assert ".." not in Path(filename).parts
    destination = output_directory / filename
    digest = hashlib.sha256()
    size = 0
    with client.stream("GET", str(artifact["download_url"])) as response:
        assert response.status_code == 200, response.text
        with destination.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    assert size == artifact["size_bytes"]
    assert digest.hexdigest() == artifact["sha256"]
    assert artifact["validated"] is True
    validate_output(
        destination,
        kind=ArtifactKind(str(artifact["kind"])),
        textured=artifact.get("textured"),
    )
    return destination


def test_real_photo_dataset_through_versioned_scan_api(tmp_path, monkeypatch):
    images = _dataset_or_skip()
    _require_native_capabilities()
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CADPRO_PUBLIC_ORIGIN", _BASE_URL)
    monkeypatch.delenv("CADPRO_TRUSTED_HOSTS", raising=False)
    app = create_app(
        storage_parent=tmp_path / "legacy-storage",
        scan_storage_parent=tmp_path / "scan-storage",
        max_image_bytes=MAX_IMAGE_BYTES,
        max_photo_set_bytes=MAX_PHOTO_SET_BYTES,
        trusted_hosts=("127.0.0.1", "testserver"),
    )

    with TestClient(app, base_url=_BASE_URL) as client, ExitStack() as opened:
        uploads = [
            (
                "files",
                (
                    path.name,
                    opened.enter_context(path.open("rb")),
                    _MIME_TYPES[path.suffix.lower()],
                ),
            )
            for path in images
        ]
        accepted = client.post(
            "/api/v2/jobs/photos",
            data={
                "quality_preset": "draft",
                "feature_matcher": "exhaustive",
                "mesher": "poisson",
                "use_gpu": "true",
                "generate_cad": "false",
            },
            files=uploads,
        )
        assert accepted.status_code == 202, accepted.text
        submitted = accepted.json()
        assert accepted.headers["location"] == submitted["status_url"]

        terminal = _await_terminal(client, submitted)
        assert terminal["status"] == JobStatus.COMPLETED.value, json.dumps(
            terminal.get("errors"), indent=2
        )
        assert terminal["progress"] == 100
        assert terminal["report"]["mode"] == "photos"
        assert terminal["report"]["metrics"]["uploaded_images"] == len(images)
        assert terminal["report"]["scale"]["calibrated"] is False
        assert terminal["report"]["cad_status"] == "skipped"
        assert terminal["tool_versions"]["pipeline_adapter"] == "colmap-openmvs"

        artifacts = {item["artifact_id"]: item for item in terminal["artifacts"]}
        assert _REQUIRED_ARTIFACTS <= set(artifacts)
        assert "fitted-step" not in artifacts
        download_directory = tmp_path / "downloads"
        download_directory.mkdir()
        downloaded = {
            artifact_id: _download_artifact(client, artifact, download_directory)
            for artifact_id, artifact in artifacts.items()
        }

        report = ReconstructionReport.model_validate_json(
            downloaded["report"].read_text(encoding="utf-8")
        )
        assert report.job_id == submitted["id"]
        assert report.metrics.accepted_images >= 3
        assert report.metrics.registered_cameras >= 3
        assert report.tool_versions["pipeline_adapter"] == "colmap-openmvs"
        manifest = json.loads(downloaded["manifest"].read_text(encoding="utf-8"))
        assert manifest["commands"]
        assert any("colmap" in Path(command[0]).name.lower() for command in manifest["commands"])
        bundle_names = validate_zip(downloaded["complete-bundle"])
        expected_bundle_names = {
            str(item["filename"])
            for artifact_id, item in artifacts.items()
            if artifact_id != "complete-bundle"
        }
        assert set(bundle_names) == expected_bundle_names
        with zipfile.ZipFile(downloaded["complete-bundle"], "r") as archive:
            assert archive.testzip() is None
            assert "reconstruction-report.json" in archive.namelist()
            assert "reproducibility-manifest.json" in archive.namelist()

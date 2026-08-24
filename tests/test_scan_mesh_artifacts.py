from __future__ import annotations

import hashlib
from pathlib import Path
import warnings
import zipfile

import numpy as np
from PIL import Image
import pytest
import trimesh

from cadpro.scan.artifacts import (
    sha256_file,
    validate_output,
    validate_zip,
    write_manifest,
    write_report,
)
from cadpro.scan.mesh import (
    export_mesh_products,
    mesh_statistics,
    repair_mesh,
    validate_mesh_arrays,
)
from cadpro.scan.models import (
    ArtifactKind,
    InputMode,
    JobStatus,
    ReconstructionMetrics,
    ReconstructionReport,
    ReproducibilityManifest,
    ScaleInformation,
    ToolchainCapabilities,
)


def _write_mesh_ply(path: Path, mesh: trimesh.Trimesh) -> Path:
    path.write_bytes(bytes(mesh.export(file_type="ply")))
    return path


def _open_grid() -> trimesh.Trimesh:
    x, y = np.meshgrid(np.arange(3, dtype=np.float64), np.arange(3, dtype=np.float64))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(9)))
    faces: list[tuple[int, int, int]] = []
    for row in range(2):
        for column in range(2):
            corner = row * 3 + column
            faces.extend(
                (
                    (corner, corner + 1, corner + 4),
                    (corner, corner + 4, corner + 3),
                )
            )
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def _write_glb(path: Path, mesh: trimesh.Trimesh) -> Path:
    path.write_bytes(bytes(mesh.export(file_type="glb")))
    return path


def _minimal_report() -> ReconstructionReport:
    capabilities = ToolchainCapabilities(
        tools={},
        photo_reconstruction=False,
        video_ingest=False,
        dense_reconstruction=False,
        texture_generation=False,
        mesh_processing=True,
        analytic_cad=True,
    )
    metrics = ReconstructionMetrics(
        uploaded_images=3,
        accepted_images=3,
        registered_cameras=3,
        registration_percentage=100.0,
        sparse_points=12,
        dense_points=42,
        reprojection_error_px=0.4,
        bounding_box=(1.0, 2.0, 3.0),
        vertices=8,
        triangles=12,
        connected_components=1,
        boundary_edges=0,
        non_manifold_edges=0,
        watertight=True,
    )
    return ReconstructionReport(
        job_id="tiny-fixture-job",
        mode=InputMode.PHOTOS,
        status=JobStatus.COMPLETED,
        quality_class="usable",
        configuration={"quality_preset": "balanced"},
        capabilities=capabilities,
        image_quality=[],
        metrics=metrics,
        scale=ScaleInformation(),
        cad_explanation="CAD was not requested for this mesh fixture.",
        tool_versions={"fixture": "1.0"},
    )


@pytest.mark.parametrize("bad_coordinate", [np.nan, np.inf, -np.inf])
def test_mesh_array_validation_rejects_nonfinite_vertices(bad_coordinate):
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    vertices[0, 0] = bad_coordinate
    mesh = trimesh.Trimesh(vertices=vertices, faces=((0, 1, 2),), process=False)

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_mesh_arrays(mesh)


@pytest.mark.parametrize("face", [(-1, 1, 2), (0, 1, 3)])
def test_mesh_array_validation_rejects_out_of_bounds_indices(face):
    mesh = trimesh.Trimesh(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        faces=(face,),
        process=False,
    )

    with pytest.raises(ValueError, match="outside the vertex array"):
        validate_mesh_arrays(mesh)


def test_mesh_statistics_report_topology_and_bounds():
    box = trimesh.creation.box(extents=(2.0, 3.0, 4.0))

    statistics = mesh_statistics(box)

    assert statistics.vertices == 8
    assert statistics.triangles == 12
    assert statistics.connected_components == 1
    assert statistics.boundary_edges == 0
    assert statistics.non_manifold_edges == 0
    assert statistics.watertight is True
    assert statistics.bounding_box == pytest.approx((2.0, 3.0, 4.0))


def test_repair_removes_only_a_tiny_disconnected_fragment():
    body = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    fragment = trimesh.Trimesh(
        vertices=((10.0, 0.0, 0.0), (10.0, 1.0, 0.0), (10.0, 0.0, 1.0)),
        faces=((0, 1, 2),),
        process=False,
    )
    combined = trimesh.util.concatenate((body, fragment))

    repaired, repair_warnings = repair_mesh(combined)

    assert len(combined.faces) == 81
    assert len(repaired.faces) == 80
    assert mesh_statistics(repaired).connected_components == 1
    assert any("Removed 1 disconnected fragment" in warning for warning in repair_warnings)


def test_mesh_products_reopen_as_glb_obj_and_ply(tmp_path):
    source = _write_mesh_ply(tmp_path / "box.ply", trimesh.creation.box())

    products = export_mesh_products(
        source,
        tmp_path / "products",
        scale=ScaleInformation(),
        stem="tiny-box",
    )

    glb = trimesh.load(products.glb_path, force="scene", process=False)
    assert isinstance(glb, trimesh.Scene)
    assert glb.geometry
    for path, kind in (
        (products.obj_path, ArtifactKind.TRIANGLE_MESH),
        (products.cleaned_mesh_path, ArtifactKind.TRIANGLE_MESH),
    ):
        reopened = trimesh.load(path, force="mesh", process=False)
        assert isinstance(reopened, trimesh.Trimesh)
        assert len(reopened.faces) == 12
        validate_output(path, kind=kind, textured=False)
    validate_output(
        products.glb_path,
        kind=ArtifactKind.TEXTURED_MODEL,
        textured=False,
    )


def test_watertight_mesh_emits_reopenable_printable_stl(tmp_path):
    source = _write_mesh_ply(tmp_path / "box.ply", trimesh.creation.box())

    products = export_mesh_products(
        source,
        tmp_path / "products",
        scale=ScaleInformation(),
        stem="printable-box",
    )

    assert products.statistics.watertight is True
    assert products.statistics.boundary_edges == 0
    assert products.statistics.non_manifold_edges == 0
    assert products.stl_path is not None
    assert products.stl_path.is_file()
    validate_output(products.stl_path, kind=ArtifactKind.PRINTABLE_MESH)
    reopened = trimesh.load(products.stl_path, force="mesh", process=True)
    assert isinstance(reopened, trimesh.Trimesh)
    assert reopened.is_watertight


def test_nonwatertight_mesh_withholds_printable_stl(tmp_path):
    source = _write_mesh_ply(tmp_path / "open-grid.ply", _open_grid())

    products = export_mesh_products(
        source,
        tmp_path / "products",
        scale=ScaleInformation(),
        stem="open-grid",
    )

    assert products.statistics.watertight is False
    assert products.statistics.boundary_edges == 8
    assert products.stl_path is None
    assert not list((tmp_path / "products").glob("*.stl"))
    assert any("printable STL was withheld" in warning for warning in products.warnings)


def test_glb_texture_label_requires_linked_uv_material_and_embedded_image(tmp_path):
    untextured = _write_glb(tmp_path / "untextured.glb", trimesh.creation.box())
    validate_output(untextured, kind=ArtifactKind.TEXTURED_MODEL, textured=False)
    with pytest.raises(RuntimeError, match="labeled textured"):
        validate_output(untextured, kind=ArtifactKind.TEXTURED_MODEL, textured=True)

    textured_mesh = trimesh.creation.box()
    vertices = np.asarray(textured_mesh.vertices)
    uv = np.column_stack(
        (
            (vertices[:, 0] - vertices[:, 0].min()) / np.ptp(vertices[:, 0]),
            (vertices[:, 1] - vertices[:, 1].min()) / np.ptp(vertices[:, 1]),
        )
    )
    material = trimesh.visual.material.SimpleMaterial(
        image=Image.new("RGB", (2, 2), (220, 40, 20))
    )
    textured_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    textured = _write_glb(tmp_path / "textured.glb", textured_mesh)

    validate_output(textured, kind=ArtifactKind.TEXTURED_MODEL, textured=True)


def test_zip_validation_accepts_safe_basenames(tmp_path):
    archive_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("mesh.glb", b"fixture mesh")
        archive.writestr("report.json", b"{}")

    assert validate_zip(archive_path) == ("mesh.glb", "report.json")


def test_zip_validation_rejects_traversal_and_duplicate_names(tmp_path):
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.txt", b"must stay inside")
    with pytest.raises(RuntimeError, match="unsafe path"):
        validate_zip(traversal)

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("same.txt", b"first")
            archive.writestr("same.txt", b"second")
    with pytest.raises(RuntimeError, match="unique non-empty"):
        validate_zip(duplicate)


def test_report_and_manifest_round_trip_schemas_and_checksums(tmp_path):
    report_artifact = write_report(_minimal_report(), tmp_path)

    parsed_report = ReconstructionReport.model_validate_json(
        report_artifact.path.read_text(encoding="utf-8")
    )
    assert parsed_report.schema_version == "3.0"
    assert report_artifact.metadata.kind == ArtifactKind.REPORT
    assert report_artifact.metadata.size_bytes == report_artifact.path.stat().st_size
    assert report_artifact.metadata.sha256 == sha256_file(report_artifact.path)

    input_digest = hashlib.sha256(b"tiny generated input").hexdigest()
    manifest_artifact = write_manifest(
        job_id=parsed_report.job_id,
        input_sha256={"capture-0001.jpg": input_digest},
        configuration=parsed_report.configuration,
        commands=[["fixture-reconstructor", "--bounded"]],
        tool_versions=parsed_report.tool_versions,
        artifacts=[report_artifact.metadata],
        output_directory=tmp_path,
    )

    parsed_manifest = ReproducibilityManifest.model_validate_json(
        manifest_artifact.path.read_text(encoding="utf-8")
    )
    assert parsed_manifest.schema_version == "1.0"
    assert parsed_manifest.input_sha256 == {"capture-0001.jpg": input_digest}
    assert parsed_manifest.artifacts == [report_artifact.metadata]
    assert parsed_manifest.created_at.tzinfo is not None
    assert manifest_artifact.metadata.kind == ArtifactKind.MANIFEST
    assert manifest_artifact.metadata.size_bytes == manifest_artifact.path.stat().st_size
    assert manifest_artifact.metadata.sha256 == sha256_file(manifest_artifact.path)

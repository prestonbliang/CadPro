import json
import struct

import numpy as np
import pygltflib
import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from cad_diff.step_io import load_step
from cadpro import artifacts as artifacts_module
from cadpro.media import Silhouette
from cadpro.reconstruct import Reconstruction


def _reconstruction() -> Reconstruction:
    silhouette = Silhouette(
        outer=np.asarray([[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]),
        holes=(),
        source_size=(100, 100),
    )
    return Reconstruction(
        shape=BRepPrimAPI_MakeBox(gp_Pnt(100, 200, 300), 10, 20, 30).Shape(),
        silhouettes=(silhouette,),
        mode="image",
        source_names=("object.png",),
    )


def _glb_positions(payload: bytes) -> np.ndarray:
    document = pygltflib.GLTF2().load_from_bytes(payload)
    blob = document.binary_blob()
    positions = []
    for mesh in document.meshes:
        for primitive in mesh.primitives:
            accessor = document.accessors[primitive.attributes.POSITION]
            view = document.bufferViews[accessor.bufferView]
            assert accessor.componentType == pygltflib.FLOAT
            assert accessor.type == pygltflib.VEC3
            assert view.byteStride in (None, 12)
            offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
            values = np.frombuffer(
                blob,
                dtype="<f4",
                count=accessor.count * 3,
                offset=offset,
            ).reshape((-1, 3))
            positions.append(values)
    return np.concatenate(positions)


def test_exports_valid_step_binary_stl_glb_preview_and_json(tmp_path):
    manifest = artifacts_module.export_artifacts(_reconstruction(), tmp_path, stem="sample")

    assert manifest.step_path == tmp_path / "sample.step"
    assert len(load_step(manifest.step_path)) == 1
    stl = manifest.stl_path.read_bytes()
    triangle_count = struct.unpack_from("<I", stl, 80)[0]
    assert triangle_count > 0
    assert len(stl) == 84 + triangle_count * 50
    glb = manifest.glb_path.read_bytes()
    magic, version, declared_length = struct.unpack_from("<4sII", glb)
    assert magic == b"glTF"
    assert version == 2
    assert declared_length == len(glb)
    assert pygltflib.GLTF2().load_from_bytes(glb).meshes
    positions = _glb_positions(glb)
    assert positions.min(axis=0) == pytest.approx((0.100, 0.300, -0.220))
    assert positions.max(axis=0) == pytest.approx((0.110, 0.330, -0.200))
    assert np.ptp(positions, axis=0) == pytest.approx((0.010, 0.030, 0.020))
    preview = manifest.preview_path.read_text(encoding="utf-8")
    assert "CadPro preview - sample" in preview
    assert "data:text/javascript;base64," in preview

    report = json.loads(manifest.report_path.read_text(encoding="utf-8"))
    assert report["reconstruction"] == {"mode": "image", "input_count": 1}
    assert report["geometry"]["dimensions_mm"] == {"x": 10.0, "y": 20.0, "z": 30.0}
    assert report["geometry"]["volume_mm3"] == pytest.approx(6000)
    assert report["geometry"]["solid_count"] == 1
    assert report["geometry"]["face_count"] == 6
    assert report["geometry"]["is_valid"] is True
    assert report["inputs"][0]["source_name"] == "object.png"
    assert set(report["artifacts"]) == {"step", "stl", "glb", "preview", "report"}
    assert manifest.metrics.dimensions_mm == pytest.approx((10, 20, 30))
    assert manifest.preview_html == manifest.preview_path
    assert manifest.report_json == manifest.report_path


def test_failed_staged_export_preserves_existing_artifacts(tmp_path, monkeypatch):
    expected = {}
    for suffix in ("step", "stl", "glb", "preview.html", "report.json"):
        path = tmp_path / f"existing.{suffix}"
        payload = f"original {suffix}".encode()
        path.write_bytes(payload)
        expected[path] = payload

    monkeypatch.setattr(
        artifacts_module,
        "build_assembly_diff_glb",
        lambda solids, **kwargs: (_ for _ in ()).throw(RuntimeError("forced GLB failure")),
    )

    with pytest.raises(RuntimeError, match="forced GLB failure"):
        artifacts_module.export_artifacts(_reconstruction(), tmp_path, stem="existing")

    assert {path: path.read_bytes() for path in expected} == expected
    assert not any(path.is_dir() and path.name.startswith(".existing-") for path in tmp_path.iterdir())


def test_image_artifact_export_requires_exactly_one_input(tmp_path):
    reconstruction = _reconstruction()
    invalid = Reconstruction(
        shape=reconstruction.shape,
        silhouettes=reconstruction.silhouettes * 2,
        mode="image",
        source_names=("front.png", "back.png"),
    )

    with pytest.raises(ValueError, match="exactly one silhouette"):
        artifacts_module.export_artifacts(invalid, tmp_path)


@pytest.mark.parametrize("stem", ["", "../escape", "has spaces", "<script>"])
def test_rejects_unsafe_artifact_stems(tmp_path, stem):
    with pytest.raises(ValueError, match="stem"):
        artifacts_module.export_artifacts(_reconstruction(), tmp_path, stem=stem)

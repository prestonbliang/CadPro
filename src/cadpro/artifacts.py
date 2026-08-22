from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import struct
import tempfile

from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Shape

from cad_diff.gltf_export import VisualSolid, build_assembly_diff_glb
from cad_diff.html_report import render_html
from cad_diff.signatures import fingerprint_solid
from cad_diff.step_io import load_step
from cadpro.reconstruct import InputDiagnostic, Reconstruction
from cadpro.step import write_step


_SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _cad_mm_z_up_to_gltf_m_y_up(
    vertex: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate CAD Z-up coordinates to glTF Y-up and convert mm to metres."""
    x, y, z = vertex
    return (x * 0.001, z * 0.001, -y * 0.001)


@dataclass(frozen=True)
class GeometryMetrics:
    dimensions_mm: tuple[float, float, float]
    volume_mm3: float
    surface_area_mm2: float
    solid_count: int
    face_count: int
    is_valid: bool


@dataclass(frozen=True)
class ArtifactManifest:
    step_path: Path
    stl_path: Path
    glb_path: Path
    preview_path: Path
    report_path: Path
    metrics: GeometryMetrics
    input_diagnostics: tuple[InputDiagnostic, ...]

    @property
    def step(self) -> Path:
        return self.step_path

    @property
    def stl(self) -> Path:
        return self.stl_path

    @property
    def glb(self) -> Path:
        return self.glb_path

    @property
    def preview_html(self) -> Path:
        return self.preview_path

    @property
    def report_json(self) -> Path:
        return self.report_path


def export_artifacts(
    reconstruction: Reconstruction,
    output_dir: str | Path,
    stem: str = "cadpro-model",
) -> ArtifactManifest:
    """Publish a validated STEP plus interoperable mesh and preview artifacts."""
    if not _SAFE_STEM.fullmatch(stem):
        raise ValueError(
            "stem must be 1-128 characters using only letters, digits, '.', '_' or '-', "
            "and must start with a letter or digit"
        )
    if reconstruction.mode not in {"photos", "video"}:
        raise ValueError(f"Unknown reconstruction mode: {reconstruction.mode!r}")
    if len(reconstruction.silhouettes) != len(reconstruction.source_names):
        raise ValueError("Reconstruction silhouettes and source_names must have equal lengths")
    if reconstruction.shape.IsNull():
        raise ValueError("Reconstruction contains no CAD shape")
    if not BRepCheck_Analyzer(reconstruction.shape).IsValid():
        raise RuntimeError("Reconstructed CAD body failed OpenCascade validity checks")

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "step": destination_dir / f"{stem}.step",
        "stl": destination_dir / f"{stem}.stl",
        "glb": destination_dir / f"{stem}.glb",
        "preview": destination_dir / f"{stem}.preview.html",
        "report": destination_dir / f"{stem}.report.json",
    }

    diagnostics = reconstruction.input_diagnostics
    with tempfile.TemporaryDirectory(prefix=f".{stem}-", dir=destination_dir) as staging_name:
        staging_dir = Path(staging_name)
        staged = {name: staging_dir / path.name for name, path in destinations.items()}

        step_path = write_step(reconstruction.shape, staged["step"])
        loaded = load_step(step_path)
        if len(loaded) != 1:
            raise RuntimeError(
                f"Exported STEP validation found {len(loaded)} solids; expected exactly one"
            )
        loaded_shape = loaded[0][1]
        metrics = _geometry_metrics(loaded_shape)
        if metrics.solid_count != 1:
            raise RuntimeError(
                f"Exported STEP contains {metrics.solid_count} solids; expected exactly one"
            )
        if not metrics.is_valid or metrics.volume_mm3 <= 0:
            raise RuntimeError("Exported STEP does not contain one valid volumetric solid")

        _write_binary_stl(loaded_shape, staged["stl"])
        glb_bytes = build_assembly_diff_glb(
            [VisualSolid(status="unchanged", modified_shape=loaded_shape)],
            vertex_transform=_cad_mm_z_up_to_gltf_m_y_up,
        )
        _validate_glb_header(glb_bytes)
        _write_bytes(staged["glb"], glb_bytes)
        _write_text(
            staged["preview"],
            render_html(glb_bytes, title=f"CadPro preview - {stem}"),
        )

        report = _report_document(
            reconstruction=reconstruction,
            diagnostics=diagnostics,
            metrics=metrics,
            staged=staged,
            destinations=destinations,
        )
        _write_text(
            staged["report"],
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _validate_staged_files(staged)

        for name in ("step", "stl", "glb", "preview", "report"):
            try:
                os.replace(staged[name], destinations[name])
            except OSError as error:
                raise RuntimeError(f"Could not publish artifact: {destinations[name]}") from error

    return ArtifactManifest(
        step_path=destinations["step"],
        stl_path=destinations["stl"],
        glb_path=destinations["glb"],
        preview_path=destinations["preview"],
        report_path=destinations["report"],
        metrics=metrics,
        input_diagnostics=diagnostics,
    )


def _geometry_metrics(shape: TopoDS_Shape) -> GeometryMetrics:
    fingerprint = fingerprint_solid("cadpro-model", shape)
    dimensions = tuple(
        round(maximum - minimum, 6)
        for minimum, maximum in zip(fingerprint.bbox_min, fingerprint.bbox_max)
    )
    return GeometryMetrics(
        dimensions_mm=dimensions,
        volume_mm3=fingerprint.volume,
        surface_area_mm2=fingerprint.surface_area,
        solid_count=_subshape_count(shape, TopAbs_SOLID),
        face_count=_subshape_count(shape, TopAbs_FACE),
        is_valid=BRepCheck_Analyzer(shape).IsValid(),
    )


def _subshape_count(shape: TopoDS_Shape, kind) -> int:
    explorer = TopExp_Explorer(shape, kind)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _write_binary_stl(shape: TopoDS_Shape, path: Path) -> None:
    mesher = BRepMesh_IncrementalMesh(shape, 0.1)
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade could not tessellate the reconstructed body for STL")
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    if not writer.Write(shape, str(path)):
        raise RuntimeError(f"Could not write binary STL file: {path.name}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Could not validate binary STL file: {path.name}") from error
    if len(payload) < 84:
        raise RuntimeError("STL exporter produced an empty or truncated file")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    if triangle_count == 0 or len(payload) != 84 + triangle_count * 50:
        raise RuntimeError("STL exporter did not produce a valid binary STL payload")


def _validate_glb_header(payload: bytes) -> None:
    if len(payload) < 12:
        raise RuntimeError("GLB exporter produced an empty or truncated header")
    magic, version, declared_length = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise RuntimeError("GLB exporter produced an invalid binary header")


def _write_bytes(path: Path, payload: bytes) -> None:
    if not payload:
        raise RuntimeError(f"Refusing to write an empty artifact: {path.name}")
    try:
        path.write_bytes(payload)
    except OSError as error:
        raise RuntimeError(f"Could not write artifact: {path.name}") from error


def _write_text(path: Path, payload: str) -> None:
    if not payload:
        raise RuntimeError(f"Refusing to write an empty artifact: {path.name}")
    try:
        path.write_text(payload, encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Could not write artifact: {path.name}") from error


def _report_document(
    *,
    reconstruction: Reconstruction,
    diagnostics: tuple[InputDiagnostic, ...],
    metrics: GeometryMetrics,
    staged: dict[str, Path],
    destinations: dict[str, Path],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "reconstruction": {
            "mode": reconstruction.mode,
            "input_count": len(reconstruction.silhouettes),
        },
        "geometry": {
            "dimensions_mm": {
                "x": metrics.dimensions_mm[0],
                "y": metrics.dimensions_mm[1],
                "z": metrics.dimensions_mm[2],
            },
            "volume_mm3": metrics.volume_mm3,
            "surface_area_mm2": metrics.surface_area_mm2,
            "solid_count": metrics.solid_count,
            "face_count": metrics.face_count,
            "is_valid": metrics.is_valid,
        },
        "inputs": [
            {
                **asdict(diagnostic),
                "source_size": {
                    "width": diagnostic.source_size[0],
                    "height": diagnostic.source_size[1],
                },
            }
            for diagnostic in diagnostics
        ],
        "artifacts": {
            name: {
                "file": destinations[name].name,
                **({"bytes": staged[name].stat().st_size} if name != "report" else {}),
            }
            for name in ("step", "stl", "glb", "preview", "report")
        },
    }


def _validate_staged_files(staged: dict[str, Path]) -> None:
    for name, path in staged.items():
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Artifact exporter produced no {name} output")
        except OSError as error:
            raise RuntimeError(f"Could not validate artifact: {path.name}") from error

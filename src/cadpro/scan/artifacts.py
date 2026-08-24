"""Atomic artifact publication, schema validation, checksums, and safe bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from pygltflib import GLTF2

from cad_diff.html_report import render_html
from cadpro.scan.models import (
    ArtifactKind,
    ArtifactMetadata,
    ReconstructionReport,
    ReproducibilityManifest,
)


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    metadata: ArtifactMetadata


def publish_file(
    source: str | Path,
    output_directory: str | Path,
    *,
    filename: str,
    artifact_id: str,
    kind: ArtifactKind,
    textured: bool | None = None,
    metric_scale: bool | None = None,
    move: bool = False,
) -> PublishedArtifact:
    if not _SAFE_NAME.fullmatch(filename) or filename in {".", ".."}:
        raise ValueError("Artifact filename is unsafe.")
    source_path = Path(source).resolve(strict=True)
    if source_path.is_symlink() or not source_path.is_file() or source_path.stat().st_size <= 0:
        raise ValueError("Artifact source must be a non-empty regular file.")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    if destination.exists() and destination.resolve() != source_path:
        raise FileExistsError(f"Refusing to overwrite artifact {filename}.")
    if destination.resolve() != source_path:
        temporary = output / f".{filename}.{os.getpid()}.tmp"
        if move:
            shutil.move(str(source_path), temporary)
        else:
            shutil.copyfile(source_path, temporary)
        os.replace(temporary, destination)
    validate_output(destination, kind=kind, textured=textured)
    return _published(
        destination,
        artifact_id=artifact_id,
        kind=kind,
        textured=textured,
        metric_scale=metric_scale,
    )


def write_report(
    report: ReconstructionReport,
    output_directory: str | Path,
    *,
    filename: str = "reconstruction-report.json",
) -> PublishedArtifact:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    payload = report.model_dump_json(indent=2) + "\n"
    _atomic_text(destination, payload)
    ReconstructionReport.model_validate_json(destination.read_text(encoding="utf-8"))
    return _published(
        destination,
        artifact_id="report",
        kind=ArtifactKind.REPORT,
        textured=None,
        metric_scale=report.scale.calibrated,
    )


def write_manifest(
    *,
    job_id: str,
    input_sha256: dict[str, str],
    configuration: dict[str, object],
    commands: list[list[str]],
    tool_versions: dict[str, str],
    artifacts: list[ArtifactMetadata],
    output_directory: str | Path,
) -> PublishedArtifact:
    manifest = ReproducibilityManifest(
        job_id=job_id,
        input_sha256=input_sha256,
        configuration=configuration,
        commands=commands,
        tool_versions=tool_versions,
        artifacts=artifacts,
        created_at=datetime.now(timezone.utc),
    )
    destination = Path(output_directory) / "reproducibility-manifest.json"
    _atomic_text(destination, manifest.model_dump_json(indent=2) + "\n")
    ReproducibilityManifest.model_validate_json(destination.read_text(encoding="utf-8"))
    return _published(
        destination,
        artifact_id="manifest",
        kind=ArtifactKind.MANIFEST,
        textured=None,
        metric_scale=None,
    )


def write_preview(
    glb_path: str | Path,
    output_directory: str | Path,
    *,
    title: str = "CadPro photogrammetry preview",
) -> PublishedArtifact:
    glb = Path(glb_path).read_bytes()
    _validate_glb(Path(glb_path), require_texture=False)
    destination = Path(output_directory) / "cadpro-scan.preview.html"
    _atomic_text(destination, render_html(glb, title=title))
    return _published(
        destination,
        artifact_id="preview",
        kind=ArtifactKind.PREVIEW,
        textured=None,
        metric_scale=None,
    )


def write_bundle(
    artifacts: list[PublishedArtifact],
    output_directory: str | Path,
    *,
    filename: str = "cadpro-scan-complete.zip",
) -> PublishedArtifact:
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    names: set[str] = set()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".bundle-", suffix=".zip", dir=output)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for artifact in artifacts:
                path = artifact.path.resolve(strict=True)
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Bundle input is not a regular file.")
                name = path.name
                if not _safe_zip_name(name) or name in names:
                    raise ValueError("Bundle artifact names must be unique safe basenames.")
                names.add(name)
                archive.write(path, arcname=name)
        validate_zip(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _published(
        destination,
        artifact_id="complete-bundle",
        kind=ArtifactKind.BUNDLE,
        textured=None,
        metric_scale=None,
    )


def validate_output(
    path: str | Path,
    *,
    kind: ArtifactKind,
    textured: bool | None = None,
) -> None:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size <= 0:
        raise RuntimeError("Advertised artifact is missing or empty.")
    suffix = candidate.suffix.lower()
    if suffix == ".glb":
        _validate_glb(candidate, require_texture=textured is True)
    elif suffix == ".obj":
        _validate_obj(candidate)
    elif suffix == ".stl":
        import trimesh

        mesh = trimesh.load(candidate, force="mesh", process=True)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise RuntimeError("STL could not be reopened as triangle geometry.")
        if kind == ArtifactKind.PRINTABLE_MESH and not mesh.is_watertight:
            raise RuntimeError("Printable STL is not watertight after reopening.")
    elif suffix == ".ply":
        from cadpro.scan.mesh import validate_point_cloud

        validate_point_cloud(candidate)
    elif suffix in {".step", ".stp"}:
        from OCP.BRepCheck import BRepCheck_Analyzer
        from cad_diff.step_io import load_step

        solids = load_step(candidate)
        if len(solids) != 1 or not BRepCheck_Analyzer(solids[0][1]).IsValid():
            raise RuntimeError("STEP did not reopen as one valid analytic solid.")
    elif suffix == ".json":
        json.loads(candidate.read_text(encoding="utf-8"))
    elif suffix == ".zip":
        validate_zip(candidate)


def validate_zip(path: str | Path) -> tuple[str, ...]:
    names: list[str] = []
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity validation failed.")
        for info in archive.infolist():
            name = info.filename
            if not _safe_zip_name(name):
                raise RuntimeError("ZIP contains an unsafe path.")
            if info.is_dir() or info.file_size <= 0:
                raise RuntimeError("ZIP contains an empty or directory entry.")
            names.append(name)
    if not names or len(names) != len(set(names)):
        raise RuntimeError("ZIP must contain unique non-empty artifacts.")
    return tuple(names)


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _published(
    path: Path,
    *,
    artifact_id: str,
    kind: ArtifactKind,
    textured: bool | None,
    metric_scale: bool | None,
) -> PublishedArtifact:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if path.suffix.lower() in {".step", ".stp"}:
        media_type = "model/step"
    metadata = ArtifactMetadata(
        artifact_id=artifact_id,
        kind=kind,
        filename=path.name,
        media_type=media_type,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        textured=textured,
        metric_scale=metric_scale,
        validated=True,
    )
    return PublishedArtifact(path, metadata)


def _validate_glb(path: Path, *, require_texture: bool) -> None:
    payload = path.read_bytes()
    if len(payload) < 12 or payload[:4] != b"glTF":
        raise RuntimeError("GLB header is invalid or truncated.")
    declared = int.from_bytes(payload[8:12], "little")
    if int.from_bytes(payload[4:8], "little") != 2 or declared != len(payload):
        raise RuntimeError("GLB version or declared length is invalid.")
    document = GLTF2.load_binary(str(path))
    if not document.meshes or not document.nodes:
        raise RuntimeError("GLB contains no renderable mesh nodes.")
    if require_texture:
        textured_primitive = False
        for mesh in document.meshes:
            for primitive in mesh.primitives:
                attributes = primitive.attributes
                if (
                    primitive.material is not None
                    and getattr(attributes, "TEXCOORD_0", None) is not None
                ):
                    textured_primitive = True
        if not textured_primitive or not document.materials or not document.textures or not document.images:
            raise RuntimeError("GLB was labeled textured but has no linked UV/material/texture data.")
        if not any(image.bufferView is not None for image in document.images):
            raise RuntimeError("Textured GLB does not embed its image payload.")


def _validate_obj(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines()
    if sum(line.startswith("v ") for line in lines) < 3:
        raise RuntimeError("OBJ contains fewer than three vertices.")
    if not any(line.startswith("f ") for line in lines):
        raise RuntimeError("OBJ contains no faces.")
    for line in lines:
        if line.startswith("mtllib "):
            reference = line.split(maxsplit=1)[1].strip()
            if Path(reference).name != reference or reference in {".", ".."}:
                raise RuntimeError("OBJ contains an unsafe material path.")


def _safe_zip_name(name: str) -> bool:
    path = Path(name.replace("\\", "/"))
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) == 1
        and _SAFE_NAME.fullmatch(name) is not None
    )


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(payload, encoding="utf-8")
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Refusing to publish an empty text artifact.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

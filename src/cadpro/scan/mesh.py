"""Mesh/cloud validation, conservative repair, and gated interchange exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import struct
from typing import Iterable

import numpy as np
import trimesh
from trimesh import Trimesh

from cadpro.scan.models import ScaleInformation


@dataclass(frozen=True)
class MeshStatistics:
    vertices: int
    triangles: int
    connected_components: int
    boundary_edges: int
    non_manifold_edges: int
    watertight: bool
    bounding_box: tuple[float, float, float]


@dataclass(frozen=True)
class MeshExports:
    glb_path: Path
    obj_path: Path
    obj_resources: tuple[Path, ...]
    stl_path: Path | None
    cleaned_mesh_path: Path
    statistics: MeshStatistics
    textured: bool
    warnings: tuple[str, ...]


def load_valid_mesh(path: str | Path) -> Trimesh:
    source = Path(path)
    loaded = trimesh.load(source, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"{source.name} contains no mesh geometry.")
        mesh = loaded.to_mesh()
    elif isinstance(loaded, Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"{source.name} is not a triangle mesh.")
    validate_mesh_arrays(mesh)
    return mesh


def validate_mesh_arrays(mesh: Trimesh) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Mesh vertices must be a non-empty N x 3 array.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
        raise ValueError("Mesh faces must be a non-empty M x 3 triangle array.")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains NaN or infinite vertices.")
    if np.issubdtype(faces.dtype, np.floating) and not np.equal(faces, np.floor(faces)).all():
        raise ValueError("Mesh face indices must be integers.")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("Mesh face indices are outside the vertex array.")
    normals = np.asarray(mesh.face_normals)
    if normals.shape != (len(faces), 3) or not np.isfinite(normals).all():
        raise ValueError("Mesh normals are missing or non-finite.")


def mesh_statistics(mesh: Trimesh) -> MeshStatistics:
    validate_mesh_arrays(mesh)
    inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    counts = np.bincount(inverse, minlength=len(mesh.edges_unique))
    components = mesh.split(only_watertight=False)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    return MeshStatistics(
        vertices=len(mesh.vertices),
        triangles=len(mesh.faces),
        connected_components=max(1, len(components)),
        boundary_edges=int(np.sum(counts == 1)),
        non_manifold_edges=int(np.sum(counts > 2)),
        watertight=bool(mesh.is_watertight),
        bounding_box=(float(extents[0]), float(extents[1]), float(extents[2])),
    )


def repair_mesh(mesh: Trimesh) -> tuple[Trimesh, tuple[str, ...]]:
    """Apply only bounded, inspectable cleanup; do not force a fake watertight result."""

    cleaned = mesh.copy()
    warnings: list[str] = []
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.update_faces(cleaned.unique_faces())
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    components = list(cleaned.split(only_watertight=False))
    if len(components) > 1:
        components.sort(key=lambda item: len(item.faces), reverse=True)
        total_faces = sum(len(item.faces) for item in components)
        discarded_faces = total_faces - len(components[0].faces)
        if total_faces and discarded_faces / total_faces <= 0.05:
            cleaned = components[0].copy()
            warnings.append(
                f"Removed {len(components) - 1} disconnected fragment(s) containing "
                f"{discarded_faces} triangle(s)."
            )
        else:
            warnings.append(
                "Multiple substantial connected components remain; they were preserved for review."
            )
    trimesh.repair.fix_normals(cleaned, multibody=True)
    if not cleaned.is_watertight:
        filled = bool(trimesh.repair.fill_holes(cleaned))
        if filled:
            warnings.append("Filled only simple triangle/quad boundary holes during repair.")
    cleaned.remove_unreferenced_vertices()
    validate_mesh_arrays(cleaned)
    return cleaned, tuple(warnings)


def export_mesh_products(
    source_mesh: str | Path,
    output_directory: str | Path,
    *,
    scale: ScaleInformation,
    stem: str = "cadpro-scan",
) -> MeshExports:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    mesh = load_valid_mesh(source_mesh)
    if scale.calibrated:
        if scale.scale_factor is None:
            raise ValueError("A calibrated scale must include a scale factor.")
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale.scale_factor
    cleaned, warnings = repair_mesh(mesh)
    statistics = mesh_statistics(cleaned)
    cleaned_mesh_path = output / f"{stem}-cleaned.ply"
    _write_export(cleaned_mesh_path, cleaned.export(file_type="ply"))
    validate_point_cloud(cleaned_mesh_path)

    obj_path = output / f"{stem}.obj"
    obj_text, resource_payloads = trimesh.exchange.obj.export_obj(
        cleaned,
        include_normals=True,
        include_color=True,
        include_texture=True,
        return_texture=True,
        mtl_name=f"{stem}.mtl",
    )
    obj_path.write_text(obj_text, encoding="utf-8")
    resources: list[Path] = []
    for raw_name, payload in resource_payloads.items():
        name = Path(str(raw_name)).name
        if not name or name != str(raw_name).replace("\\", "/") or name in {".", ".."}:
            raise RuntimeError("OBJ exporter returned an unsafe resource name.")
        destination = output / name
        _write_export(destination, payload)
        resources.append(destination)
    _validate_obj(obj_path)

    glb_path = output / f"{stem}.glb"
    glb_payload = cleaned.export(file_type="glb")
    _write_export(glb_path, glb_payload)
    _validate_glb(glb_path)

    textured = _has_texture(cleaned) and bool(resources)
    stl_path: Path | None = None
    if statistics.watertight and statistics.boundary_edges == 0 and statistics.non_manifold_edges == 0:
        candidate = output / f"{stem}-watertight.stl"
        _write_export(candidate, cleaned.export(file_type="stl"))
        reopened = trimesh.load(candidate, force="mesh", process=True)
        if not isinstance(reopened, Trimesh) or not reopened.is_watertight:
            candidate.unlink(missing_ok=True)
            warnings = (*warnings, "STL reopen validation failed; printable STL was withheld.")
        else:
            validate_mesh_arrays(reopened)
            stl_path = candidate
    else:
        warnings = (
            *warnings,
            "Repair did not produce a watertight manifold mesh; printable STL was withheld.",
        )
    return MeshExports(
        glb_path=glb_path,
        obj_path=obj_path,
        obj_resources=tuple(resources),
        stl_path=stl_path,
        cleaned_mesh_path=cleaned_mesh_path,
        statistics=statistics,
        textured=textured,
        warnings=warnings,
    )


def export_point_cloud(
    source: str | Path,
    destination: str | Path,
    *,
    scale: ScaleInformation,
) -> Path:
    loaded = trimesh.load(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("Point-cloud source contains no geometry.")
        loaded = loaded.to_mesh()
    if isinstance(loaded, Trimesh):
        points = np.asarray(loaded.vertices, dtype=np.float64)
        colors = getattr(loaded.visual, "vertex_colors", None)
    elif isinstance(loaded, trimesh.points.PointCloud):
        points = np.asarray(loaded.vertices, dtype=np.float64)
        colors = loaded.colors
    else:
        raise ValueError("Unsupported point-cloud source.")
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Point cloud must contain finite N x 3 coordinates.")
    if scale.calibrated:
        if scale.scale_factor is None:
            raise ValueError("A calibrated scale must include a scale factor.")
        points = points * scale.scale_factor
    color_array = np.asarray(colors) if colors is not None else np.empty((0, 0))
    if (
        color_array.ndim != 2
        or color_array.shape[0] != len(points)
        or color_array.shape[1] not in {3, 4}
    ):
        colors = None
    cloud = trimesh.points.PointCloud(points, colors=colors)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_export(path, cloud.export(file_type="ply"))
    validate_point_cloud(path)
    return path


def validate_point_cloud(path: str | Path) -> int:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("PLY contains no geometry.")
        vertices = np.vstack([np.asarray(item.vertices) for item in loaded.geometry.values()])
    elif isinstance(loaded, (Trimesh, trimesh.points.PointCloud)):
        vertices = np.asarray(loaded.vertices)
    else:
        raise ValueError("PLY could not be reopened as points or mesh geometry.")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("PLY contains no 3D points.")
    if not np.isfinite(vertices).all():
        raise ValueError("PLY contains non-finite point coordinates.")
    return len(vertices)


def copy_textured_obj_bundle(
    obj_path: str | Path,
    resources: Iterable[str | Path],
    output_directory: str | Path,
) -> tuple[Path, tuple[Path, ...]]:
    """Copy a native texturer's already-related OBJ files without following paths."""

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(obj_path), *(Path(item) for item in resources)]
    copied: list[Path] = []
    names: set[str] = set()
    for source in sources:
        resolved = source.resolve(strict=True)
        name = resolved.name
        if name in names:
            raise ValueError("Textured OBJ resources must have unique basenames.")
        names.add(name)
        destination = output / name
        shutil.copyfile(resolved, destination)
        copied.append(destination)
    _validate_obj(copied[0])
    return copied[0], tuple(copied[1:])


def _has_texture(mesh: Trimesh) -> bool:
    if getattr(mesh.visual, "kind", None) != "texture":
        return False
    material = getattr(mesh.visual, "material", None)
    return bool(material is not None and getattr(material, "image", None) is not None)


def _write_export(path: Path, payload: bytes | str | dict[str, bytes | str]) -> None:
    if isinstance(payload, dict):
        raise TypeError("A single export path cannot receive a resource dictionary.")
    data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    if not data:
        raise RuntimeError(f"Exporter produced an empty {path.suffix} file.")
    path.write_bytes(data)


def _validate_glb(path: Path) -> None:
    payload = path.read_bytes()
    if len(payload) < 12:
        raise RuntimeError("GLB output is truncated.")
    magic, version, declared = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or declared != len(payload):
        raise RuntimeError("GLB output has an invalid header.")
    reopened = trimesh.load(path, force="scene", process=False)
    if not isinstance(reopened, trimesh.Scene) or not reopened.geometry:
        raise RuntimeError("GLB output could not be reopened as geometry.")


def _validate_obj(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="strict")
    vertices = sum(1 for line in text.splitlines() if line.startswith("v "))
    faces = sum(1 for line in text.splitlines() if line.startswith("f "))
    if vertices < 3 or faces < 1:
        raise RuntimeError("OBJ output contains no triangle geometry.")

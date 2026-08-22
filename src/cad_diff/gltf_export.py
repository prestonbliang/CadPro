from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable

import pygltflib as gltf
from OCP.TopoDS import TopoDS_Shape

from cad_diff.diff_model import FaceDiff
from cad_diff.tessellate import FaceMesh, tessellate_shape

# RGBA, matching the pitch's visual language: gray/yellow/green/red.
_STATUS_COLOR = {
    "unchanged": (0.62, 0.62, 0.62, 1.0),
    "modified": (0.90, 0.72, 0.10, 1.0),
    "added": (0.27, 0.68, 0.35, 1.0),
    "removed": (0.82, 0.24, 0.20, 0.5),  # translucent — this geometry is a ghost, it no longer exists
}
_STATUS_ORDER = ["unchanged", "modified", "added", "removed"]  # draw removed (translucent) last

Vertex = tuple[float, float, float]
VertexTransform = Callable[[Vertex], Vertex]


@dataclass(frozen=True)
class VisualSolid:
    """One assembly solid and the geometry needed to visualize its status."""

    status: str
    base_shape: TopoDS_Shape | None = None
    modified_shape: TopoDS_Shape | None = None
    face_diffs: tuple[FaceDiff, ...] = ()


def build_diff_glb(base_shape, modified_shape, face_diffs: list[FaceDiff]) -> bytes:
    """Tessellate both shapes and bucket their triangles into one GLB by diff
    status — a single self-contained scene showing the current (modified)
    geometry with additions/changes highlighted, plus translucent red ghosts
    of whatever the base had that's now gone."""
    return build_assembly_diff_glb(
        [
            VisualSolid(
                status="modified",
                base_shape=base_shape,
                modified_shape=modified_shape,
                face_diffs=tuple(face_diffs),
            )
        ]
    )


def build_assembly_diff_glb(
    solids: list[VisualSolid],
    *,
    vertex_transform: VertexTransform | None = None,
) -> bytes:
    """Build a complete assembly GLB without dropping whole-solid changes.

    ``vertex_transform`` is deliberately opt-in so cad-diff reports retain
    their established model coordinates. Interchange exporters can use it to
    convert source units and axes while the mesh is serialized.
    """
    if not solids:
        raise ValueError("cannot build a visual diff for an empty assembly")

    buckets: dict[str, list[FaceMesh]] = {status: [] for status in _STATUS_ORDER}
    for solid in solids:
        _append_visual_solid(buckets, solid)

    return _build_glb(
        {status: meshes for status, meshes in buckets.items() if meshes},
        vertex_transform=vertex_transform,
    )


def _append_visual_solid(buckets: dict[str, list[FaceMesh]], solid: VisualSolid) -> None:
    if solid.status not in _STATUS_ORDER:
        raise ValueError(f"unknown solid diff status: {solid.status}")

    if solid.status == "modified":
        if solid.base_shape is None or solid.modified_shape is None or not solid.face_diffs:
            raise ValueError("modified visual solids require both shapes and face diffs")
        base_meshes = tessellate_shape(solid.base_shape)
        modified_meshes = tessellate_shape(solid.modified_shape)
        for face_diff in solid.face_diffs:
            if face_diff.status == "removed":
                if face_diff.base is None:
                    raise ValueError("removed face diff is missing its base fingerprint")
                buckets["removed"].append(base_meshes[face_diff.base.index])
            else:
                if face_diff.modified is None:
                    raise ValueError(
                        f"{face_diff.status} face diff is missing its modified fingerprint"
                    )
                buckets[face_diff.status].append(modified_meshes[face_diff.modified.index])
        return

    if solid.face_diffs:
        raise ValueError(f"{solid.status} visual solids cannot contain face diffs")
    shape = solid.base_shape if solid.status == "removed" else solid.modified_shape
    if shape is None:
        side = "base" if solid.status == "removed" else "modified"
        raise ValueError(f"{solid.status} visual solid is missing its {side} shape")
    buckets[solid.status].extend(tessellate_shape(shape).values())


def _build_glb(
    buckets: dict[str, list[FaceMesh]],
    *,
    vertex_transform: VertexTransform | None = None,
) -> bytes:
    binary = bytearray()
    buffer_views: list[gltf.BufferView] = []
    accessors: list[gltf.Accessor] = []
    materials: list[gltf.Material] = []
    primitives: list[gltf.Primitive] = []

    for status in _STATUS_ORDER:
        meshes = buckets.get(status)
        if not meshes:
            continue

        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []
        for mesh in meshes:
            offset = len(vertices)
            if vertex_transform is None:
                vertices.extend(mesh.vertices)
            else:
                vertices.extend(vertex_transform(vertex) for vertex in mesh.vertices)
            triangles.extend((a + offset, b + offset, c + offset) for a, b, c in mesh.triangles)
        if not triangles:
            continue

        pos_view_idx, pos_accessor_idx = _append_positions(binary, buffer_views, accessors, vertices)
        idx_view_idx, idx_accessor_idx = _append_indices(binary, buffer_views, accessors, triangles, vertex_count=len(vertices))

        material_idx = len(materials)
        r, g, b, a = _STATUS_COLOR[status]
        materials.append(
            gltf.Material(
                name=status,
                pbrMetallicRoughness=gltf.PbrMetallicRoughness(baseColorFactor=[r, g, b, a], metallicFactor=0.05, roughnessFactor=0.7),
                alphaMode="BLEND" if a < 1.0 else "OPAQUE",
                doubleSided=True,
            )
        )
        primitives.append(gltf.Primitive(attributes=gltf.Attributes(POSITION=pos_accessor_idx), indices=idx_accessor_idx, material=material_idx))

    document = gltf.GLTF2(
        asset=gltf.Asset(version="2.0", generator="cad-diff"),
        scene=0,
        scenes=[gltf.Scene(nodes=[0])],
        nodes=[gltf.Node(mesh=0)],
        meshes=[gltf.Mesh(primitives=primitives)],
        materials=materials,
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[gltf.Buffer(byteLength=len(binary))],
    )
    document.set_binary_blob(bytes(binary))
    return b"".join(document.save_to_bytes())


def _append_positions(binary: bytearray, buffer_views: list, accessors: list, vertices: list[tuple[float, float, float]]) -> tuple[int, int]:
    offset = len(binary)
    for v in vertices:
        binary.extend(struct.pack("<3f", *v))
    _pad_to(binary, 4)

    xs, ys, zs = zip(*vertices)
    buffer_views.append(gltf.BufferView(buffer=0, byteOffset=offset, byteLength=len(vertices) * 12, target=gltf.ARRAY_BUFFER))
    accessors.append(
        gltf.Accessor(
            bufferView=len(buffer_views) - 1,
            componentType=gltf.FLOAT,
            count=len(vertices),
            type=gltf.VEC3,
            min=[min(xs), min(ys), min(zs)],
            max=[max(xs), max(ys), max(zs)],
        )
    )
    return len(buffer_views) - 1, len(accessors) - 1


def _append_indices(binary: bytearray, buffer_views: list, accessors: list, triangles: list[tuple[int, int, int]], vertex_count: int) -> tuple[int, int]:
    # glTF requires UNSIGNED_INT indices for meshes this large; small diffs
    # fit in UNSIGNED_SHORT, which most viewers handle a little more cheaply.
    use_short = vertex_count <= 65535
    fmt, component_type, size = ("<H", gltf.UNSIGNED_SHORT, 2) if use_short else ("<I", gltf.UNSIGNED_INT, 4)

    offset = len(binary)
    for tri in triangles:
        for idx in tri:
            binary.extend(struct.pack(fmt, idx))
    _pad_to(binary, 4)

    buffer_views.append(gltf.BufferView(buffer=0, byteOffset=offset, byteLength=len(triangles) * 3 * size, target=gltf.ELEMENT_ARRAY_BUFFER))
    accessors.append(gltf.Accessor(bufferView=len(buffer_views) - 1, componentType=component_type, count=len(triangles) * 3, type=gltf.SCALAR))
    return len(buffer_views) - 1, len(accessors) - 1


def _pad_to(binary: bytearray, alignment: int) -> None:
    while len(binary) % alignment != 0:
        binary.append(0)

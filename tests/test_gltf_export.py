from pathlib import Path

import pygltflib

from cad_diff.face_matcher import match_faces
from cad_diff.face_signatures import extract_faces
from cad_diff.gltf_export import VisualSolid, build_assembly_diff_glb, build_diff_glb
from cad_diff.matcher import match_solids
from cad_diff.signatures import fingerprint_solid
from cad_diff.step_io import load_step

EXAMPLES = Path(__file__).parent.parent / "examples"


def _single_shape(step_path: Path):
    (name, shape), = load_step(step_path)
    return shape


def _assembly_visuals(base_path: Path, modified_path: Path) -> list[VisualSolid]:
    base_items = load_step(base_path)
    modified_items = load_step(modified_path)
    base_fps = [fingerprint_solid(name, shape) for name, shape in base_items]
    modified_fps = [fingerprint_solid(name, shape) for name, shape in modified_items]
    base_shapes = {id(fp): shape for fp, (_, shape) in zip(base_fps, base_items)}
    modified_shapes = {id(fp): shape for fp, (_, shape) in zip(modified_fps, modified_items)}

    visuals = []
    for solid_diff in match_solids(base_fps, modified_fps):
        base_shape = None if solid_diff.base is None else base_shapes[id(solid_diff.base)]
        modified_shape = None if solid_diff.modified is None else modified_shapes[id(solid_diff.modified)]
        face_diffs = ()
        if solid_diff.status == "modified":
            face_diffs = tuple(match_faces(extract_faces(base_shape), extract_faces(modified_shape)))
        visuals.append(VisualSolid(solid_diff.status, base_shape, modified_shape, face_diffs))
    return visuals


def test_glb_round_trips_and_colors_unchanged_and_modified():
    base_shape = _single_shape(EXAMPLES / "fillet_v1.step")
    mod_shape = _single_shape(EXAMPLES / "fillet_v2.step")
    diffs = match_faces(extract_faces(base_shape), extract_faces(mod_shape))

    glb_bytes = build_diff_glb(base_shape, mod_shape, diffs)
    loaded = pygltflib.GLTF2().load_from_bytes(glb_bytes)

    material_names = {m.name for m in loaded.materials}
    assert material_names == {"unchanged", "modified"}
    for material in loaded.materials:
        assert material.alphaMode == "OPAQUE"


def test_removed_geometry_is_translucent_and_from_the_base_shape():
    base_shape = _single_shape(EXAMPLES / "boss_removed_v1.step")
    mod_shape = _single_shape(EXAMPLES / "boss_removed_v2.step")
    diffs = match_faces(extract_faces(base_shape), extract_faces(mod_shape))

    glb_bytes = build_diff_glb(base_shape, mod_shape, diffs)
    loaded = pygltflib.GLTF2().load_from_bytes(glb_bytes)

    material_by_name = {m.name: m for m in loaded.materials}
    assert "removed" in material_by_name
    assert material_by_name["removed"].alphaMode == "BLEND"
    assert material_by_name["removed"].pbrMetallicRoughness.baseColorFactor[3] < 1.0


def test_empty_diff_still_produces_a_loadable_glb():
    base_shape = _single_shape(EXAMPLES / "bracket_v1.step")
    diffs = match_faces(extract_faces(base_shape), extract_faces(base_shape))  # self-diff -> all unchanged

    glb_bytes = build_diff_glb(base_shape, base_shape, diffs)
    loaded = pygltflib.GLTF2().load_from_bytes(glb_bytes)

    assert [m.name for m in loaded.materials] == ["unchanged"]


def test_assembly_glb_includes_whole_added_solid():
    glb_bytes = build_assembly_diff_glb(
        _assembly_visuals(EXAMPLES / "bracket_v1.step", EXAMPLES / "bracket_v2.step")
    )
    loaded = pygltflib.GLTF2().load_from_bytes(glb_bytes)

    assert {material.name for material in loaded.materials} >= {"modified", "added"}


def test_assembly_glb_includes_whole_removed_solid():
    glb_bytes = build_assembly_diff_glb(
        _assembly_visuals(EXAMPLES / "bracket_v2.step", EXAMPLES / "bracket_v1.step")
    )
    loaded = pygltflib.GLTF2().load_from_bytes(glb_bytes)

    material_by_name = {material.name: material for material in loaded.materials}
    assert "removed" in material_by_name
    assert material_by_name["removed"].alphaMode == "BLEND"


def test_invalid_visual_solid_fails_fast():
    import pytest

    with pytest.raises(ValueError, match="missing its modified shape"):
        build_assembly_diff_glb([VisualSolid(status="added")])

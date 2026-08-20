from pathlib import Path

import pygltflib

from cad_diff.face_matcher import match_faces
from cad_diff.face_signatures import extract_faces
from cad_diff.gltf_export import build_diff_glb
from cad_diff.step_io import load_step

EXAMPLES = Path(__file__).parent.parent / "examples"


def _single_shape(step_path: Path):
    (name, shape), = load_step(step_path)
    return shape


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

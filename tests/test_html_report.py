import json
import re
from pathlib import Path

from cad_diff.face_matcher import match_faces
from cad_diff.face_signatures import extract_faces
from cad_diff.gltf_export import build_diff_glb
from cad_diff.html_report import render_html
from cad_diff.step_io import load_step

EXAMPLES = Path(__file__).parent.parent / "examples"


def _single_shape(step_path: Path):
    (name, shape), = load_step(step_path)
    return shape


def _render_fillet_report() -> str:
    base_shape = _single_shape(EXAMPLES / "fillet_v1.step")
    mod_shape = _single_shape(EXAMPLES / "fillet_v2.step")
    diffs = match_faces(extract_faces(base_shape), extract_faces(mod_shape))
    glb_bytes = build_diff_glb(base_shape, mod_shape, diffs)
    return render_html(glb_bytes, title="fillet_v1 → fillet_v2")


def test_import_map_is_valid_json_with_every_bare_specifier():
    html = _render_fillet_report()
    match = re.search(r'<script type="importmap">\s*(\{.*?\})\s*</script>', html, re.DOTALL)
    assert match is not None

    import_map = json.loads(match.group(1))
    assert set(import_map["imports"]) == {
        "three", "three-core", "three-gltf-loader", "three-orbit-controls",
        "three-buffer-geometry-utils", "three-skeleton-utils",
    }
    for uri in import_map["imports"].values():
        assert uri.startswith("data:text/javascript;base64,")


def test_no_relative_imports_survived_the_vendoring_rewrite():
    # A relative import inside a module loaded from a data: URI won't
    # resolve the way it would from a normal file URL — every cross-file
    # reference between the vendored three.js files must be a bare
    # specifier that goes through the import map instead.
    vendor_dir = Path("src/cad_diff/viewer/vendor")
    if not vendor_dir.exists():
        vendor_dir = Path(__file__).parent.parent / "src" / "cad_diff" / "viewer" / "vendor"
    for js_file in vendor_dir.glob("*.js"):
        text = js_file.read_text(encoding="utf-8")
        for match in re.finditer(r"from\s+['\"](\.[^'\"]*)['\"]", text):
            raise AssertionError(f"{js_file.name} has an unrewritten relative import: {match.group(1)}")


def test_glb_is_embedded_and_not_html_escaped():
    base_shape = _single_shape(EXAMPLES / "fillet_v1.step")
    mod_shape = _single_shape(EXAMPLES / "fillet_v2.step")
    diffs = match_faces(extract_faces(base_shape), extract_faces(mod_shape))
    glb_bytes = build_diff_glb(base_shape, mod_shape, diffs)

    html = render_html(glb_bytes, title="test")

    import base64
    expected_b64 = base64.b64encode(glb_bytes).decode("ascii")
    assert expected_b64 in html


def test_report_has_no_premature_script_close_and_is_reasonably_sized():
    html = _render_fillet_report()
    # A stray "</script" inside embedded source would truncate the module
    # script early and silently break the page.
    script_open_count = html.count("<script")
    script_close_count = html.count("</script>")
    assert script_open_count == script_close_count
    assert len(html) > 1_000_000  # the vendored three.js alone is >2MB base64-encoded


def test_report_exposes_view_controls_bounds_and_calibration_point_messages():
    html = _render_fillet_report()

    for control_id in (
        "viewer-grid",
        "viewer-axes",
        "viewer-wireframe",
        "viewer-normals",
        "viewer-texture",
        "model-bounds",
        "picked-points",
    ):
        assert f'id="{control_id}"' in html
    assert 'type: "cadpro-point-picked"' in html
    assert 'renderer.domElement.addEventListener("dblclick"' in html

from __future__ import annotations

import base64
import json
from pathlib import Path

from jinja2 import Template

_VIEWER_DIR = Path(__file__).parent / "viewer"
_VENDOR_DIR = _VIEWER_DIR / "vendor"

# Bare specifier -> vendored file. Every relative import inside these files
# was rewritten to one of these bare names at vendoring time (see git log),
# so a static import map is enough to resolve the whole graph with no
# bundler and no network access.
_VENDOR_FILES = {
    "three": "three.module.js",
    "three-core": "three.core.js",
    "three-gltf-loader": "GLTFLoader.js",
    "three-orbit-controls": "OrbitControls.js",
    "three-buffer-geometry-utils": "BufferGeometryUtils.js",
    "three-skeleton-utils": "SkeletonUtils.js",
}


def _data_uri(js_source: str) -> str:
    encoded = base64.b64encode(js_source.encode("utf-8")).decode("ascii")
    return f"data:text/javascript;base64,{encoded}"


def _import_map_json() -> str:
    imports = {name: _data_uri((_VENDOR_DIR / filename).read_text()) for name, filename in _VENDOR_FILES.items()}
    return json.dumps({"imports": imports})


def render_html(glb_bytes: bytes, title: str) -> str:
    """A single self-contained HTML file: the diff GLB and all of three.js
    embedded inline as base64, zero network access, zero server, drag into
    a browser or attach to Slack."""
    template = Template((_VIEWER_DIR / "template.html.j2").read_text())
    return template.render(
        title=title,
        import_map_json=_import_map_json(),
        glb_base64=base64.b64encode(glb_bytes).decode("ascii"),
        viewer_js=(_VIEWER_DIR / "viewer.js").read_text(),
    )

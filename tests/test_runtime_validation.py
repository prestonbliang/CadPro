import os
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cad_diff.cli import app
from cad_diff.face_matcher import match_faces
from cad_diff.face_signatures import extract_faces
from cad_diff.gltf_export import build_diff_glb
from cad_diff.html_report import render_html
from cad_diff.step_io import load_step


ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
VENDOR = ROOT / "src" / "cad_diff" / "viewer" / "vendor"


def _fillet_assets() -> tuple[bytes, str]:
    (_, base_shape), = load_step(EXAMPLES / "fillet_v1.step")
    (_, modified_shape), = load_step(EXAMPLES / "fillet_v2.step")
    faces = match_faces(extract_faces(base_shape), extract_faces(modified_shape))
    glb = build_diff_glb(base_shape, modified_shape, faces)
    return glb, render_html(glb, title="runtime validation")


def test_node_links_vendor_graph_and_parses_generated_glb(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; cannot validate the vendored module graph")

    glb, _ = _fillet_assets()
    glb_path = tmp_path / "diff.glb"
    glb_path.write_bytes(glb)
    script = r"""
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const vendor = process.argv[2];
const glbPath = process.argv[3];
const files = {
  "three": "three.module.js",
  "three-core": "three.core.js",
  "three-gltf-loader": "GLTFLoader.js",
  "three-orbit-controls": "OrbitControls.js",
  "three-buffer-geometry-utils": "BufferGeometryUtils.js",
  "three-skeleton-utils": "SkeletonUtils.js",
};
const context = vm.createContext({
  console, setTimeout, clearTimeout, URL, Blob, fetch, performance,
  AbortController, AbortSignal,
  TextDecoder, TextEncoder,
  ProgressEvent: class ProgressEvent { constructor(type, init = {}) { this.type = type; Object.assign(this, init); } },
});
const modules = new Map();
function moduleFor(specifier) {
  if (!files[specifier]) throw new Error(`Unmapped module specifier: ${specifier}`);
  if (!modules.has(specifier)) {
    const source = fs.readFileSync(path.join(vendor, files[specifier]), "utf8");
    modules.set(specifier, new vm.SourceTextModule(source, { context, identifier: specifier }));
  }
  return modules.get(specifier);
}
const loaderModule = new vm.SourceTextModule(`
  export { GLTFLoader } from "three-gltf-loader";
  import "three-orbit-controls";
  import "three-buffer-geometry-utils";
  import "three-skeleton-utils";
`, { context, identifier: "cad-diff-runtime-validation" });
await loaderModule.link((specifier) => moduleFor(specifier));
await loaderModule.evaluate();
const bytes = fs.readFileSync(glbPath);
context.glbInput = bytes;
const arrayBuffer = vm.runInContext("new Uint8Array(glbInput).buffer", context);
await new Promise((resolve, reject) => {
  new loaderModule.namespace.GLTFLoader().parse(arrayBuffer, "", (gltf) => {
    let meshes = 0;
    gltf.scene.traverse((child) => { if (child.isMesh) meshes += 1; });
    if (meshes === 0) reject(new Error("Generated GLB contains no meshes"));
    else resolve();
  }, reject);
});
"""
    result = subprocess.run(
        [node, "--experimental-vm-modules", "--input-type=module", "-", str(VENDOR), str(glb_path)],
        input=script,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _browser_executable() -> str | None:
    configured = os.environ.get("CAD_DIFF_BROWSER")
    if configured:
        path = Path(configured)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"CAD_DIFF_BROWSER is not an executable file: {path}")
        return str(path)
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((str(path) for path in candidates if path and Path(path).is_file()), None)


def test_configured_missing_browser_fails_fast(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CAD_DIFF_BROWSER", str(tmp_path / "missing-browser"))

    with pytest.raises(RuntimeError, match="CAD_DIFF_BROWSER is not an executable file"):
        _browser_executable()


@pytest.mark.parametrize(
    ("base_name", "modified_name", "expected_statuses"),
    [
        ("fillet_v1.step", "fillet_v2.step", {"unchanged", "modified"}),
        ("bracket_v1.step", "bracket_v2.step", {"unchanged", "modified", "added"}),
        ("bracket_v2.step", "bracket_v1.step", {"unchanged", "modified", "removed"}),
    ],
)
def test_generated_report_renders_and_toggles_layers_in_headless_chromium(
    tmp_path: Path,
    base_name: str,
    modified_name: str,
    expected_statuses: set[str],
):
    browser = _browser_executable()
    if browser is None:
        pytest.skip("No Chrome, Chromium, or Edge executable is installed")

    report_path = tmp_path / "diff.html"
    cli_result = CliRunner().invoke(
        app,
        [
            str(EXAMPLES / base_name),
            str(EXAMPLES / modified_name),
            "--html",
            str(report_path),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output

    toggle_status = sorted(expected_statuses)[0]
    html = report_path.read_text(encoding="utf-8")
    interaction_probe = f"""
<script type="module">
const probe = setInterval(() => {{
  if (document.getElementById("loading").style.display !== "none") return;
  const checkbox = document.querySelector('input[data-status="{toggle_status}"]');
  if (!checkbox) throw new Error("toggle probe could not find its layer");
  checkbox.click();
  clearInterval(probe);
}}, 25);
</script>
"""
    report_path.write_text(html.replace("</body>", interaction_probe + "</body>"), encoding="utf-8")
    profile = tmp_path / "browser-profile"
    result = subprocess.run(
        [
            browser,
            "--headless=new",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-sync",
            "--metrics-recording-only",
            f"--user-data-dir={profile}",
            "--allow-file-access-from-files",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--virtual-time-budget=5000",
            "--dump-dom",
            report_path.as_uri(),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '<canvas' in result.stdout
    assert 'id="loading" style="display: none;"' in result.stdout
    assert result.stdout.count('class="legend-row"') == len(expected_statuses)
    for status in expected_statuses:
        assert f'data-status="{status}"' in result.stdout
    assert f'data-status="{toggle_status}" data-visible="false"' in result.stdout

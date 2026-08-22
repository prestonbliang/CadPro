import base64
import re
from pathlib import Path

import pygltflib
from typer.testing import CliRunner

from cad_diff.cli import app

EXAMPLES = Path(__file__).parent.parent / "examples"


def _embedded_glb(html: str) -> bytes:
    match = re.search(r'window\.__CAD_DIFF_GLB_BASE64__ = "([A-Za-z0-9+/=]+)"', html)
    assert match is not None
    return base64.b64decode(match.group(1))


def test_html_contains_all_assembly_change_statuses(tmp_path):
    output = tmp_path / "bracket-diff.html"

    result = CliRunner().invoke(
        app,
        [str(EXAMPLES / "bracket_v1.step"), str(EXAMPLES / "bracket_v2.step"), "--html", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    glb = pygltflib.GLTF2().load_from_bytes(_embedded_glb(html))
    assert {material.name for material in glb.materials} >= {"modified", "added"}


def test_self_diff_writes_unchanged_visualization(tmp_path):
    output = tmp_path / "unchanged.html"
    model = EXAMPLES / "bracket_v1.step"

    result = CliRunner().invoke(app, [str(model), str(model), "--html", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_missing_input_fails_without_writing_html(tmp_path):
    output = tmp_path / "should-not-exist.html"

    result = CliRunner().invoke(
        app,
        [str(tmp_path / "missing.step"), str(EXAMPLES / "bracket_v1.step"), "--html", str(output)],
    )

    assert result.exit_code != 0
    assert not output.exists()
    assert isinstance(result.exception, RuntimeError)


def test_corrupt_input_fails_without_writing_html(tmp_path):
    corrupt = tmp_path / "corrupt.step"
    corrupt.write_text("this is not STEP data", encoding="utf-8")
    output = tmp_path / "should-not-exist.html"

    result = CliRunner().invoke(
        app,
        [str(corrupt), str(EXAMPLES / "bracket_v1.step"), "--html", str(output)],
    )

    assert result.exit_code != 0
    assert not output.exists()
    assert isinstance(result.exception, RuntimeError)

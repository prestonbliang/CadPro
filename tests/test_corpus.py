from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cad_diff.corpus import (
    CORPUS_ENV,
    BooleanVolumeExpectation,
    CorpusCase,
    CorpusManifestError,
    configured_corpus_root,
    load_manifest,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


def _write_manifest(root: Path, document: object) -> None:
    (root / "manifest.json").write_text(json.dumps(document), encoding="utf-8")


def _valid_document() -> dict:
    return {
        "schema_version": 1,
        "cases": [
            {
                "id": "sample",
                "vendor": "Example CAD",
                "base": "base.step",
                "modified": "modified.step",
                "expected": {"solid_statuses": {"unchanged": 1}},
            }
        ],
    }


def _semantic_document() -> dict:
    document = _valid_document()
    document["cases"][0]["expected"] = {
        "solid_statuses": {"modified": 1},
        "face_statuses": {"unchanged": 5, "modified": 1},
        "boolean_volumes": {
            "added": 0.0,
            "removed": 25.75,
            "absolute_tolerance": 0.01,
        },
    }
    return document


def test_configured_corpus_is_optional(monkeypatch, tmp_path):
    monkeypatch.delenv(CORPUS_ENV, raising=False)
    assert configured_corpus_root() is None


def test_configured_missing_corpus_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setenv(CORPUS_ENV, str(tmp_path / "absent"))
    with pytest.raises(CorpusManifestError, match="not an existing directory"):
        configured_corpus_root()


def test_manifest_rejects_bad_json(tmp_path):
    (tmp_path / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(CorpusManifestError, match=r"invalid JSON.*line 1, column 2"):
        load_manifest(tmp_path)


def test_manifest_rejects_missing_declared_fixture(tmp_path):
    _write_manifest(tmp_path, _valid_document())
    with pytest.raises(CorpusManifestError, match=r"declared fixture does not exist.*base\.step"):
        load_manifest(tmp_path)


def test_manifest_loads_valid_case(tmp_path):
    (tmp_path / "base.step").touch()
    (tmp_path / "modified.step").touch()
    _write_manifest(tmp_path, _valid_document())
    (case,) = load_manifest(tmp_path)
    assert case.case_id == "sample"
    assert case.vendor == "Example CAD"
    assert case.expected_solid_statuses == {"unchanged": 1}
    assert case.expected_face_statuses is None
    assert case.expected_boolean_volumes is None


def test_manifest_loads_optional_semantic_expectations(tmp_path):
    (tmp_path / "base.step").touch()
    (tmp_path / "modified.step").touch()
    _write_manifest(tmp_path, _semantic_document())
    (case,) = load_manifest(tmp_path)
    assert case.expected_face_statuses == {"unchanged": 5, "modified": 1}
    assert case.expected_boolean_volumes is not None
    assert case.expected_boolean_volumes.removed == 25.75
    assert case.expected_boolean_volumes.absolute_tolerance == 0.01


def test_semantic_expectations_require_one_modified_solid(tmp_path):
    (tmp_path / "base.step").touch()
    (tmp_path / "modified.step").touch()
    document = _semantic_document()
    document["cases"][0]["expected"]["solid_statuses"] = {"unchanged": 1}
    _write_manifest(tmp_path, document)
    with pytest.raises(CorpusManifestError, match=r"solid_statuses\.modified to be exactly 1"):
        load_manifest(tmp_path)


def test_boolean_expectation_requires_explicit_tolerance(tmp_path):
    (tmp_path / "base.step").touch()
    (tmp_path / "modified.step").touch()
    document = _semantic_document()
    del document["cases"][0]["expected"]["boolean_volumes"]["absolute_tolerance"]
    _write_manifest(tmp_path, document)
    with pytest.raises(CorpusManifestError, match=r"boolean_volumes is missing: absolute_tolerance"):
        load_manifest(tmp_path)


def test_manifest_rejects_unknown_expectation(tmp_path):
    (tmp_path / "base.step").touch()
    (tmp_path / "modified.step").touch()
    document = _valid_document()
    document["cases"][0]["expected"]["face_status"] = {"modified": 1}
    _write_manifest(tmp_path, document)
    with pytest.raises(CorpusManifestError, match=r"expected has unknown field\(s\): face_status"):
        load_manifest(tmp_path)


def test_semantic_case_runner_on_synthetic_fixture():
    case = CorpusCase(
        case_id="synthetic-fillet-unit-test",
        vendor="OCP synthetic fixture",
        base=EXAMPLES / "fillet_v1.step",
        modified=EXAMPLES / "fillet_v2.step",
        expected_solid_statuses={"modified": 1},
        expected_face_statuses={"unchanged": 2, "modified": 5},
        expected_boolean_volumes=BooleanVolumeExpectation(
            added=0.0,
            removed=25.752,
            absolute_tolerance=0.001,
        ),
    )
    _assert_case(case)


def test_external_cross_vendor_corpus():
    root = configured_corpus_root()
    if root is None:
        pytest.skip(f"set {CORPUS_ENV} to an available external corpus directory")

    for case in load_manifest(root):
        _assert_case(case)


def _assert_case(case: CorpusCase) -> None:
    from cad_diff.matcher import match_solids
    from cad_diff.signatures import fingerprint_solid
    from cad_diff.step_io import load_step

    base_items = load_step(case.base)
    modified_items = load_step(case.modified)
    base = [fingerprint_solid(name, shape) for name, shape in base_items]
    modified = [fingerprint_solid(name, shape) for name, shape in modified_items]
    solid_diffs = match_solids(base, modified)
    actual = Counter(diff.status for diff in solid_diffs)
    assert actual == Counter(case.expected_solid_statuses), (
        f"corpus case {case.case_id!r} ({case.vendor}) produced solid statuses "
        f"{dict(actual)}, expected {case.expected_solid_statuses}"
    )

    if case.expected_face_statuses is None and case.expected_boolean_volumes is None:
        return
    (solid_diff,) = [diff for diff in solid_diffs if diff.status == "modified"]
    base_shapes = {id(fp): shape for fp, (_, shape) in zip(base, base_items)}
    modified_shapes = {id(fp): shape for fp, (_, shape) in zip(modified, modified_items)}
    base_shape = base_shapes[id(solid_diff.base)]
    modified_shape = modified_shapes[id(solid_diff.modified)]

    if case.expected_face_statuses is not None:
        from cad_diff.face_matcher import match_faces
        from cad_diff.face_signatures import extract_faces

        face_diffs = match_faces(extract_faces(base_shape), extract_faces(modified_shape))
        face_statuses = Counter(diff.status for diff in face_diffs)
        assert face_statuses == Counter(case.expected_face_statuses), (
            f"corpus case {case.case_id!r} ({case.vendor}) produced face statuses "
            f"{dict(face_statuses)}, expected {case.expected_face_statuses}"
        )

    if case.expected_boolean_volumes is not None:
        from cad_diff.boolean_diff import boolean_cross_check

        volumes = boolean_cross_check(base_shape, modified_shape)
        expected = case.expected_boolean_volumes
        assert volumes.added_volume == pytest.approx(
            expected.added, abs=expected.absolute_tolerance, rel=0.0
        )
        assert volumes.removed_volume == pytest.approx(
            expected.removed, abs=expected.absolute_tolerance, rel=0.0
        )

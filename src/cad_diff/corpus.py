from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORPUS_ENV = "CAD_DIFF_CORPUS"
MANIFEST_NAME = "manifest.json"


class CorpusManifestError(ValueError):
    """Raised when an external validation corpus has an invalid contract."""


@dataclass(frozen=True)
class BooleanVolumeExpectation:
    added: float
    removed: float
    absolute_tolerance: float


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    vendor: str
    base: Path
    modified: Path
    expected_solid_statuses: dict[str, int]
    expected_face_statuses: dict[str, int] | None
    expected_boolean_volumes: BooleanVolumeExpectation | None


def configured_corpus_root() -> Path | None:
    """Return the optional corpus root; reject a configured missing path."""
    configured = os.environ.get(CORPUS_ENV)
    if not configured:
        return None
    root = Path(configured).expanduser()
    if not root.is_dir():
        raise CorpusManifestError(f"{CORPUS_ENV} is not an existing directory: {root}")
    return root.resolve()


def load_manifest(root: Path) -> list[CorpusCase]:
    """Parse and validate ``root/manifest.json``, including declared files."""
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusManifestError(f"corpus manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusManifestError(
            f"invalid JSON in {manifest_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    document = _object(raw, "manifest")
    _reject_unknown_fields(document, {"schema_version", "cases"}, "manifest")
    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise CorpusManifestError("manifest.schema_version must be 1")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CorpusManifestError("manifest.cases must be a non-empty array")

    parsed = [_parse_case(item, index, root) for index, item in enumerate(cases)]
    ids = [case.case_id for case in parsed]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise CorpusManifestError(f"duplicate case id(s): {', '.join(duplicates)}")
    return parsed


def _parse_case(raw: Any, index: int, root: Path) -> CorpusCase:
    label = f"manifest.cases[{index}]"
    item = _object(raw, label)
    _reject_unknown_fields(item, {"id", "vendor", "base", "modified", "expected"}, label)
    case_id = _nonempty_string(item.get("id"), f"{label}.id")
    vendor = _nonempty_string(item.get("vendor"), f"{label}.vendor")
    base = _fixture(root, item.get("base"), f"{label}.base")
    modified = _fixture(root, item.get("modified"), f"{label}.modified")
    expected = _object(item.get("expected"), f"{label}.expected")
    _reject_unknown_fields(
        expected,
        {"solid_statuses", "face_statuses", "boolean_volumes"},
        f"{label}.expected",
    )
    solid_statuses = _status_counts(expected.get("solid_statuses"), f"{label}.expected.solid_statuses")
    face_statuses = _optional_status_counts(expected, "face_statuses", f"{label}.expected")
    boolean_volumes = _optional_boolean_volumes(expected, f"{label}.expected")
    if (face_statuses is not None or boolean_volumes is not None) and solid_statuses.get("modified", 0) != 1:
        raise CorpusManifestError(
            f"{label}.expected semantic checks require solid_statuses.modified to be exactly 1"
        )
    return CorpusCase(
        case_id,
        vendor,
        base,
        modified,
        solid_statuses,
        face_statuses,
        boolean_volumes,
    )


def _fixture(root: Path, value: Any, label: str) -> Path:
    relative = Path(_nonempty_string(value, label))
    if relative.is_absolute():
        raise CorpusManifestError(f"{label} must be relative to the corpus root")
    resolved = (root / relative).resolve()
    if root != resolved and root not in resolved.parents:
        raise CorpusManifestError(f"{label} escapes the corpus root: {relative}")
    if not resolved.is_file():
        raise CorpusManifestError(f"declared fixture does not exist for {label}: {resolved}")
    return resolved


def _status_counts(value: Any, label: str) -> dict[str, int]:
    statuses = _object(value, label)
    allowed = {"unchanged", "modified", "added", "removed"}
    unknown = set(statuses) - allowed
    if unknown:
        raise CorpusManifestError(f"{label} has unknown status(es): {', '.join(sorted(unknown))}")
    if not statuses:
        raise CorpusManifestError(f"{label} must not be empty")
    for status, count in statuses.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise CorpusManifestError(f"{label}.{status} must be a non-negative integer")
    return dict(statuses)


def _optional_status_counts(parent: dict[str, Any], key: str, label: str) -> dict[str, int] | None:
    if key not in parent:
        return None
    return _status_counts(parent[key], f"{label}.{key}")


def _optional_boolean_volumes(
    expected: dict[str, Any], label: str
) -> BooleanVolumeExpectation | None:
    key = "boolean_volumes"
    if key not in expected:
        return None
    raw = _object(expected[key], f"{label}.{key}")
    required = {"added", "removed", "absolute_tolerance"}
    missing = required - set(raw)
    unknown = set(raw) - required
    if missing:
        raise CorpusManifestError(f"{label}.{key} is missing: {', '.join(sorted(missing))}")
    if unknown:
        raise CorpusManifestError(f"{label}.{key} has unknown field(s): {', '.join(sorted(unknown))}")
    added = _nonnegative_number(raw["added"], f"{label}.{key}.added")
    removed = _nonnegative_number(raw["removed"], f"{label}.{key}.removed")
    tolerance = _nonnegative_number(
        raw["absolute_tolerance"], f"{label}.{key}.absolute_tolerance"
    )
    return BooleanVolumeExpectation(added, removed, tolerance)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CorpusManifestError(f"{label} must be an object")
    return value


def _reject_unknown_fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise CorpusManifestError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusManifestError(f"{label} must be a non-empty string")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise CorpusManifestError(f"{label} must be a finite non-negative number")
    return float(value)

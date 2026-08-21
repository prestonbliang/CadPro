# External STEP validation corpus

Real CAD-vendor exports are intentionally not committed or synthesized here. Put an
independently sourced corpus in any directory and point `CAD_DIFF_CORPUS` at it. The
directory must contain `manifest.json` and every STEP file declared by that manifest.

## Manifest schema

The manifest is UTF-8 JSON with this shape:

```json
{
  "schema_version": 1,
  "cases": [
    {
      "id": "unique-case-name",
      "vendor": "CAD product and version",
      "base": "relative/path/to/base.step",
      "modified": "relative/path/to/modified.step",
      "expected": {
        "solid_statuses": {
          "unchanged": 0,
          "modified": 1,
          "added": 0,
          "removed": 0
        },
        "face_statuses": {
          "unchanged": 5,
          "modified": 1,
          "added": 0,
          "removed": 0
        },
        "boolean_volumes": {
          "added": 0.0,
          "removed": 25.75,
          "absolute_tolerance": 0.01
        }
      }
    }
  ]
}
```

`id` values must be unique. Fixture paths must be relative, remain inside the corpus
directory, and name existing files. `solid_statuses` may contain any of `unchanged`,
`modified`, `added`, and `removed`; counts are non-negative integers. Omitted statuses
mean zero because the observed status map must exactly match the manifest.

`face_statuses` and `boolean_volumes` are optional semantic checks. They are allowed
only when `solid_statuses.modified` is exactly `1`, ensuring there is one unambiguous
matched solid pair to inspect. Face statuses use the same names and count rules as
solid statuses. Boolean volumes are cubic model units and require expected `added` and
`removed` values plus an explicit non-negative `absolute_tolerance`. These assertions
exercise the face-matching cascade and independent boolean cross-check, respectively.

Malformed JSON, unsupported schema versions, invalid fields, duplicate IDs, missing
declared fixtures, and a configured corpus path that does not exist fail the test with
a specific error. Unknown fields are rejected so misspelled expectations cannot pass
silently. The optional test skips only when the environment variable is unset, without
implying that cross-vendor behavior has been validated.

## Run on Windows PowerShell

```powershell
$env:CAD_DIFF_CORPUS = "C:\path\to\private-step-corpus"
.venv\Scripts\python.exe -m pytest tests/test_corpus.py -v
```

Remove the variable after testing if desired:

```powershell
Remove-Item Env:CAD_DIFF_CORPUS
```

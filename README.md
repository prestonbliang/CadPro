# cad-diff

Semantic version control for mechanical CAD. A git-diff for solid geometry — point it
at two STEP files and get back what actually changed, not "the binary is different."

```
$ cad-diff examples/fillet_v1.step examples/fillet_v2.step

   examples/fillet_v1.step  →  examples/fillet_v2.step
┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Solid   ┃ Status   ┃ Volume Δ (mm³) ┃ Surface Δ (mm²) ┃
┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ solid_1 │ modified │        -25.752 │         -13.735 │
└─────────┴──────────┴────────────────┴─────────────────┘

solid_1 face detail  (boolean cross-check: +0.000 / -25.752 mm³, tier 5 ground truth)
┏━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Face ┃ Type     ┃ Status    ┃ Tier ┃ Area Δ (mm²) ┃ Param Δ        ┃
┡━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 3    │ Cylinder │ modified  │ T1   │      +31.416 │ radius: +2.000 │
└──────┴──────────┴───────────┴──────┴──────────────┴────────────────┘
```

That's the whole pitch in one line: a fillet radius changed from 2mm to 4mm, and it's
reported as a modified face with a dimensional delta — not an unmatched delete+add.

## Status: Phase 3 — visual diff

- **Tier 0** — whole-solid matching by geometric fingerprint (volume, surface area,
  center of mass, bounding box), robust to assembly reordering. Rejects pairings that
  are too dissimilar to be the same solid, instead of force-matching whatever's left.
- **Tiers 1–4** — face-level matching cascade: bucket-and-assign by surface type
  (Tier 1), adjacency-graph propagation from confident anchors (Tier 3), residual
  subgraph isomorphism for what's left (Tier 4) — never run on the whole model, only
  the small leftover island, since general graph isomorphism is NP-complete.
- **Tier 5** — independent boolean (`Cut`/`Common`) volumetric cross-check, the same
  technique SolidWorks's own Compare Geometry tool uses internally.
- Matched faces of the same analytic surface type report the actual dimensional delta
  (`radius: +2.000`), not just "this region differs."
- `--html out.html` renders a color-coded 3D diff — gray/yellow/green/translucent-red
  for unchanged/modified/added/removed — as one self-contained file: the geometry
  (glTF) and all of three.js are embedded as base64 data URIs behind a static import
  map, so it opens with zero network access and zero server, drag-into-browser or
  attach-to-Slack.

**Real-world validation, two vendors now:** `examples/real_world/` holds real,
non-OCP-authored parts from two different CAD export pipelines — a u-blox SolidWorks
2014 assembly (housing, PCB, antenna — one with actual free-form `BSplineSurface`
faces) and an Adafruit Fusion 360 part, both real products, not synthetic fixtures
(see `NOTICE.md` for source and license on each). Findings:
- The XCAF loader originally treated a real assembly as one opaque compound. Real
  assemblies decompose into multiple named leaf solids (`load_step` now walks and
  positions them recursively); fixed after this file exposed it.
- Solid-level matching found the loose bug documented above (`NOT_A_MATCH_COST` never
  wired up) — same class of gap, real data surfaced it faster than synthetic data did.
- Re-exporting the same real assembly in a different STEP schema (AP203 → AP214, 23
  seconds apart, same SolidWorks session) correctly reports all three parts unchanged —
  zero false positives from schema noise.
- A genuine small edit (a 0.3mm hole cut into a real 54-face SolidWorks housing wall)
  stays correctly localized: exactly 1 added face + 2 modified faces out of 55, not a
  cascade of spurious changes. The real `BSplineSurface` faces on the antenna
  participate in matching correctly and never get a fabricated dimensional delta.
- The same edit on the real Fusion 360 part exposed a second real bug: Tier 5's boolean
  cross-check (`BRepAlgoAPI_Cut`) reported success (`IsDone() == True`) while producing
  a geometrically **invalid** result — two independently STEP-round-tripped copies of
  "the same" solid don't align to sub-micron tolerance everywhere, and the boolean op
  silently degenerated instead of failing loudly. It printed `+1768/-1772mm³` for an
  actual ~4mm³ edit. Fixed by checking `BRepCheck_Analyzer(...).IsValid()` and refusing
  to report the cross-check as trustworthy when it isn't — the face-level diff (which
  stayed correct throughout) is the one you should trust either way.

**Still open — the actual go/no-go gate:** two vendors (SolidWorks, Fusion 360) is
real progress but not the full picture. NX and Creo exports remain untested, and I
don't have those CAD packages to produce that data myself.

**Also honest:** the `--html` viewer's rendering pipeline (WebGL, DOM, OrbitControls)
is unverified in a real browser — this sandbox has no attached display and every
headless-browser option (Playwright's Chromium/WebKit, Safari automation) is blocked by
the OS version or a GUI permission dialog. What *is* verified, independently, in a real
JS engine (Node): the entire vendored three.js/GLTFLoader/OrbitControls module graph
links and executes with no missing exports, three.js's own `GLTFLoader` successfully
parses cad-diff's actual GLB output (not just pygltflib's), and the base64 payload
embedded in a real generated report decodes byte-for-byte identical to the source GLB.
Open a generated report in a real browser before trusting it renders correctly.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python examples/generate.py          # bracket_v1/v2 — added/modified solids
.venv/bin/cad-diff examples/bracket_v1.step examples/bracket_v2.step
.venv/bin/cad-diff examples/fillet_v1.step examples/fillet_v2.step --html /tmp/fillet_diff.html
.venv/bin/pytest
```

## Layout

```
src/cad_diff/
  step_io.py          STEP → XCAF loading
  signatures.py        per-solid geometric fingerprint (Tier 0)
  matcher.py             Tier 0: solid fingerprint matching (Hungarian assignment)
  face_signatures.py       per-face signature + adjacency graph
  face_matcher.py             Tiers 1–4: the face-matching cascade
  boolean_diff.py                Tier 5: boolean cross-check
  tessellate.py                    per-face triangulation, orientation-aware winding
  gltf_export.py                     tessellated + classified faces -> colored GLB
  html_report.py                       GLB + vendored three.js -> one self-contained HTML
  viewer/
    template.html.j2                     the HTML shell (import map, legend, canvas)
    viewer.js                              scene setup, GLB load, layer-toggle legend
    vendor/                                  three.js core + GLTFLoader + OrbitControls,
                                              relative imports rewritten to bare specifiers
  diff_model.py                    shared data contract
  report.py                          terminal rendering
  cli.py                                entry point
```

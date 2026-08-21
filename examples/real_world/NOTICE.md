# Real-world fixtures

`SAM_AP203.STEP` and `SAM_AP214.STEP` are the same real u-blox SAM module assembly
(a housing, a PCB, and an antenna — three actual SolidWorks-authored parts, one of
which has genuine free-form `BSplineSurface` faces), exported from SolidWorks 2014 in
two different STEP schemas 23 seconds apart. Unlike every other fixture in this repo,
these did not come from cad-diff's own OCP-based writer — they're real interoperability
test data, the exact kind of thing the project's roadmap flags as the actual go/no-go
gate.

Source: [u-blox/3D-Step-Models-Library](https://github.com/u-blox/3D-Step-Models-Library)
(`POS/SAM (STEP-AP203).STEP` and `POS/SAM (STEP-AP214).STEP`).

> u-blox grants you the right to use, copy, modify and distribute the Deliverable
> provided hereunder for any purpose without fee.

`tactile_switch.step` is a real Fusion 360-authored part (a tactile on/off switch), a
different CAD kernel's export pipeline than the SolidWorks files above — its header
identifies the STEP Tools ST-Developer translator Fusion 360's STEP export uses
internally, distinct provenance from the SwSTEP-branded SolidWorks header.

Source: [adafruit/Adafruit_CAD_Parts](https://github.com/adafruit/Adafruit_CAD_Parts),
`1092 Tactile On Off Switch/1092 Tactile On Off Switch.step`, MIT licensed.

`sam_cavity_v1/v2_hole.step`, `sam_ant_v1/v2_hole.step`, and
`tactile_switch_v1/v2_hole.step` are derived from the real parts above, each with a
small hole cut at a verified in-material point (see `generate_derived.py` —
regenerate with `.venv/bin/python examples/real_world/generate_derived.py`).

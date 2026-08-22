# CadPro

CadPro converts pictures and videos into real, solid STEP geometry. It extracts object
silhouettes, builds watertight OpenCascade B-reps, validates them, and writes ordinary
`.step` files that open in FreeCAD, Fusion, SolidWorks, Onshape, and other CAD tools.

The repository previously contained only `cad-diff`, a tool for comparing existing STEP
files. That command remains available for compatibility, while `cadpro` is the main tool.

## Status: Phase 2 — multi-view turntable reconstruction

CadPro now has two reconstruction paths:

- `cadpro convert` turns one image—or the clearest frame in a video—into a scaled,
  constant-thickness solid. It preserves the outside profile and visible through-holes.
- `cadpro turntable` samples one complete 360° video and intersects the silhouettes'
  viewing volumes into a solid visual hull. This recovers shape changes around the object
  instead of merely extruding one frame.

Both paths produce boundary-representation STEP solids, not an STL/triangle mesh renamed
to `.step`. Output replacement is atomic, so a failed conversion does not destroy an
existing destination file.

Supported images: PNG, JPEG, WebP, BMP, and TIFF. Supported videos: MP4, MOV, AVI, MKV,
M4V, and WebM.

## Install

CadPro requires Python 3.10 or newer.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### macOS/Linux

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Profile extrusion from an image

```powershell
.venv\Scripts\cadpro.exe convert bracket.png --width-mm 120 --depth-mm 8 -o bracket.step
```

The detected outline is scaled to 120 mm wide and extruded 8 mm deep. Without those
options, `convert` uses a 100 mm width and 10 mm depth. Transparent PNGs are supported;
otherwise CadPro estimates the background from the image border.

The same command accepts a normal video and automatically uses its clearest detectable
frame:

```powershell
.venv\Scripts\cadpro.exe convert inspection.mp4 --width-mm 75 --depth-mm 4 -o plate.step
```

This mode is best for flat plates, brackets, gaskets, signs, and other parts with a
constant thickness.

## 3D visual hull from a turntable video

Record one complete, constant-speed revolution and run:

```powershell
.venv\Scripts\cadpro.exe turntable part.mp4 --width-mm 75 --views 12 -o part.step
```

Here, `--width-mm` means the real maximum horizontal span visible anywhere in the complete
rotation. CadPro uses one consistent pixel-to-millimeter scale across every view.

If the useful revolution occupies only part of a longer recording, provide an inclusive
start frame and exclusive end frame:

```powershell
.venv\Scripts\cadpro.exe turntable raw.mp4 --width-mm 75 --views 12 `
  --start-frame 40 --end-frame 280 --clockwise -o part.step
```

Rotation direction is viewed from above. The default is `--counterclockwise`; choosing the
wrong direction mirrors asymmetric geometry. More views capture the silhouette more
closely but create more faces and take longer. The supported range is 4–24, with 8 as the
default.

### Turntable capture checklist

1. Trim or select exactly one 360° revolution at roughly constant speed.
2. Keep the camera fixed and level—no panning, zooming, or autofocus breathing.
3. Aim the camera at the vertical rotation axis and keep that axis at frame center.
4. Use a distant or zoomed camera for an approximately orthographic view.
5. Keep the entire object visible in every sampled frame; border contact is rejected.
6. Use a plain, contrasting background with no platform or shadow merged into the object.
7. Keep the object rigidly fixed to the rotation axis.

## Honest limitations

A visual hull reconstructs only geometry that changes an object's silhouettes. It cannot
recover concavities that are never visible in an outline, internal cavities, top-only
holes, texture, material, or exact analytic features such as an inferred cylinder radius.
Perspective, camera movement, turntable wobble, reflections, transparency, motion blur,
and very thin features reduce accuracy. The result is real CAD geometry, but it is a
silhouette-derived approximation rather than a native parametric feature tree.

A single image contains even less depth information, so `convert` intentionally limits
that case to a measured profile extrusion instead of inventing hidden geometry.

## Development

```powershell
.venv\Scripts\python.exe -m pytest
```

Core modules:

```text
src/cadpro/media.py     decoding, segmentation, contour extraction, frame sampling
src/cadpro/step.py      profile solids, visual hull, validation, atomic STEP export
src/cadpro/pipeline.py  profile and turntable conversion APIs
src/cadpro/cli.py       convert and turntable commands
```

The earlier semantic STEP comparison command remains available:

```powershell
.venv\Scripts\cad-diff.exe old.step new.step --html diff.html
```

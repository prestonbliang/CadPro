# CadPro

CadPro converts a picture or video of a flat mechanical part into a real, solid STEP file.
It detects the object's outside profile and visible through-holes, scales that profile to a
real-world width, builds a watertight OpenCascade B-rep, and extrudes it to the requested
depth. The resulting `.step` file opens in normal CAD software such as FreeCAD, Fusion,
SolidWorks, and Onshape.

This repository previously contained only `cad-diff`, a tool for comparing existing STEP
files. That command is still included for compatibility, but `cadpro` is now the main tool.

## What works now

- PNG, JPEG, WebP, BMP, and TIFF images
- MP4, MOV, AVI, MKV, M4V, and WebM videos
- Automatic background/foreground separation
- Transparent PNG input
- Outside contours and visible through-holes
- User-controlled part width and extrusion depth in millimeters
- Validity checking before export
- A true STEP solid, not an STL/triangle mesh with a different extension

For video, CadPro samples the clip and uses the sharpest frame with a clean detectable
silhouette. Put the part on a plain, contrasting background and film it approximately
straight-on.

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

## Convert an image

```powershell
.venv\Scripts\cadpro.exe convert bracket.png --width-mm 120 --depth-mm 8 --output bracket.step
```

The detected outline will be scaled to 120 mm wide and extruded 8 mm deep. If the size
flags are omitted, CadPro uses a 100 mm width and 10 mm depth.

## Convert a video

```powershell
.venv\Scripts\cadpro.exe convert turntable.mp4 --width-mm 75 --depth-mm 4 -o plate.step
```

The command prints the selected frame, outline point count, and number of holes when the
conversion succeeds.

## Getting a good result

1. Place the part on a plain background with strong color/brightness contrast.
2. Keep the camera perpendicular to the face of the part to avoid perspective distortion.
3. Make sure the entire outline is visible and evenly lit.
4. Supply one known physical dimension with `--width-mm`.
5. Use `--depth-mm` for the part's measured thickness.

## Important limitation

A single ordinary photo does not contain the hidden dimensions needed to reconstruct an
arbitrary 3D object. This first version intentionally solves the useful, verifiable case:
flat or constant-thickness parts that can be represented by extruding a photographed
profile. It does not invent unseen back-side features, infer stepped depths, or perform
multi-view photogrammetry. Those require calibrated multi-view reconstruction plus feature
recognition and are future work.

## Development

```powershell
.venv\Scripts\python.exe -m pytest
```

Core modules:

```text
src/cadpro/media.py     image/video decoding, segmentation, contour extraction
src/cadpro/step.py      contour-to-B-rep construction, validation, STEP export
src/cadpro/pipeline.py  conversion API
src/cadpro/cli.py       command-line interface
```

The earlier semantic STEP comparison command remains available as:

```powershell
.venv\Scripts\cad-diff.exe old.step new.step --html diff.html
```

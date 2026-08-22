# CadPro

CadPro turns exactly one clean object image into a validated, measured 2.5D CAD solid
through a guided web application. Upload one image, enter the object's measured width,
choose the extrusion depth, inspect the result, and download:

- **STEP** for Onshape, Fusion, SolidWorks, FreeCAD, and other CAD systems
- **STL** for Blender, slicers, and mesh workflows
- **GLB** for Blender, web viewers, and real-time 3D tools
- **JSON diagnostics** recording the inputs and verified geometry measurements

CadPro extracts the object's front-view silhouette, scales its horizontal span to the
width you entered, and extrudes that profile uniformly to the chosen depth. The STEP
output is a genuine OpenCascade boundary-representation solid. CadPro reloads every
exported STEP and rejects it unless it contains exactly one valid volumetric solid. STL
and GLB are included because Blender does not natively import STEP in a standard
installation.

## Version 1.1 — single-image web workflow

- Responsive single-image upload studio
- Exactly-one-file enforcement in the browser and API
- Measured-width calibration and user-selected extrusion depth
- Client- and server-side format, dimension, size, and scale checks
- Private asynchronous reconstruction jobs with progress and actionable errors
- Automatic background separation and silhouette extraction
- Profile and visible-hole extraction with uniform-depth extrusion
- OpenCascade validity, connected-solid, positive-volume, and STEP round-trip checks
- Interactive offline 3D result preview
- Atomic STEP, binary STL, GLB, HTML preview, and report generation
- Opaque artifact download IDs; server filesystem paths are never sent to the browser
- Bounded admission (one active reconstruction plus one queued upload by default)
- 24-hour job expiry with periodic cleanup while running and cleanup at shutdown
- Trusted Host, same-origin browser upload, and early multipart body-size enforcement
- Docker packaging and health endpoint
- Advanced command-line converters and `cad-diff` remain available

## Install and launch

CadPro requires Python 3.10 or newer.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\cadpro.exe web
```

### macOS/Linux

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/cadpro web
```

The website opens at `http://127.0.0.1:8000`. Use `cadpro web --no-open` when you do not
want CadPro to open a browser automatically.

## Single-image workflow

1. Put the object against a plain background that strongly contrasts with its outline.
2. Photograph the profile square-on. Keep the entire object visible and away from every
   image edge; minimize perspective by moving farther away and zooming in if practical.
3. Select exactly one JPEG, PNG, WebP, or BMP image in the website.
4. Enter the object's measured maximum horizontal width in millimeters.
5. Choose the uniform extrusion depth you want in millimeters.
6. Build the model, inspect the preview, and download the appropriate CAD or mesh format.

Uploads are limited to 25 MiB, 12.5 million pixels total, and 8,192 pixels on either
side so malformed or extreme images cannot exhaust reconstruction memory.

The width is a real calibration measurement. The depth is a design input, not a value
inferred from the photograph. For example, a 120 mm-wide bracket photographed from the
front with an entered depth of 8 mm produces its measured front profile extruded exactly
8 mm. A visible enclosed opening becomes a through-hole through that extrusion.

This is intentionally a **2.5D profile extrusion**. One image cannot reveal the object's
back, side-wall shape, changing depth, or hidden geometry. See
[Technical truth and limitations](#technical-truth-and-limitations) before using an
output for engineering or manufacturing.

## Docker

```bash
docker build -t cadpro .
docker run --rm -p 127.0.0.1:8000:8000 \
  -e CADPRO_PUBLIC_ORIGIN=http://localhost:8000 \
  cadpro
```

The explicit `127.0.0.1` publish address keeps the container reachable only from the local
machine. `CADPRO_PUBLIC_ORIGIN` supplies social-preview URLs and the permitted browser
upload origin; its hostname is also trusted for the HTTP `Host` header. Loopback hosts are
always trusted. For a reverse proxy with another internal host, add a comma-separated
`CADPRO_TRUSTED_HOSTS` value and configure the proxy to preserve the public Host. The image
does not trust forwarded headers by default.

The service has no user accounts. Keep it on localhost, or place it behind authentication,
TLS, and network-level upload/rate limits before exposing it to the public internet. The
application itself admits at most two unfinished jobs by default, preventing an unbounded
reconstruction queue.

## Advanced command-line workflows

The website accepts one still image only. The lower-level commands below remain available
for existing scripts and experiments; they are separate from the website's single-image
workflow.

Profile-extrude one image (or the clearest frame selected from a normal video by the
legacy media converter):

```powershell
.venv\Scripts\cadpro.exe convert bracket.png --width-mm 120 --depth-mm 8 -o bracket.step
```

Run the experimental visual-hull converter on a complete turntable video:

```powershell
.venv\Scripts\cadpro.exe turntable part.mp4 --width-mm 75 --views 24 -o part.step
```

Compare two existing STEP files:

```powershell
.venv\Scripts\cad-diff.exe old.step new.step --html diff.html
```

## STEP comparison and real-world validation

The included `cad-diff` tool provides semantic version control for mechanical CAD: it
reports which solids and faces changed instead of treating two STEP files as unrelated
binary blobs.

- **Tier 0** matches whole solids by volume, surface area, center of mass, and bounding
  box, while tolerating assembly reordering and rejecting implausible matches.
- **Tiers 1–4** match faces by analytic surface type, adjacency, and residual subgraph
  structure. Matched analytic faces report dimensional changes such as
  `radius: +2.000`.
- **Tier 5** independently cross-checks added and removed volume with OpenCascade boolean
  operations. A cross-check is reported only when the resulting shapes pass geometric
  validity checks.
- `--html out.html` creates a self-contained, offline 3D report with unchanged, modified,
  added, and removed geometry shown as separate color-coded layers.

The bundled `examples/real_world/` corpus covers SolidWorks 2014 and Fusion 360 exports,
including assemblies, schema changes, free-form `BSplineSurface` faces, and localized
edits. See `examples/real_world/NOTICE.md` for provenance and licensing. It exposed and
helped fix assembly-leaf traversal, overly permissive solid matching, and a failure mode
where a boolean operation claimed success but returned invalid geometry.

Observed regression results on that corpus include:

- Re-exporting one three-part SolidWorks assembly from AP203 to AP214 reports all three
  parts unchanged, with no false positives from schema noise.
- Cutting a 0.3 mm hole in a 54-face housing remains localized to one added and two
  modified faces; its free-form antenna faces still match without invented dimensions.
- A 7 MB, four-part, 1,700+-face SolidWorks assembly loaded in about five seconds. Its
  811-face PCB self-diff completed in 0.33 seconds, including a 671×671 plane-face
  assignment.
- An invalid Fusion 360 boolean result that overstated an approximately 4 mm³ edit as
  thousands of cubic millimeters is now rejected instead of presented as ground truth.

These are measured development results, not universal performance guarantees. Coverage
still does not include NX, Creo, or CATIA exports. An additional licensed or private
vendor corpus can be exercised through `CAD_DIFF_CORPUS`; its manifest format and test
command are documented in `docs/external-corpus.md`.

## Technical truth and limitations

The website creates a **measured 2.5D profile extrusion**, not a full 3D reconstruction
and not a native parametric feature history. Its front outline and any visible enclosed
holes come from the image. The horizontal scale comes from the measured width you enter,
and every point is extruded by the exact depth you choose.

A single image cannot infer:

- the true depth, backside outline, or rear-face features
- pockets, steps, bosses, side holes, or other changes along the extrusion direction
- hidden cavities, internal structure, or features obscured in the photograph
- exact design intent such as a nominal radius, thread, tolerance, or material

Perspective, lens distortion, reflections, transparency, shadows joined to the object,
blur, low contrast, and very thin features reduce profile accuracy. A visible opening is
treated as a through-hole; CadPro cannot know whether it is actually blind. No software
can recover geometry that one view never observes, so verify critical dimensions and add
missing features in CAD before manufacturing.

The advanced turntable CLI builds a visual hull from multiple silhouettes and has
different capture requirements, but it still cannot recover concavities or hidden
structure that never affect a silhouette.

## Development

```powershell
.venv\Scripts\python.exe -m pytest
```

```text
src/cadpro/web.py          single-image API, job queue, artifact security, website serving
src/cadpro/web_assets/     responsive single-image profile-extrusion interface
src/cadpro/reconstruct.py  advanced ordered-photo and sampled-video reconstruction
src/cadpro/artifacts.py    STEP/STL/GLB/preview/report export and verification
src/cadpro/media.py        decoding, segmentation, and contour extraction
src/cadpro/step.py         B-rep construction and visual-hull booleans
```

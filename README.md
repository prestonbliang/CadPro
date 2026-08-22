# CadPro

CadPro turns a complete photo orbit or turntable video into validated 3D geometry through
a guided web application. Upload 20–50 ordered photos—or one steady 360° video—enter one
real measurement, inspect the reconstructed model, and download:

- **STEP** for Onshape, Fusion, SolidWorks, FreeCAD, and other CAD systems
- **STL** for Blender, slicers, and mesh workflows
- **GLB** for Blender, web viewers, and real-time 3D tools
- **JSON diagnostics** recording the inputs and verified geometry measurements

The STEP output is a genuine OpenCascade boundary-representation solid. CadPro reloads
every exported STEP and rejects it unless it contains exactly one valid volumetric solid.
STL and GLB are included because Blender does not natively import STEP in a standard
installation.

## Version 1.0 — complete web workflow

- Responsive photo/video upload studio
- Ordered photo thumbnails with manual reordering
- Client- and server-side count, format, dimension, size, and scale checks
- Private asynchronous reconstruction jobs with progress and actionable errors
- Automatic background separation and silhouette extraction
- 20–50 evenly spaced views around one complete revolution
- Clockwise and counterclockwise capture support
- Fixed-axis, consistently scaled visual-hull reconstruction
- OpenCascade validity, connected-solid, positive-volume, and STEP round-trip checks
- Interactive offline 3D result preview
- Atomic STEP, binary STL, GLB, HTML preview, and report generation
- Opaque artifact download IDs; server filesystem paths are never sent to the browser
- Bounded admission (one active reconstruction plus one queued upload by default)
- 24-hour job expiry with periodic cleanup while running and cleanup at shutdown
- Trusted Host, same-origin browser upload, and early multipart body-size enforcement
- Docker packaging and health endpoint
- The earlier command-line converters and `cad-diff` remain available

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

## Photo workflow

1. Place the object on a turntable against a plain, contrasting background.
2. Fix the camera on a tripod, level with the object and aimed at the rotation axis.
3. Take 20–50 evenly spaced photos over exactly one full revolution.
4. Keep camera position, zoom, focus, image dimensions, and lighting unchanged.
5. Select the images in rotation order. The website lets you correct their order.
6. Enter the object's real maximum horizontal span in millimeters.
7. Choose the rotation direction as viewed from above and build the model.

CadPro interprets photo `n` as angle `n × 360 / photo_count`; filenames do not supply
angles. Every photo must show the complete object without touching the image border.

## Video workflow

Record exactly one constant-speed 360° revolution, then select 20–50 sampled views in the
website. A distant or zoomed camera gives a closer approximation to the orthographic
projection used by visual-hull reconstruction.

If only part of a longer recording contains the clean revolution, the Python API and CLI
also support a half-open frame range: the start frame is included and the end frame is not.

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

## Command-line workflows

Profile-extrude one image or the clearest frame in a normal video:

```powershell
.venv\Scripts\cadpro.exe convert bracket.png --width-mm 120 --depth-mm 8 -o bracket.step
```

Reconstruct a turntable video directly:

```powershell
.venv\Scripts\cadpro.exe turntable part.mp4 --width-mm 75 --views 24 -o part.step
```

Compare two existing STEP files:

```powershell
.venv\Scripts\cad-diff.exe old.step new.step --html diff.html
```

## Technical truth and limitations

CadPro's multi-view path builds a **visual hull** by intersecting the viewing volume of
every extracted silhouette. This is real, watertight CAD geometry, but it is not magic and
it is not the same as a native parametric feature history.

Ordinary photos cannot reveal:

- concavities that never alter an outside silhouette
- hidden cavities or internal structure
- top- or bottom-only features not seen by the level camera
- exact design intent such as a nominal radius, thread, tolerance, or material

Perspective, camera movement, a miscentered rotation axis, turntable wobble, reflections,
transparency, shadows joined to the object, motion blur, and very thin features reduce
accuracy. CadPro detects many unusable captures and fails clearly, but no software can
make an exact engineering model from visual information that the capture never contains.
For production parts, use the result as a reconstruction/reference body and verify critical
dimensions in CAD before manufacturing.

## Development

```powershell
.venv\Scripts\python.exe -m pytest
```

```text
src/cadpro/web.py          upload API, job queue, artifact security, website serving
src/cadpro/web_assets/     responsive reconstruction interface
src/cadpro/reconstruct.py  ordered-photo and sampled-video reconstruction
src/cadpro/artifacts.py    STEP/STL/GLB/preview/report export and verification
src/cadpro/media.py        decoding, segmentation, and contour extraction
src/cadpro/step.py         B-rep construction and visual-hull booleans
```

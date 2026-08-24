# CadPro

CadPro turns object captures into interoperable CAD and 3D files through a local web
application. Version 2.2 keeps two deliberately separate output lanes:

- **Measured STEP reconstruction** uses a real measurement and local geometry processing to
  create the validated CAD exports described below.
- **AI visual mesh generation** optionally sends a prompt or representative images to Meshy
  and returns visual GLB/STL assets. These meshes are non-metric and are never STEP.

The measured reconstruction lane accepts exactly one of these capture types:

- one object photo;
- 20–50 ordered photos covering one complete revolution; or
- one turntable video, sampled into 20–50 evenly spaced views.

Every successful measured reconstruction exports a validated **STEP** solid, binary **STL**,
**GLB**, an interactive offline HTML preview, and a JSON reconstruction report. STEP is
intended for Onshape, Fusion, SolidWorks, FreeCAD, and similar CAD systems. STL and GLB are
included for Blender, slicers, web viewers, and mesh workflows. The optional Meshy lane
instead exports an AI-generated GLB, STL, preview, and provider report; it does not create or
modify a STEP solid.

CadPro is measurement-driven reconstruction, not a magic recovery of hidden design intent.
Read [Technical truth and limitations](#technical-truth-and-limitations) before using an
output for engineering or manufacturing.

## Version 2.2 capability contract

| Website mode | Required input | Measurement | Geometry produced |
| --- | --- | --- | --- |
| One photo | Exactly one square-on object image | Real profile width; chosen depth or trained neural depth estimate | Measurement-scaled 2.5D silhouette/profile extrusion |
| Photo orbit | 20–50 ordered, evenly spaced photos | Real maximum width | Measurement-scaled silhouette visual hull |
| Turntable video | Exactly one steady full-revolution video; choose 20–50 sampled views | Real maximum width | Measurement-scaled silhouette visual hull |
| Optional Meshy visual mesh | Text, one image, or at most four provider-selected representative views | None; output is non-metric | Generative polygon GLB/STL only; never STEP |

The three measured modes share the same guarded job queue, isolated transient storage, progress
UI, OpenCascade export pipeline, and artifact verification. A completed STEP file is reloaded
and rejected unless it contains exactly one valid, connected, positive-volume solid. STL and
GLB are validated as non-empty geometry before the result is published. Meshy jobs use a
separate external-provider path and never enter that STEP export pipeline.

Version 2.2 includes:

- an optional, server-side Meshy provider for text, one-image, and representative multi-view
  AI visual-mesh generation, kept separate from measured STEP reconstruction;
- a trainable local neural network that predicts bounded depth-to-width ratios from images;
- JSONL datasets using labeled dimensions or aligned image/STEP training pairs;
- safe data-only NPZ checkpoints, Adam training, CLI inference, and website inference;
- responsive one-photo, photo-orbit, and turntable-video workflows;
- browser- and server-side file count, type, byte, pixel, and measurement limits;
- order-preserving photo upload with sequential, memory-bounded thumbnail inspection;
- configurable turntable direction and 20–50-view video sampling;
- measured profile extrusion for one photo and silhouette intersection for full orbits;
- optional OpenAI vision and cited web-reference research, kept separate from geometry;
- private asynchronous jobs, opaque artifact IDs, expiry cleanup, and bounded admission;
- Trusted Host, same-origin upload, and early multipart request-size enforcement;
- STEP, STL, GLB, HTML preview, and JSON report generation;
- Docker packaging, a health endpoint, command-line converters, and `cad-diff`.

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

Open `http://127.0.0.1:8000`. Use `cadpro web --no-open` when you do not want CadPro to
open a browser automatically.

## Capture and reconstruction workflows

### One photo

1. Put the object against a plain background that strongly contrasts with its outline.
2. Photograph the desired profile square-on, with the whole object clear of every edge.
3. Choose **One photo** and select one JPEG, PNG, WebP, or BMP image.
4. Enter the object's measured horizontal width and either enter the uniform depth or select a
   configured trained neural checkpoint to predict it.
5. Build, inspect the preview, download the files, and verify critical dimensions in CAD.

CadPro extracts the visible outline and enclosed openings, scales the outline to the entered
width, and extrudes it by the chosen or predicted depth. Manual depth is an exact design input;
neural depth is a learned estimate recorded in the result and report. A visible enclosed opening
becomes a through-hole.

### Ordered photo orbit

1. Fix the camera, zoom, focus, lighting, and a plain contrasting background.
2. Keep the entire object in frame and rotate the object through exactly one revolution.
3. Capture 20–50 evenly spaced photos with identical pixel dimensions.
4. Choose **Photo orbit**, select the images in rotation order, and correct their order in the
   thumbnail strip if needed.
5. Enter the object's measured maximum width and the rotation direction as viewed from above.

CadPro extracts one silhouette from each ordered view, scales the capture from the entered
width, and intersects the view volumes into one visual hull. Photo order, even angular spacing,
stable framing, and a stationary camera are part of the reconstruction contract.

### Turntable video

1. Fix the camera and record the object completing exactly one steady 360-degree turn.
2. Avoid pauses, speed changes, camera motion, autofocus shifts, and cropped frames.
3. Choose **Turntable video**, select one supported video, and choose 20–50 sampled views.
4. Enter the object's measured maximum width and rotation direction.

CadPro samples evenly spaced frames from the selected revolution and feeds their silhouettes
through the same visual-hull reconstruction used by a photo orbit. More sampled views improve
angular coverage but cannot reveal a feature that never changes an outside silhouette.

Image files are limited to 25 MiB each, 12.5 million pixels, and 8,192 pixels on either edge.
A photo set is limited to 500 MiB and a video to 2 GiB. Server settings can lower these byte
limits. Measurements must be finite and greater than zero.

## Trainable neural image-to-STEP prediction

CadPro 2.2 includes the real train/predict pipeline in `src/cadpro/neural.py`. It converts each
image into a normalized 24 x 24 silhouette raster plus aspect, foreground, hole, and symmetry
features. A two-hidden-layer neural network learns the logarithmic depth-to-width ratio. At
inference time, the user still supplies one real width measurement; the model predicts depth,
the image supplies the visible profile, and OpenCascade builds and validates the STEP solid.

This design learns a useful hidden parameter without pretending that one image contains unseen
topology. It does not infer backside outlines, side holes, pockets, threads, tolerances, or native
feature history.

### Prepare training data

Create a UTF-8 JSON Lines file with at least four samples. Production models should use a
representative, carefully measured dataset with separate validation objectsâ€”normally hundreds
or thousands of examples, not the four-sample smoke-test minimum.
See `examples/neural_dataset.jsonl.example` for a copyable mixed-label manifest.

Use explicit labels:

```json
{"image":"images/bracket-001.png","width_mm":120.0,"depth_mm":28.0}
{"image":"images/bracket-002.png","width_mm":84.5,"depth_mm":16.0}
```

Or pair an image with one aligned STEP solid:

```json
{"image":"images/bracket-003.png","step":"steps/bracket-003.step"}
```

For STEP-paired training, align the solid so X is the photographed horizontal width and Z is the
hidden extrusion depth. CadPro rejects files without exactly one solid and derives the labels
from the STEP bounding box. Relative paths are resolved from the manifest directory.

### Train and predict

```powershell
.venv\Scripts\cadpro.exe neural-train dataset.jsonl `
  --checkpoint models/cadpro-depth-model.npz `
  --epochs 300 --batch-size 16 --validation-fraction 0.2

.venv\Scripts\cadpro.exe neural-predict bracket.png `
  --checkpoint models/cadpro-depth-model.npz `
  --width-mm 120 --output bracket-neural.step
```

Training uses deterministic NumPy operations and Adam optimization, so no multi-gigabyte ML
framework is required. Checkpoints contain arrays and JSON metadata only and are loaded with
pickle disabled. Validate a checkpoint on held-out objects from the same capture process before
enabling it for users.

### Enable the trained model on the website

```powershell
$env:CADPRO_NEURAL_ENABLED = "1"
$env:CADPRO_NEURAL_CHECKPOINT = "C:\absolute\path\to\cadpro-depth-model.npz"
.venv\Scripts\cadpro.exe web
```

Choose **One photo**, enter the measured width, and select **Predict hidden depth with a trained
neural network**. The result page and JSON report show the predicted depth, learned ratio,
validation error, heuristic confidence score, and explicit manufacturing warnings. Checkpoint
paths are never returned by the health or job APIs. A checkpoint trained without a held-out
validation split receives an automatic confidence penalty.

## Optional AI and cited web enrichment

CadPro can add an advisory research brief to a result. This is deliberately separate from the
measurement-driven reconstruction and is off by default. To enable it on the server:

```powershell
$env:CADPRO_AI_ENRICHMENT = "1"
$env:OPENAI_API_KEY = "your-api-key"
# Optional provider model override:
$env:CADPRO_AI_MODEL = "your-supported-model"
.venv\Scripts\cadpro.exe web
```

On macOS/Linux, set the same environment variables with `export`. A user must also select the
optional intelligence checkbox for an individual job. An API key by itself does not upload any
capture.

When requested, CadPro sends at most six bounded representative views to the OpenAI Responses
API for vision analysis and allows cited web search. Images are resized and compressed before
the request. The validated response can contain an object identity hypothesis, candidate
dimensions, visible feature observations, uncertainties, and source URLs. Those findings are
written into the job result and JSON report for human review.

AI/web enrichment **never changes the measured width, extrusion depth, silhouettes, B-rep, or
STEP output**. A visual estimate is not a measurement. A published dimension may belong to a
different product revision. Check every cited source and confirm the photographed object before
using any advisory information. If enrichment is disabled or fails, local reconstruction still
continues.

Enabling this feature sends representative images and the optional object hint to an external
provider. Review your provider's privacy, retention, regional-processing, and billing terms
before enabling it for confidential objects.

## Optional Meshy AI visual-mesh provider

Version 2.2 can use Meshy's hosted API to generate a detailed visual mesh from text or object
images. This provider is off by default and runs on the server so its credential is never placed
in browser JavaScript. It is an independent **AI visual mesh** lane: Meshy output never changes,
replaces, or supplies geometry to CadPro's measurement-driven STEP lane.

Create and fund a Meshy API account, create an API key in Meshy's settings, and then configure
the CadPro server:

```powershell
$env:CADPRO_MESHY_ENABLED = "1"
$env:MESHY_API_KEY = "msy_your-api-key"
.venv\Scripts\cadpro.exe web
```

On macOS/Linux, set the same variables with `export` and start the existing `cadpro web`
command. CadPro does not bundle a Meshy account, credits, model weights, or an API license.
Leaving either the feature flag or credential unavailable keeps the provider disabled without
disabling local measured reconstruction.

### Meshy inputs and provider settings

The website can submit these visual-generation inputs:

- **Text:** a description is sent through Meshy's Text-to-3D workflow.
- **One image:** one bounded object image is sent to Meshy's Image-to-3D workflow.
- **Multiple photos or video:** CadPro selects at most four representative, well-separated views
  from the ordered capture or locally sampled video frames. Meshy's Multi-Image-to-3D API accepts
  only one to four images, so it does not receive the entire 20–50-view reconstruction set or a
  raw video.

The primary/front view should show the object clearly, and every submitted view should depict
the same object with consistent lighting and little occlusion. More supplied photos do not make
Meshy's four-image limit larger; the full measured capture still belongs to CadPro's local
silhouette reconstruction lane.

The optional provider settings affect only the AI mesh:

- **Textured** asks Meshy to synthesize texture rather than return geometry alone.
- **PBR** additionally requests metallic, roughness, and normal material maps and therefore only
  applies when texturing is enabled.
- **Remesh topology and target faces** choose triangle or quad-dominant output and an approximate
  polygon target. Meshy may deviate from the requested face count. Remeshing does not recover
  analytic planes, cylinders, holes, sketches, dimensions, constraints, or CAD feature history.
- **Rigging** is optional post-processing for a suitable textured humanoid GLB. Meshy's current
  API documentation limits reliable programmatic rigging to standard biped humanoids with clear
  limbs and body structure; it is not a general rigging mode for mechanical parts, animals, or
  arbitrary objects. The character-height field guides rig scaling only and is not a measured
  dimension for CAD.

Meshy jobs run asynchronously. CadPro tracks the provider task and downloads completed assets
while their signed URLs are available. A successful visual-mesh job publishes an AI-generated
**GLB**, **STL**, interactive preview, and JSON provider report. Use GLB for materials and a
Blender/web workflow; STL contains geometry only. A successful optional humanoid-rigging stage
adds a separate rigged GLB. Inspect the report for the selected input mode, provider settings,
provenance, and warnings.

### Mesh is not STEP

Meshy generates triangles or quad-dominant polygon meshes. It does not return STEP, B-rep faces,
parametric features, design history, tolerances, or verified real-world dimensions. CadPro does
not wrap, rename, or advertise a Meshy mesh as STEP. Even when the object looks convincing or a
provider estimates scale, the result remains a **non-metric visual asset** and may hallucinate
the hidden side, close real holes, add details, or omit functional geometry.

Use the separate measured lane when a STEP file is required: provide a real width and the depth
or a full ordered capture, let the local OpenCascade pipeline build and validate the solid, and
then verify it in CAD. Use the Meshy lane for concept visualization, Blender work, game assets,
or a manual remodeling reference. Never use its apparent dimensions for manufacturing.

### Billing, data, retention, and rights

Enabling Meshy sends the selected prompt or representative images to an external provider.
Before enabling it for user, client, proprietary, or export-controlled objects, review the live
provider agreement and obtain any contract your use requires:

- Meshy's [authentication guide](https://docs.meshy.ai/en/api/authentication) explains API-key
  creation and storage. The key must remain a server secret; Meshy's
  [error reference](https://docs.meshy.ai/en/api/errors) says direct browser CORS calls are not
  permitted.
- API generation uses paid credits. Review Meshy's current
  [API pricing](https://docs.meshy.ai/en/api/pricing) and
  [rate limits](https://docs.meshy.ai/en/api/rate-limits); costs and model availability can
  change.
- Meshy's [Terms of Service](https://www.meshy.ai/terms-of-use) currently state that
  non-Enterprise API output is deleted from Meshy's service three days after generation. Keep
  required CadPro downloads and reports under your own retention policy.
- Those Terms also currently permit training on non-Enterprise customer inputs and outputs
  unless otherwise agreed. Do not promise confidential or no-training handling based only on
  enabling this integration; obtain an appropriate written plan or agreement when required.
- You must own or have permission to upload every image and to use the resulting asset. Review
  the plan-specific output license and any attribution, privacy, regional-processing, and
  commercial-use obligations before distribution.

The relevant provider workflows are documented by Meshy at
[Text to 3D](https://docs.meshy.ai/en/api/text-to-3d),
[Image to 3D](https://docs.meshy.ai/en/api/image-to-3d),
[Multi-Image to 3D](https://docs.meshy.ai/en/api/multi-image-to-3d), and
[Rigging](https://docs.meshy.ai/en/api/rigging). Provider documentation and terms can change;
review them again before each production deployment.

### Test the Meshy lane

1. Set `CADPRO_MESHY_ENABLED=1` and `MESHY_API_KEY`, restart the web server, and open the local
   website.
2. Confirm the optional Meshy controls are available. Start with a non-confidential test prompt
   or one clear image and geometry-only settings to avoid unnecessary texture/rigging credits.
3. Wait for the asynchronous job to finish, then download and open the GLB in Blender or another
   glTF viewer and inspect the STL in a mesh viewer or slicer.
4. Open the provider report and confirm its input mode, view count when applicable, requested
   settings, warnings, and visual-mesh/non-metric classification. Confirm that the Meshy result
   contains no STEP artifact.
5. Run a normal measured capture separately, download its STEP, and verify that its dimensions
   and reconstruction report come from the measured local lane rather than the Meshy job.
6. Restart without the Meshy feature flag to confirm the external controls become unavailable
   while one-photo, photo-orbit, and turntable-video STEP reconstruction remain usable.

## Legacy optional Hunyuan-compatible concept mesh worker

`src/cadpro/ml_mesh.py` contains an opt-in integration seam for an administrator-operated,
Hunyuan-compatible image-to-3D worker. When the worker is available, a website user can request
an additional concept GLB alongside the normal validated exports. Concept generation remains a
separate companion path and never changes or replaces the STEP reconstruction. The worker must
expose `POST /generate`, accept one bounded base64 JPEG in JSON, and return a self-contained
binary glTF 2.0 file.

The client contract follows Tencent's official
[Hunyuan3D-2 API server](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) pattern. Run the exact
worker/model version you have reviewed as a separate GPU service; pin its revision and follow
that repository's installation, hardware, and license instructions. If a newer worker such as
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) exposes a different response,
put a small adapter in front of it that preserves the contract above.

Configure it with:

```powershell
$env:CADPRO_ML_MESH_ENABLED = "1"
$env:CADPRO_ML_MESH_LICENSE_ACCEPTED = "1"
$env:CADPRO_ML_MESH_ENDPOINT = "https://your-worker.example/generate"
# Optional server-owned bearer token:
$env:CADPRO_ML_MESH_TOKEN = "your-worker-token"
```

Only enable this after reviewing and explicitly accepting the exact model, weights, code, and
deployment licenses used by your worker. CadPro does not bundle, endorse, or silently accept a
third-party model license. A user must also select **High-detail AI concept mesh** for an
individual job; configuring the worker does not send every capture automatically.

The returned `cadpro-ai-concept.glb` is a **non-metric visual concept mesh**. It is not derived
from the validated STEP solid, is not manufacturing CAD, and is never converted or presented as
STEP. Treat it as a visual reference for Blender or a manual remodeling workflow. The module
validates the GLB container and embedded position geometry, but that does not establish scale,
dimensional accuracy, watertightness, topology quality, or manufacturability.

## Docker and network safety

```bash
docker build -t cadpro .
docker run --rm -p 127.0.0.1:8000:8000 \
  -e CADPRO_PUBLIC_ORIGIN=http://localhost:8000 \
  cadpro
```

The explicit loopback publish address keeps the container reachable only from the local
machine. `CADPRO_PUBLIC_ORIGIN` supplies social-preview URLs and the permitted browser upload
origin; its hostname is also trusted for the HTTP `Host` header. Loopback hosts are always
trusted. For a reverse proxy, set `CADPRO_TRUSTED_HOSTS` to the necessary comma-separated hosts
and preserve the public Host header. Forwarded headers are not trusted by default.

The service has no user accounts. Keep it on localhost, or place it behind authentication, TLS,
network-level upload/rate limits, and appropriate external-provider controls before public use.
The default application admission limit allows one active reconstruction plus one queued job,
preventing an unbounded reconstruction queue. Job data expires after 24 hours by default.

## Command-line tools

Create a profile extrusion from an image (or the clearest frame selected from a normal video by
the legacy media converter):

```powershell
.venv\Scripts\cadpro.exe convert bracket.png --width-mm 120 --depth-mm 8 -o bracket.step
```

Run the lower-level experimental turntable converter:

```powershell
.venv\Scripts\cadpro.exe turntable part.mp4 --width-mm 75 --views 24 -o part.step
```

Compare two existing STEP files:

```powershell
.venv\Scripts\cad-diff.exe old.step new.step --html diff.html
```

The CLI turntable command is retained for scripts and has its own legacy sampling range. The
website capability contract is 20–50 views.

## STEP comparison

The included `cad-diff` tool provides semantic version control for mechanical CAD. It matches
solids and faces using volume, surface area, center of mass, bounding boxes, analytic surface
types, adjacency, and residual topology, then independently checks added and removed volume with
OpenCascade booleans. `--html out.html` creates a self-contained offline 3D report.

The bundled `examples/real_world/` corpus includes licensed SolidWorks and Fusion 360 exports.
See `examples/real_world/NOTICE.md` for provenance and `docs/external-corpus.md` for the optional
external corpus format.

## Technical truth and limitations

CadPro's measured reconstruction lane outputs one valid boundary-representation solid, but that
does not mean the solid is an exact copy of the original object or a native parametric
feature-history model. The optional Meshy and legacy Hunyuan lanes output polygon concept
meshes, not boundary-representation solids.

One-photo mode cannot determine:

- the true depth, backside outline, or rear-face features;
- pockets, steps, bosses, side holes, or other depth changes;
- hidden cavities, internal structure, threads, tolerances, or material; or
- whether a visible opening is a through-hole or a blind pocket.

A neural checkpoint can learn statistical depth patterns from its labeled dataset, but that
does not turn its prediction into an observation. Dataset bias, capture differences, and
out-of-distribution objects can produce confident-looking but wrong depths. Always retain the
measured width and independently verify the predicted dimension.

Photo-orbit and video modes add outside shape coverage, but a silhouette visual hull still
cannot recover concavities, cavities, holes, or recesses that never affect an outline. It can
also overfill space between visible limbs or features. It is not texture-based photogrammetry,
neural radiance-field reconstruction, or native editable design history.

Every mode is sensitive to perspective, lens distortion, reflections, transparency, shadows
connected to the object, blur, low contrast, thin features, changing zoom, bad photo order, and
an incorrect real-world width. The visual hull assumes a centered object, fixed camera, even
angular spacing, and one complete revolution.

Before manufacturing:

1. inspect the preview and reconstruction report;
2. import STEP into a trusted CAD system and run its geometry checks;
3. measure overall dimensions, holes, interfaces, wall thicknesses, and hidden features;
4. remodel missing design intent and add tolerances, threads, and material requirements; and
5. validate the final edited design against the physical object and its intended load case.

Do not use inferred or advisory dimensions for safety-critical parts without independent
measurement and qualified engineering review.

## Development and testing

```powershell
.venv\Scripts\python.exe -m pytest
```

The end-to-end suite builds actual one-photo, 20-photo, and turntable-video reconstructions,
downloads every measured artifact, and reloads the generated STEP solids. External-provider
tests must use mocked provider responses; the normal test suite should not spend Meshy credits
or upload test fixtures to a third party.

```text
src/cadpro/web.py          measured/text APIs, job queue, request guards, artifact security
src/cadpro/web_assets/     responsive capture, calibration, progress, and result interface
src/cadpro/reconstruct.py  profile extrusion and ordered silhouette visual-hull reconstruction
src/cadpro/neural.py       trainable depth model, safe checkpoints, image features, STEP inference
src/cadpro/enrichment.py   optional OpenAI vision and cited web-reference report enrichment
src/cadpro/meshy.py        optional hosted Meshy tasks and validated visual-mesh exports
src/cadpro/ml_mesh.py      optional external concept-mesh worker client and GLB validation
src/cadpro/artifacts.py    STEP/STL/GLB/preview/report export and verification
src/cadpro/media.py        decoding, frame sampling, segmentation, and contour extraction
src/cadpro/step.py         OpenCascade B-rep construction and visual-hull booleans
```

## Contributors

- Preston L
- Ethan C (`yil91974@gmail.com`)

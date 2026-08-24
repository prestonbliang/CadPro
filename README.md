# CadPro 3.0

CadPro is a local web application for turning an overlapping photo set or an orbit video into
a real photogrammetry reconstruction. It separates sparse points, dense points, triangle
meshes, printable meshes, and analytic CAD so that a good-looking mesh is never mislabeled as
a STEP model.

The product version is **3.0**. The real-scan HTTP contract is intentionally versioned at
`/api/v2`; the API version and application release version are independent.

> [!IMPORTANT]
> **Verification status on August 23, 2026:** this Windows development machine does not have
> FFmpeg, FFprobe, COLMAP, or OpenMVS installed or discoverable. `cadpro scan-doctor` reports
> photo SfM, video ingest, and camera texturing as unavailable. Blender 4.2.3 LTS, trimesh
> 4.12.2, and OpenCascade Python bindings 7.9.3.1 are available. No user photo set or video was
> reconstructed with the native pipeline on this machine. Automated coverage uses mocked
> process calls and the explicitly injected `SyntheticTestAdapter`; that adapter is test-only,
> requires `allow_test_only=True`, and is not a production fallback. Install the native tools
> below and run a real capture before treating this host as an end-to-end scanner.

## What version 3.0 does

The standard lane is local and does not require a paid cloud service:

1. stream and validate untrusted uploads;
2. inspect video with FFprobe and extract bounded candidate frames with FFmpeg;
3. reject blurry, duplicate, badly exposed, or low-feature views;
4. estimate cameras and a sparse cloud with COLMAP;
5. select the strongest COLMAP component instead of assuming `sparse/0`;
6. build dense points and a mesh with OpenMVS, or with COLMAP's CUDA dense tools;
7. refine, texture when available, and conservatively repair the triangle mesh;
8. apply scale only from an explicit two-point measurement;
9. fit a simple analytic box or right cylinder when confidence gates pass;
10. reopen and validate every advertised artifact before publishing it.

Version 3.0 does **not** infer an accurate object from one photograph. The
`POST /api/v2/jobs/single-image` route returns `409 single_image_provider_unavailable` because
no local single-image provider is configured. Use multiple overlapping views or an orbit video.

### Input contract

| Input | API contract | Practical capture target |
| --- | --- | --- |
| Photos | `POST /api/v2/jobs/photos`; 3–100 JPEG, PNG, or WebP files; 25 MiB per image and 500 MiB total | 20–50 sharp, substantially overlapping views at multiple elevations |
| Video | `POST /api/v2/jobs/video`; one AVI, M4V, MKV, MOV, MP4, or WebM up to 2 GiB | One slow, steady orbit; default duration limit 300 seconds and default target 40 useful frames |
| Single image | Explicitly unavailable in the v3 scan lane | Capture more views; CadPro will not fabricate hidden geometry |

The video API accepts a target from 8–200 selected frames. Its duration parameter defaults to
300 seconds and has a hard schema ceiling of 3,600 seconds; the public capability response and
normal website workflow advertise 300 seconds. FFmpeg first extracts at most 600 candidates at
3 candidates/second by default, then CadPro applies blur, spacing, similarity, and viewpoint
change checks. Fewer than eight useful frames is a failed capture, not a synthetic success.

## Mesh is not CAD

CadPro keeps these representations distinct:

| Representation | Meaning | Typical artifact |
| --- | --- | --- |
| Sparse point cloud | Feature tracks used to solve camera poses | `cadpro-sparse-cloud.ply` |
| Dense point cloud | Multi-view stereo samples | `cadpro-dense-cloud.ply` |
| Triangle mesh | Reconstructed surface approximation | cleaned PLY and OBJ |
| Visualization model | Mesh packaged for a viewer, textured only when texture data survives validation | GLB |
| Printable mesh | Reopened watertight manifold mesh | STL, conditionally |
| Analytic CAD/B-rep | A compact fitted box or cylinder with valid topology and metric dimensions | STEP, conditionally |

OBJ, GLB, STL, and PLY are not parametric CAD. Converting triangles to a faceted STEP shell or
renaming a mesh would not recover planes, holes, sketches, constraints, tolerances, or design
history. CadPro therefore publishes STEP only after scale, analytic-fit, topology, volume,
dimension, and reopen checks pass. Organic objects and most manufactured parts with several
features should be remodeled from the mesh in a CAD system.

## Quick start

Python 3.11 or newer is required. These commands install the Python application and its
OpenCascade binding; the native reconstruction programs are separate installations.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"

.venv\Scripts\cadpro.exe scan-doctor
.venv\Scripts\cadpro.exe web --no-open
```

### Linux or macOS

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/cadpro scan-doctor
.venv/bin/cadpro web --no-open
```

Open `http://127.0.0.1:8000`. The FastAPI server, static website, persistent SQLite scan queue,
and one bounded worker start together; there is no separate frontend or worker command.
`cadpro web` opens the browser unless `--no-open` is supplied. Keep the default loopback host
unless you have added authentication and a trusted reverse proxy—the application itself does
not provide multi-user authentication.

## Native dependency setup

The adapter was audited against these upstream versions and command help/source contracts:

| Component | Reference version | Purpose | Official source |
| --- | --- | --- | --- |
| COLMAP | 4.1.1, released July 17, 2026 | features, matching, camera/SfM solution, and optional CUDA dense reconstruction | [4.1.1 release](https://github.com/colmap/colmap/releases/tag/4.1.1), [4.1.1 CLI guide](https://github.com/colmap/colmap/blob/4.1.1/doc/cli.rst) |
| OpenMVS | 2.4.0, released January 20, 2026 | CPU-capable dense reconstruction, mesh refinement, and camera-derived texture export | [2.4.0 release](https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0), [usage guide](https://github.com/cdcseacave/openMVS/wiki/Usage) |
| FFmpeg/FFprobe | 9.0.1, released August 12, 2026 | video metadata and bounded frame extraction | [download/source page](https://ffmpeg.org/download.html), [FFprobe](https://ffmpeg.org/ffprobe.html), [filters](https://ffmpeg.org/ffmpeg-filters.html#select_002c-aselect) |
| cadquery-ocp-novtk | 7.9.3.1.1 package line, OCP runtime 7.9.3.1 on this host | STEP construction and reopen validation | [package record](https://pypi.org/project/cadquery-ocp-novtk/) |

`scan-doctor` is the source of truth for a particular machine. A version number alone does not
prove that a build contains the required commands, CUDA support, codecs, or compatible DLLs.

### Windows

1. Download one official COLMAP 4.1.1 archive:

   - `colmap-x64-windows-cuda.zip` — SHA-256
     `b06064e7e4bd34f5b4ef71b442d3537d95d57c666dbec5a3b475902ccd832b9b`;
   - `colmap-x64-windows-nocuda.zip` — SHA-256
     `faf1247d2ec90933aa8bd003709790abf0211cdc132cceec4c831718f2e0895a`.

   The no-CUDA build is sufficient for sparse reconstruction when OpenMVS supplies the dense
   stage. COLMAP's `patch_match_stereo` path requires a compatible CUDA build and NVIDIA GPU.
   Upstream documents `COLMAP.bat -h` for its release layout. CadPro launches argument arrays
   without a shell; in the standard archive it resolves the wrapper to `bin\colmap.exe`. Keep
   the archive's required DLL directories discoverable by Windows.

2. Download one official OpenMVS 2.4.0 archive:

   - `OpenMVS_Windows_x64.zip` — SHA-256
     `0c31660c15c9ebc4c106873cf67564d9570d404aef7a6403451da1b6178b2167`;
   - `OpenMVS_Windows_x64_CUDA.7z` — SHA-256
     `6aac6b14ef478e501d2514cd1d74ed20e659b53b3bc2835d45a473ddfb921621`.

   CadPro needs `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, and `RefineMesh` for
   the OpenMVS geometry route. `TextureMesh` is additionally required for camera-derived
   textures. Review the release's third-party notices before deployment.

3. Obtain FFmpeg 9.0.1. FFmpeg.org publishes source and links to external Windows binary
   providers; FFmpeg.org does not itself publish the Windows executable build. Record the
   provider and configuration you choose, and verify both `ffmpeg.exe` and `ffprobe.exe`.

4. Verify downloaded official release archives before extracting:

   ```powershell
   Get-FileHash -Algorithm SHA256 -LiteralPath C:\Downloads\colmap-x64-windows-nocuda.zip
   Get-FileHash -Algorithm SHA256 -LiteralPath C:\Downloads\OpenMVS_Windows_x64.zip
   ```

5. Either put the executables on the service account's `PATH`, or set explicit absolute paths.
   Replace these examples with the actual extracted locations:

   ```powershell
   $env:CADPRO_COLMAP_PATH = "C:\Tools\COLMAP-4.1.1\bin\colmap.exe"
   $env:CADPRO_FFMPEG_PATH = "C:\Tools\FFmpeg-9.0.1\bin\ffmpeg.exe"
   $env:CADPRO_FFPROBE_PATH = "C:\Tools\FFmpeg-9.0.1\bin\ffprobe.exe"
   $env:CADPRO_OPENMVS_INTERFACE_PATH = "C:\Tools\OpenMVS-2.4.0\bin\InterfaceCOLMAP.exe"
   $env:CADPRO_OPENMVS_DENSIFY_PATH = "C:\Tools\OpenMVS-2.4.0\bin\DensifyPointCloud.exe"
   $env:CADPRO_OPENMVS_RECONSTRUCT_PATH = "C:\Tools\OpenMVS-2.4.0\bin\ReconstructMesh.exe"
   $env:CADPRO_OPENMVS_REFINE_PATH = "C:\Tools\OpenMVS-2.4.0\bin\RefineMesh.exe"
   $env:CADPRO_OPENMVS_TEXTURE_PATH = "C:\Tools\OpenMVS-2.4.0\bin\TextureMesh.exe"

   .venv\Scripts\cadpro.exe scan-doctor
   ```

   Environment variables must name files, not directories. If a direct COLMAP executable
   starts only from its release folder, fix the release DLL search configuration before
   launching CadPro.

### Linux

Use the official [COLMAP build instructions](https://colmap.github.io/install.html) to build or
install tag 4.1.1. A distribution package may be older, so confirm the command help rather than
assuming its version. For OpenMVS, use the official `OpenMVS_Ubuntu_x64.zip` 2.4.0 asset
(SHA-256 `7104ae1ddd6ca38fbca9e0e4a70b20af59e21e0b497eb7181c864fbf38ca8d00`)
or build tag 2.4.0 with the official [building guide](https://github.com/cdcseacave/openMVS/wiki/Building).
Build/install FFmpeg 9.0.1 from its signed source release or use a distribution build whose
configuration you have checked.

Set the same `CADPRO_*_PATH` variables to absolute executable files when the programs are not
on `PATH`:

```bash
export CADPRO_COLMAP_PATH=/opt/colmap-4.1.1/bin/colmap
export CADPRO_FFMPEG_PATH=/opt/ffmpeg-9.0.1/bin/ffmpeg
export CADPRO_FFPROBE_PATH=/opt/ffmpeg-9.0.1/bin/ffprobe
export CADPRO_OPENMVS_INTERFACE_PATH=/opt/openmvs-2.4.0/bin/InterfaceCOLMAP
export CADPRO_OPENMVS_DENSIFY_PATH=/opt/openmvs-2.4.0/bin/DensifyPointCloud
export CADPRO_OPENMVS_RECONSTRUCT_PATH=/opt/openmvs-2.4.0/bin/ReconstructMesh
export CADPRO_OPENMVS_REFINE_PATH=/opt/openmvs-2.4.0/bin/RefineMesh
export CADPRO_OPENMVS_TEXTURE_PATH=/opt/openmvs-2.4.0/bin/TextureMesh
.venv/bin/cadpro scan-doctor
```

### macOS

Build/install COLMAP 4.1.1 according to the official
[macOS instructions](https://colmap.github.io/install.html#macos). OpenMVS 2.4.0 publishes
`OpenMVS_macOS_arm64.zip` with SHA-256
`3d4c616c97031b1ab6e2eecb0ddd5614fb99513c0782a32c9350602faf38799b`; Intel machines need a
compatible source build. Obtain FFmpeg 9.0.1 from signed source or a binary provider linked by
FFmpeg.org. Then use the Unix environment-variable pattern above and run `scan-doctor`.

### Verify the native programs

Use the programs' own help/version output before launching a real job:

```text
ffmpeg -version
ffmpeg -L
ffprobe -version
colmap -h
InterfaceCOLMAP --help
DensifyPointCloud --help
ReconstructMesh --help
RefineMesh --help
TextureMesh --help
```

There is no documented `colmap version` subcommand in the 4.1.1 CLI contract; CadPro probes
`colmap -h`. Some OpenMVS programs print help and return nonzero, so `scan-doctor` accepts a
nonempty help response as evidence that the executable started.

### CPU, GPU, memory, and disk expectations

- COLMAP is always required for sparse camera reconstruction. `use_gpu=false` disables GPU use
  for feature extraction and matching.
- When the four core OpenMVS executables are available, CadPro uses the OpenMVS dense/mesh route.
  This is the practical CPU route; OpenMVS build options may add their own acceleration.
- Without a complete OpenMVS route, CadPro requires `use_gpu=true` and attempts COLMAP
  PatchMatch, geometric fusion, and Poisson or Delaunay meshing. A no-CUDA COLMAP build will fail
  this branch with an actionable error.
- Texture projection currently comes only from OpenMVS `TextureMesh`. The COLMAP-only branch
  publishes untextured GLB/OBJ and labels them accordingly.
- Resource use depends strongly on input resolution, accepted view count, preset, scene detail,
  and native build. A reasonable small-object starting point is a modern 4+ core CPU, 16 GiB RAM,
  and 20–50 GiB free workspace. High-resolution/high-preset captures can need 32 GiB or more RAM
  and well over 100 GiB of temporary disk. The COLMAP dense route needs a supported NVIDIA CUDA
  GPU with enough VRAM for the chosen image size. These are planning estimates, not enforced
  minimums or benchmarks from this machine.

Quality presets change real native image-size limits:

| Preset | COLMAP feature maximum edge | Dense undistortion maximum edge |
| --- | ---: | ---: |
| Draft | 1,600 px | 1,200 px |
| Balanced | 2,400 px | 2,000 px |
| High | 3,200 px | 3,200 px |

## Capture guide

Reconstruction quality depends heavily on overlap, lighting, surface texture, reflections,
transparency, motion blur, camera calibration, and complete viewpoint coverage.

For photos:

1. Keep the object still and move the camera around it, or rotate it while keeping a trackable
   background visible. Do not change the object between shots.
2. Capture 20–50 original photos with roughly 60–80% overlap. Include a full waist-level orbit
   plus higher and lower angles; keep the whole object in every frame.
3. Use soft, even lighting and stable focus, exposure, focal length, and zoom. Avoid edited,
   resized, filtered, or heavily compressed inputs.
4. Add non-repeating visual texture around matte, featureless objects. Glossy, transparent,
   translucent, very thin, deforming, or repetitive surfaces are poor photogrammetry targets.
5. Keep hands, turntable supports, and moving shadows from covering important surfaces.
6. When dimensions matter, include geometry on which two reconstructed points and their real
   separation can later be identified precisely. CadPro does not silently infer units.

For video, record one slow and steady 360-degree orbit with high and low coverage if possible,
aiming for 20–50 useful selected views. Avoid autofocus pulsing, digital zoom, motion blur,
pauses, rapid pans, and a featureless background. CadPro selects useful frames rather than
sending hundreds of near-duplicates to COLMAP; review the generated contact sheet.

## Website and `/api/v2`

Run `cadpro scan-doctor` first, then start `cadpro web --no-open`. The website exposes capture,
settings, progress, cancellation, result warnings, preview, calibration, and artifact downloads.
The standard lane makes no external network request.

### Capabilities

```bash
curl http://127.0.0.1:8000/api/v2/capabilities
```

This response reports individual native tools and derived photo, video, dense, texturing, mesh,
and analytic-CAD capabilities. A missing tool required by the selected ingest mode or CPU route
causes an actionable HTTP error before upload; a CUDA/runtime mismatch can still fail later when
the native COLMAP dense stage starts.

### Submit photos

Repeat `files=@...` for every view:

```bash
curl -i -X POST http://127.0.0.1:8000/api/v2/jobs/photos \
  -F 'files=@capture/001.jpg' \
  -F 'files=@capture/002.jpg' \
  -F 'files=@capture/003.jpg' \
  -F 'quality_preset=balanced' \
  -F 'feature_matcher=exhaustive' \
  -F 'mesher=poisson' \
  -F 'use_gpu=false' \
  -F 'generate_cad=true'
```

Three is only the API minimum; 20–50 is the recommended capture. Use `sequential` matching only
when input order really follows the orbit. With no complete OpenMVS installation, `use_gpu=false`
cannot run the dense stage.

### Submit a video

```bash
curl -i -X POST http://127.0.0.1:8000/api/v2/jobs/video \
  -F 'file=@object-orbit.mp4' \
  -F 'quality_preset=balanced' \
  -F 'target_frames=40' \
  -F 'maximum_duration_seconds=300' \
  -F 'feature_matcher=sequential' \
  -F 'mesher=poisson' \
  -F 'use_gpu=false' \
  -F 'generate_cad=true'
```

A successful submission returns `202`, a job snapshot, `Location` pointing at the status URL,
and `Retry-After: 1`.

### Poll, cancel, and download

```bash
curl http://127.0.0.1:8000/api/v2/jobs/JOB_ID
curl -X POST http://127.0.0.1:8000/api/v2/jobs/JOB_ID/cancel
curl -OJ http://127.0.0.1:8000/api/v2/jobs/JOB_ID/artifacts/visual-glb
curl -OJ http://127.0.0.1:8000/api/v2/jobs/JOB_ID/artifacts/complete-bundle
```

Use artifact IDs returned in the completed job; not every conditional artifact exists. Download
responses are private and `no-store`. Queued cancellation is immediate. Active cancellation sets
a cancellation token and terminates, then kills if necessary, the current native child process.

### Two-point scale and a new revision

An uncalibrated result remains in arbitrary reconstruction units and carries the warning
“Scale is unknown; do not use dimensions for manufacturing.” It can have mesh artifacts, but it
cannot have a STEP artifact.

After a job completes, select two distinct points in reconstructed coordinates and enter their
measured separation:

```bash
curl -i -X POST http://127.0.0.1:8000/api/v2/jobs/JOB_ID/calibration \
  -H 'Content-Type: application/json' \
  -d '{
    "point_a": [0.12, -0.44, 1.08],
    "point_b": [0.12, 0.56, 1.08],
    "real_distance": 100.0,
    "unit": "mm",
    "selection_uncertainty": 0.01
  }'
```

Units are `mm`, `cm`, `m`, or `in`. Scale factor is `real_distance / reconstructed_distance`;
the reported uncertainty is the point-selection uncertainty multiplied by that factor. The
endpoint returns a **new job ID**, records `calibration_revision_of`, and hard-links or copies
the source inputs plus the completed job's immutable sparse cloud, dense cloud, mesh, and
validated texture bundle when present. The revision reruns the downstream scale, mesh-export,
CAD-fit, and artifact-validation stages; it does **not** rerun native camera estimation or
multi-view stereo. Its report labels `pipeline_adapter` as `immutable-artifact-reuse` and records
the source job ID. It does not mutate the original job or silently relabel its units. A source
that is already calibrated, expired, missing artifacts, or has an invalid report cannot be used
as the calibration source. Accuracy still depends on reconstruction and point placement.

## Audited native command contract

CadPro does not concatenate upload data into shell command strings. It launches bounded argument
arrays with isolated working directories, logs, timeouts, and cancellation. Paths below are
placeholders; the option names and order match the current adapter.

### FFprobe and FFmpeg 9.0.1

```text
ffprobe -v error -select_streams v:0 \
  -show_entries format=format_name,duration,size:stream=index,codec_name,codec_type,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,duration \
  -of json <video>

ffmpeg -hide_banner -nostdin -v error -i <video> -map 0:v:0 \
  -vf "select=isnan(prev_selected_t)+gte(t-prev_selected_t\,<spacing>),scale=<maximum-edge>:-2:force_original_aspect_ratio=decrease" \
  -fps_mode vfr -frames:v <maximum-candidate-frames> -q:v 2 -n <candidate-%06d.jpg>
```

The use of current `-fps_mode vfr`, bounded `-frames:v`, and no-overwrite `-n` is deliberate.

### COLMAP 4.1.1

Before a job, CadPro probes every required subcommand with `colmap <command> -h`. The staged
commands are:

```text
colmap feature_extractor --database_path <database.db> --image_path <images> \
  --FeatureExtraction.max_image_size <preset-size> --FeatureExtraction.use_gpu <0|1>

colmap <exhaustive_matcher|sequential_matcher> --database_path <database.db> \
  --FeatureMatching.use_gpu <0|1>

colmap mapper --database_path <database.db> --image_path <images> --output_path <sparse>

colmap model_analyzer --path <each-sparse-component>

colmap model_converter --input_path <selected-component> --output_path <sparse.ply> \
  --output_type PLY

colmap image_undistorter --image_path <images> --input_path <selected-component> \
  --output_path <dense> --output_type COLMAP --max_image_size <preset-size>
```

Every directory under the sparse output is analyzed. CadPro selects the component with the most
registered images, then the most points, and requires at least
`max(3, ceil(accepted_images * 0.5))` registered cameras. It never assumes `sparse/0`.

If OpenMVS is unavailable and GPU dense processing was requested, the remaining COLMAP commands
are:

```text
colmap patch_match_stereo --workspace_path <dense> --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion --workspace_path <dense> --workspace_format COLMAP \
  --input_type geometric --output_path <dense/fused.ply>

colmap poisson_mesher --input_path <dense/fused.ply> --output_path <meshed-poisson.ply>
```

For Delaunay, the final line is:

```text
colmap delaunay_mesher --input_path <dense> --output_path <meshed-delaunay.ply>
```

The paired geometric-consistency and geometric-fusion options must remain aligned.

### OpenMVS 2.4.0

With all four core OpenMVS programs available, CadPro uses this sequence:

```text
InterfaceCOLMAP -i <absolute-colmap-dense> -o <scene.mvs> \
  --image-folder <absolute-colmap-dense/images>

DensifyPointCloud <scene.mvs> -o <scene_dense.mvs>

ReconstructMesh <scene_dense.mvs> -p <scene_dense.ply> -o <scene_dense_mesh.mvs>

RefineMesh <scene_dense.mvs> -m <scene_dense_mesh.ply> \
  -o <scene_dense_mesh_refine.mvs> --scales 1 --max-face-area 16

TextureMesh <scene_dense.mvs> -m <scene_dense_mesh_refine.ply> \
  -o <native-textured/model_glb.mvs> --export-type glb

TextureMesh <scene_dense.mvs> -m <scene_dense_mesh_refine.ply> \
  -o <native-textured/model_obj.mvs> --export-type obj
```

Absolute InterfaceCOLMAP input and image-folder paths avoid 2.4.0's relative image-folder
resolution trap. CadPro expects exactly one GLB and one OBJ when texturing is requested. The live
OpenMVS wiki may describe newer `TransformScene --convert` behavior; that command is **not** part
of CadPro's audited OpenMVS 2.4.0 sequence.

## Artifacts and gates

A completed native job can advertise only artifacts that exist, are nonempty, and pass their
format-specific reopen checks:

| Artifact ID | File/meaning | Publication gate |
| --- | --- | --- |
| `sparse-ply` | sparse camera/SfM points | native sparse model converted and reopened as finite PLY points |
| `dense-ply` | dense MVS points | native dense cloud exists and reopens with finite points |
| `mesh-ply` | cleaned triangle mesh | finite vertices/normals, valid indices, nonempty faces |
| `visual-glb` | glTF 2.0 viewing mesh | valid GLB header/length and renderable mesh nodes; texture claim requires embedded linked UV/material/image data |
| `mesh-obj` | exchange mesh | at least three vertices and a face; safe material references |
| `texture-*` | MTL/image resources | only resources actually emitted and safely named |
| `printable-stl` | watertight print mesh | zero boundary and non-manifold edges before export, then watertight after reopen |
| `contact-sheet` | selected video views | video mode only |
| `preview` | self-contained HTML viewer | source GLB validated first |
| `fitted-step` | metric analytic box/cylinder B-rep | CAD requested, two-point scale known, fit accepted, valid positive-volume topology, one solid after reopen, matching volume and bounds |
| `cad-script` | compact editable Python construction | emitted only with a validated fitted STEP |
| `native-log-*` | bounded native diagnostics | up to 20 nonempty logs |
| `report` | schema-validated reconstruction JSON | numerical metrics, settings, warnings, timings, tool versions, scale and fit details |
| `manifest` | reproducibility JSON | input SHA-256 values, exact command arrays, configuration, versions, artifact metadata |
| `complete-bundle` | ZIP of published outputs | safe unique basenames, nonempty entries, ZIP integrity check |

Mesh cleanup removes degenerate/duplicate faces, unreferenced vertices, and only small
disconnected fragments; it fills only simple triangle/quad holes. It does not force a fake
watertight result. A missing STL or STEP is therefore a truthful conditional outcome, not an
export bug by itself.

Completed reports classify quality with visible underlying metrics:

- **excellent:** at least 20 accepted images, at least 90% camera registration, no more than
  0.8 px mean reprojection error when reported, at least 500,000 dense points, and one mesh
  component;
- **usable:** at least 12 accepted images, at least 70% registration, no more than 1.5 px
  reprojection error when reported, and at least 100,000 dense points;
- **weak:** every other completed result.

A process error is a failed job rather than a completed result hidden behind a weak label.

## Storage, queue, cancellation, and restart behavior

`cadpro web` sets `CADPRO_STORAGE_DIR` when it is not already configured:

- Windows: `%LOCALAPPDATA%\CadPro`;
- Linux/macOS: `$XDG_DATA_HOME/cadpro`, or `~/.local/share/cadpro` when `XDG_DATA_HOME` is unset.

The v3 scan store lives in the internal `scan-v2` subdirectory because its name follows the API
schema version. Each UUID job has isolated inputs, work files, artifacts, and SQLite metadata.
The queue is disk-backed with one worker and accepts four queued scan jobs by default. Uploads
are streamed in chunks rather than loaded as one multipart body, and video candidate frames are
bounded and removed after final selection.

On restart, queued scan jobs are re-enqueued. A job that was `running` when the process stopped
is marked failed with `worker_interrupted` and must be submitted again; CadPro does not pretend
to resume halfway through an external native stage.

Completed, failed, and cancelled `/api/v2` jobs have a default 24-hour retention period. A safe
sweeper runs every 60 seconds and deletes expired SQLite records and their validated UUID job
directories; queued or running jobs are not expired. Embedded deployments can set
`create_app(job_retention_seconds=..., job_sweep_interval_seconds=...)`; a retention value of `0`
disables automatic terminal-job cleanup. The `cadpro web` command does not currently expose
those values as CLI options. Download or back up valuable artifacts and create a calibrated
revision before its unscaled source expires. Do not delete a job directory while CadPro is
running.

The older `/api/jobs/*` manager has its own sweeper and the same default 24-hour terminal-job
TTL; legacy and `/api/v2` job records remain separate.

## Architecture

```text
Browser / API client
        |
        v
FastAPI website + /api/v2 validation
        |
        v
SQLite JobStore -> bounded persistent worker -> isolated UUID workspace
        |                                      |
        |                                      +-> FFprobe / FFmpeg (video)
        |                                      +-> image quality gates
        |                                      +-> COLMAP sparse SfM
        |                                      +-> OpenMVS dense/mesh/texture
        |                                          or COLMAP CUDA dense/mesh
        |                                      +-> trimesh repair and mesh exports
        |                                      +-> two-point scale
        |                                      +-> box/cylinder fit + OCP STEP gate
        v
validated artifacts + report + manifest + ZIP
```

Relevant modules:

```text
src/cadpro/scan/api.py              /api/v2 validation, upload, status, cancel, calibration
src/cadpro/scan/jobs.py             SQLite state machine, workspaces, restart, bounded worker
src/cadpro/scan/capabilities.py     executable/Python probes and install hints
src/cadpro/scan/video.py            FFprobe, FFmpeg, frame quality/selection, contact sheet
src/cadpro/scan/quality.py          EXIF orientation, normalization, blur/exposure/features
src/cadpro/scan/photogrammetry.py   staged COLMAP/OpenMVS adapter and test-only synthetic adapter
src/cadpro/scan/mesh.py             finite geometry checks, conservative repair, GLB/OBJ/STL/PLY
src/cadpro/scan/scale.py            explicit two-point scale and uncertainty
src/cadpro/scan/cad_fit.py          robust primitive fits and metric analytic STEP export
src/cadpro/scan/artifacts.py        reopen validation, hashes, report, manifest, preview, ZIP
src/cadpro/scan/pipeline.py         state-machine orchestration and quality classification
src/cadpro/web_assets/              bundled responsive website
```

## Docker limitation

The included `Dockerfile` builds the Python 3.12 web/API shell, runs as a non-root user, stores
data at `/var/lib/cadpro/jobs`, and checks `/api/health`. It intentionally does **not** install
FFmpeg, FFprobe, COLMAP, OpenMVS, their runtime libraries, or CUDA/GPU wiring. Consequently the
stock image can serve the site and capability errors but cannot perform a real photo/video scan.

For native reconstruction, build and review a derived image that installs the pinned tools,
persists `/var/lib/cadpro/jobs`, and—in the COLMAP dense case—configures the matching NVIDIA
container runtime. Run `cadpro scan-doctor` inside that final image. Mounting host executables
without their matching libraries is not a reliable installation.

## Privacy and security

- The standard `/api/v2` pipeline runs locally and has no paid-cloud dependency or telemetry.
- Legacy optional AI providers are disabled unless an administrator configures them; see below.
- Upload types, signatures, sizes, counts, dimensions, and duration are checked. Generated names
  and ZIP entries are constrained, artifact lookup uses opaque IDs, and path traversal is
  rejected.
- Native programs receive argument arrays with `shell=False`, bounded captured output, timeouts,
  cancellation, and per-job working directories.
- Capture files, reports, logs, and models may contain proprietary geometry. Protect the storage
  directory, keep the server on loopback by default, and add authentication/TLS before exposing
  it to a network.
- Diagnostic logs are sanitized and bounded, but administrators should still review them before
  sharing a complete bundle.

## Troubleshooting

### “Dependency unavailable” before upload

Run `cadpro scan-doctor --json`. Fix every required executable path and restart the server;
capability detection is cached at application startup. Photos need COLMAP. Video also needs
FFmpeg and FFprobe. Dense reconstruction needs the complete OpenMVS core or a working CUDA
COLMAP dense build.

### COLMAP starts manually but CadPro cannot start it

`CADPRO_COLMAP_PATH` must identify a file. For the official Windows release, use its standard
wrapper layout or point to `bin\colmap.exe` and ensure the archive's DLL directories are
discoverable. CadPro does not run arbitrary `.bat` command strings.

### CPU job says dense reconstruction is unavailable

Install all four core OpenMVS executables and set their individual paths. Otherwise use a
verified CUDA COLMAP build and submit with `use_gpu=true`; merely changing the flag cannot add
CUDA support to a no-CUDA binary.

### Few cameras register or several sparse components appear

Add overlap and trackable detail, include intermediate angles, reduce blur/reflections, keep
focus and zoom fixed, and avoid changing the scene. CadPro keeps the strongest connected
component and reports discarded components; it will stop if fewer than half the accepted views
register.

### Video produces fewer than eight useful views

Record a slower, sharper orbit with more viewpoint change and less repetition. The selector is
designed to reject near-duplicate or blurry frames rather than inflate the view count.

### GLB/OBJ is untextured

Install a complete OpenMVS build including `TextureMesh`. CadPro also withholds the textured
label if the exported model cannot prove linked texture data. The COLMAP-only branch is geometry
only.

### STL is missing

The repaired mesh was not demonstrably watertight and manifold after reopen. Inspect the report's
boundary edges, non-manifold edges, and component count, then repair deliberately in a mesh tool.

### STEP is missing

Confirm that CAD generation was enabled and scale was calibrated. STEP is still skipped unless
the surface passes the supported axis-aligned box or right-cylinder fit and all OCP validation
checks. Use the GLB/OBJ/PLY as reverse-engineering reference for more complex parts.

### A job failed after restart

If it was running when the process exited, this is intentional `worker_interrupted` recovery.
Submit the capture again. Queued jobs are restored, but native programs are not checkpointed
mid-command.

### Calibration says the source artifacts are unavailable

Calibration is allowed only from a completed, unscaled job with an intact schema-valid report
and immutable sparse/dense/mesh artifacts. The default retention sweeper removes terminal jobs
after 24 hours. Calibrate before that deadline, or retain the original capture and submit a new
native job. A calibrated revision reuses the source reconstruction artifacts and does not spend
time rerunning COLMAP/OpenMVS.

## Development and verification

Run all three checks from an installed development environment.

Windows:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m mypy src/cadpro/scan
```

Linux/macOS:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src/cadpro/scan
```

The fast API end-to-end test injects `SyntheticTestAdapter` explicitly and verifies job creation,
processing, report/artifact validation, downloads, calibration, cancellation, and missing-tool
behavior without running native photogrammetry. Native command tests mock process boundaries and
assert safe exact argv arrays. These tests prove the surrounding application and adapter
contract; they do not prove that a real capture reconstructs on a particular machine.

`tests/integration/test_real_scan_api.py` is the opt-in real-capture test. It accepts direct-child
JPEG, PNG, or WebP files from a private local directory (at least 3, with 20–50 recommended),
runs the real `/api/v2` adapter, polls for up to 30 minutes, downloads/reopens the advertised
artifacts, and checks the report, hashes, manifest command provenance, and ZIP. It skips during a
normal test run unless the dataset and native capabilities are available.

Windows:

```powershell
$env:CADPRO_REAL_SCAN_DATASET = "C:\absolute\path\to\rights-cleared-photos"
.venv\Scripts\python.exe -m pytest -m integration tests/integration/test_real_scan_api.py
```

Linux/macOS:

```bash
CADPRO_REAL_SCAN_DATASET=/absolute/path/to/rights-cleared-photos \
  .venv/bin/python -m pytest -m integration tests/integration/test_real_scan_api.py
```

No dataset was configured and this integration test was skipped on the Windows host described
at the top of this README. After a real run, inspect camera registration and the models
independently, calibrate two points through the website/API, and reopen any conditional STEP in
Onshape, FreeCAD, or another CAD system. Record the exact report and manifest rather than
claiming native success from mocked tests alone.

## Dependency licenses

Review the exact binaries and build options you distribute; this summary is not legal advice.

- COLMAP itself uses the [new BSD license](https://github.com/colmap/colmap/blob/4.1.1/COPYING.txt),
  while its dependencies have separate terms that can affect a binary distribution.
- OpenMVS is [AGPL-licensed and includes third-party notices](https://github.com/cdcseacave/openMVS/blob/v2.4.0/COPYRIGHT.md).
  Its 2.4.0 notice lists, among other components, an `ibfs` research-purpose restriction. Review
  the complete archive and obtain legal guidance before operating or distributing it in a hosted
  or commercial service.
- FFmpeg is normally LGPL 2.1-or-later, but enabling GPL components makes the whole FFmpeg build
  GPL; inspect `ffmpeg -L` and `ffmpeg -buildconf` and follow the
  [official legal guidance](https://ffmpeg.org/legal.html).
- The OCP binding carries the
  [Open CASCADE LGPL 2.1 exception terms](https://github.com/CadQuery/OCP/blob/7.9.3.1/LICENSE).
- This repository currently has no top-level `LICENSE` file. Do not assume permission to
  redistribute CadPro itself until the maintainers add an explicit project license.

## Legacy v2 tools and providers

The repository still contains the older measured-silhouette and optional generative workflows
for compatibility. They are **not** the `/api/v2` photogrammetry lane, are not a fallback when
native tools are missing, and the CadPro 3.0 website must not be assumed to expose all of them.
Inspect `cadpro --help` and the legacy API source before integrating them.

- `cadpro convert` builds a measured 2.5D profile extrusion from one image or one selected video
  frame using explicit width and depth.
- `cadpro turntable` builds an ordered silhouette visual hull from 4–24 video views and an
  explicit maximum width. It is silhouette geometry, not texture-based photogrammetry.
- `cadpro neural-train` and `cadpro neural-predict` train/load a data-only NPZ depth-ratio model.
  Prediction still needs a measured width and remains an estimate of hidden depth.
- Legacy HTTP handlers remain under `/api/jobs/image`, `/api/jobs/photos`, `/api/jobs/video`, and
  `/api/jobs/text`; they use a separate in-memory worker/retention model and should not be mixed
  with `/api/v2/jobs/*` IDs.
- The optional Meshy provider is enabled only with `CADPRO_MESHY_ENABLED=1` and a server-side
  `MESHY_API_KEY`. It can submit text, one image, or at most four selected representative views
  and publish non-metric visual mesh assets. Meshy output is never STEP. Review Meshy's live
  [authentication](https://docs.meshy.ai/en/api/authentication),
  [image-to-3D](https://docs.meshy.ai/en/api/image-to-3d),
  [multi-image](https://docs.meshy.ai/en/api/multi-image-to-3d),
  [pricing](https://docs.meshy.ai/en/api/pricing), and
  [terms](https://www.meshy.ai/terms-of-use) before sending any capture or spending credits.
- `src/cadpro/ml_mesh.py` retains an administrator-operated Hunyuan-compatible concept-mesh
  seam. It is disabled by default, externally hosted, non-metric, and never supplies STEP
  geometry. Review the exact worker revision and its license before enabling it.

Legacy measured silhouette/neural STEP files and v3 fitted STEP files have different evidence
and assumptions. Their reports must remain attached so a downstream user can tell how each
model was produced.

## Known limitations

- Photogrammetry reconstructs visible surfaces, not hidden cavities, threads, tolerances,
  material, manufacturing intent, or a native parametric feature tree.
- Scale from one selected point pair corrects global size; it does not remove local warp or lens,
  camera, matching, and surface errors.
- Current analytic export recognizes only a simple axis-aligned box or right cylinder. Planes are
  useful diagnostics but are not independently exported as solids.
- Texture generation depends on OpenMVS and suitable photos. A valid untextured mesh can still be
  a successful geometric result.
- Mesh repair is conservative. Complex holes and non-manifold regions are reported, not silently
  filled into invented surfaces.
- One photograph cannot support the standard v3 scan. Legacy or cloud generative approximations
  are separate, inferred, and non-equivalent workflows.
- Native-tool compatibility has been checked against upstream 4.1.1/2.4.0/9.0.1 contracts, but
  not executed on the current Windows host.

Never manufacture a safety-critical part directly from a reconstruction. Verify overall size,
interfaces, holes, wall thickness, hidden features, tolerances, and material with independent
measurement and qualified engineering review.

## Contributors

- Preston L
- Ethan C (`yil91974@gmail.com`)

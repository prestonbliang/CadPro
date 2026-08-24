"use strict";

const MEBIBYTE = 1024 * 1024;
const GIBIBYTE = 1024 * MEBIBYTE;
const MAX_IMAGE_BYTES = 25 * MEBIBYTE;
const MAX_PHOTO_SET_BYTES = 500 * MEBIBYTE;
const MAX_VIDEO_BYTES = 2 * GIBIBYTE;
const THUMBNAIL_MAX_EDGE = 240;
const MAX_IMAGE_EDGE = 20_000;
const MAX_IMAGE_PIXELS = 40_000_000;
const MAX_PHOTOS = 100;
const MIN_PHOTOS = 3;
const RECOMMENDED_PHOTOS = 20;
const STORAGE_KEY = "cadpro.scan.active-job.v2";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  mode: "photos",
  files: [],
  thumbnailGeneration: 0,
  capabilities: null,
  jobId: null,
  statusUrl: null,
  pollTimer: null,
  submitting: false,
  pickedPoints: [],
};

const modeCopy = {
  photos: {
    title: "Add overlapping object photos",
    drop: "Drop 20–50 photos here",
    copy: "PNG, JPEG, or WebP · 25 MiB each · 100 maximum",
    browse: "CHOOSE PHOTOS",
    counter: "0 / 20 recommended",
    warning: "Keep the whole object visible and move around it with 60–80% overlap. Include high and low angles; do not resize or edit the originals.",
  },
  video: {
    title: "Add one continuous orbit video",
    drop: "Drop one orbit video here",
    copy: "MP4, MOV, MKV, WebM, or AVI · 2 GiB maximum · 5 minutes default",
    browse: "CHOOSE VIDEO",
    counter: "0 / 1 video",
    warning: "Move slowly around the object once. Keep focus and zoom stable; CadPro removes blurred and nearly identical candidate frames before reconstruction.",
  },
  image: {
    title: "Single-photo experimental approximation",
    drop: "Drop one reference photo here",
    copy: "No local provider is configured; multiple views are required",
    browse: "CHOOSE PHOTO",
    counter: "0 / 1 image",
    warning: "A single image cannot measure hidden surfaces. CadPro will not fabricate a successful scan or STEP file when no explicit reconstruction provider is configured.",
  },
};

const stageLabels = {
  queued: ["Queued for the local worker", "The persistent job is waiting for the bounded worker."],
  validating: ["Validating untrusted inputs", "Checking file metadata, limits, signatures, and isolated paths."],
  extracting_frames: ["Selecting useful video frames", "FFprobe inspected the stream; FFmpeg candidates are ranked by blur, similarity, spacing, and viewpoint change."],
  analyzing_images: ["Analyzing capture quality", "Correcting orientation and rejecting blur, duplicates, and unusable exposures."],
  estimating_cameras: ["Estimating cameras", "COLMAP is matching visual features and building a sparse camera model."],
  building_dense_cloud: ["Building the dense point cloud", "Multi-view stereo is estimating surface samples from registered cameras."],
  building_mesh: ["Reconstructing a triangle surface", "The dense cloud is becoming a connected visualization mesh."],
  repairing_mesh: ["Checking and repairing the mesh", "Removing tiny fragments and testing manifold and watertight status."],
  texturing: ["Projecting camera textures", "The native texturer is building UVs and a camera-derived texture atlas."],
  applying_scale: ["Applying explicit scale", "Scale remains unknown unless a two-point measurement was supplied."],
  fitting_cad: ["Testing analytic CAD fits", "Only conservative boxes and cylinders can pass the STEP gate in this release."],
  exporting: ["Writing separate representations", "Publishing PLY, OBJ, GLB, conditional STL, and conditional STEP."],
  validating_outputs: ["Reopening every advertised output", "Checking geometry, schemas, hashes, and safe ZIP paths."],
};

function setMode(mode) {
  if (!modeCopy[mode] || state.submitting) return;
  state.mode = mode;
  resetFiles();
  $$(".mode-button").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  const copy = modeCopy[mode];
  $("#capture-panel").setAttribute("aria-labelledby", `mode-${mode}`);
  $("#upload-title").textContent = copy.title;
  $("#drop-title").textContent = copy.drop;
  $("#drop-copy").textContent = copy.copy;
  $("#browse-chip").textContent = copy.browse;
  $("#file-counter").textContent = copy.counter;
  $("#mode-warning").textContent = copy.warning;
  const input = $("#file-input");
  input.multiple = mode === "photos";
  input.accept = mode === "video"
    ? "video/mp4,video/quicktime,video/x-matroska,video/webm,video/x-msvideo"
    : "image/png,image/jpeg,image/webp";
  $$(".video-setting").forEach((element) => { element.hidden = mode !== "video"; });
  renderReadiness();
}

async function chooseFiles(fileList) {
  const incoming = [...fileList];
  clearMessage();
  const error = validateSelection(incoming);
  if (error) {
    resetFiles();
    showMessage(error);
    renderReadiness();
    return;
  }
  resetFiles();
  state.files = incoming;
  const thumbnailGeneration = state.thumbnailGeneration;
  $("#selection").hidden = false;
  $("#selection-count").textContent = state.mode === "photos"
    ? `${incoming.length} photos selected`
    : incoming[0].name;
  $("#selection-size").textContent = formatBytes(incoming.reduce((sum, file) => sum + file.size, 0));
  try {
    await renderThumbnails(incoming, thumbnailGeneration);
  } catch (error) {
    if (thumbnailGeneration !== state.thumbnailGeneration) return;
    resetFiles();
    showMessage(error.message || "A photo preview could not be decoded.");
  }
  renderReadiness();
}

function validateSelection(files) {
  if (state.mode === "photos" && (files.length < MIN_PHOTOS || files.length > MAX_PHOTOS)) {
    return `Choose ${MIN_PHOTOS}–${MAX_PHOTOS} overlapping photos. ${RECOMMENDED_PHOTOS}–50 is recommended.`;
  }
  if (state.mode !== "photos" && files.length !== 1) return "Choose exactly one file for this mode.";
  const video = state.mode === "video";
  const imageTypes = new Set(["image/png", "image/jpeg", "image/webp", "image/jpg"]);
  const videoTypes = new Set(["video/mp4", "video/quicktime", "video/x-matroska", "video/webm", "video/x-msvideo", "video/avi"]);
  for (const file of files) {
    const allowed = video ? videoTypes : imageTypes;
    if (file.type && !allowed.has(file.type)) return `${file.name} is not a supported ${video ? "video" : "image"} type.`;
    if (file.size <= 0) return `${file.name} is empty.`;
    if (!video && file.size > MAX_IMAGE_BYTES) return `${file.name} exceeds 25 MiB.`;
    if (video && file.size > MAX_VIDEO_BYTES) return `${file.name} exceeds 2 GiB.`;
  }
  if (!video && files.reduce((sum, file) => sum + file.size, 0) > MAX_PHOTO_SET_BYTES) {
    return "The photo set exceeds the 500 MiB job limit.";
  }
  return null;
}

async function renderThumbnails(files, generation) {
  const strip = $("#thumbnail-strip");
  strip.replaceChildren();
  const visible = files.slice(0, 50);
  for (let index = 0; index < visible.length; index += 1) {
    const file = visible[index];
    const item = document.createElement("div");
    item.className = `thumb ${state.mode === "video" ? "video-thumb" : ""}`;
    if (state.mode !== "video") {
      const image = document.createElement("img");
      image.src = await inspectImage(file);
      if (generation !== state.thumbnailGeneration) return;
      image.alt = `Selected view ${index + 1}`;
      item.appendChild(image);
    }
    const label = document.createElement("span");
    label.textContent = state.mode === "photos" ? String(index + 1).padStart(2, "0") : "VIDEO";
    item.appendChild(label);
    strip.appendChild(item);
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  if (files.length > visible.length) {
    const remainder = document.createElement("div");
    remainder.className = "thumb thumb-more";
    remainder.textContent = `+${files.length - visible.length}`;
    strip.appendChild(remainder);
  }
}

async function inspectImage(file) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    await image.decode();
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    if (!width || !height || width > MAX_IMAGE_EDGE || height > MAX_IMAGE_EDGE || width * height > MAX_IMAGE_PIXELS) {
      throw new Error(`${file.name} exceeds the ${MAX_IMAGE_PIXELS.toLocaleString()}-pixel image limit.`);
    }
    const ratio = Math.min(1, THUMBNAIL_MAX_EDGE / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * ratio));
    canvas.height = Math.max(1, Math.round(height * ratio));
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("This browser cannot create bounded image previews.");
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.74);
  } catch (error) {
    if (error instanceof Error && error.message.includes("pixel image limit")) throw error;
    throw new Error(`${file.name} is not a readable browser image.`);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function resetFiles() {
  state.thumbnailGeneration += 1;
  state.files = [];
  $("#file-input").value = "";
  $("#selection").hidden = true;
  $("#thumbnail-strip").replaceChildren();
}

function modeReady() {
  const capabilities = state.capabilities?.capabilities;
  if (!capabilities) return false;
  if (state.mode === "image") return state.capabilities.single_image?.available === true;
  if (!capabilities.photo_reconstruction || !capabilities.mesh_processing) return false;
  if (!$("#use-gpu").checked) {
    const tools = capabilities.tools || {};
    const cpuDenseTools = ["interface_colmap", "densify_point_cloud", "reconstruct_mesh", "refine_mesh"];
    if (!cpuDenseTools.every((key) => tools[key]?.available === true)) return false;
  }
  return state.mode !== "video" || capabilities.video_ingest === true;
}

function renderReadiness() {
  const count = state.files.length;
  const captureReady = state.mode === "photos" ? count >= MIN_PHOTOS && count <= MAX_PHOTOS : count === 1;
  const ready = captureReady && modeReady() && !state.submitting;
  $("#build-button").disabled = !ready;
  $("#capture-check").textContent = state.mode === "photos"
    ? (count >= RECOMMENDED_PHOTOS ? `${count} overlapping views ready` : `${count} selected · ${RECOMMENDED_PHOTOS} recommended`)
    : (count ? "One file ready" : "Choose one file");
  $("#toolchain-check").textContent = modeReady() ? "Required local tools ready" : "Dependency unavailable";
  const dots = $$(".quality-dot");
  dots[0].className = `quality-dot ${captureReady ? "good" : "waiting"}`;
  dots[1].className = "quality-dot waiting";
  dots[2].className = `quality-dot ${modeReady() ? "good" : "bad"}`;
  $("#file-counter").textContent = state.mode === "photos"
    ? `${count} / ${RECOMMENDED_PHOTOS} recommended`
    : `${count} / 1 ${state.mode === "video" ? "video" : "image"}`;
  renderDependencyPanel();
}

function renderDependencyPanel() {
  if (!state.capabilities) return;
  const panel = $("#dependency-panel");
  if (modeReady()) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const list = $("#dependency-list");
  list.replaceChildren();
  if (state.mode === "image") {
    $("#dependency-title").textContent = "Single-photo provider unavailable";
    $("#dependency-copy").textContent = state.capabilities.single_image?.reason || "Upload multiple views instead.";
    return;
  }
  $("#dependency-title").textContent = "Real local reconstruction is not ready";
  $("#dependency-copy").textContent = "CadPro will not silently substitute the legacy silhouette builder or a mock model.";
  const tools = state.capabilities.capabilities?.tools || {};
  const required = state.mode === "video" ? ["colmap", "ffmpeg", "ffprobe", "trimesh"] : ["colmap", "trimesh"];
  if (!$("#use-gpu").checked) required.push("interface_colmap", "densify_point_cloud", "reconstruct_mesh", "refine_mesh");
  required.forEach((key) => {
    const tool = tools[key];
    if (!tool || tool.available) return;
    const item = document.createElement("li");
    item.textContent = `${tool.name}: ${tool.reason || "unavailable"} ${tool.install_hint || ""}`;
    list.appendChild(item);
  });
}

async function startBuild() {
  if ($("#build-button").disabled || state.submitting) return;
  state.submitting = true;
  renderReadiness();
  clearMessage();
  const form = new FormData();
  form.append("quality_preset", $("#quality-preset").value);
  form.append("feature_matcher", $("#feature-matcher").value);
  form.append("mesher", $("#mesher").value);
  form.append("use_gpu", String($("#use-gpu").checked));
  form.append("generate_cad", String($("#generate-cad").checked));
  let endpoint;
  if (state.mode === "photos") {
    endpoint = "/api/v2/jobs/photos";
    state.files.forEach((file) => form.append("files", file, file.name));
  } else if (state.mode === "video") {
    endpoint = "/api/v2/jobs/video";
    form.append("file", state.files[0], state.files[0].name);
    form.append("target_frames", $("#target-frames").value);
    form.append("maximum_duration_seconds", $("#maximum-duration").value);
  } else {
    endpoint = "/api/v2/jobs/single-image";
    form.append("file", state.files[0], state.files[0].name);
  }
  try {
    const response = await fetch(endpoint, { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(apiMessage(payload, `Submission failed (${response.status}).`));
    state.jobId = payload.id;
    state.statusUrl = payload.status_url;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ id: state.jobId, status_url: state.statusUrl }));
    showProgress();
    await pollJob();
  } catch (error) {
    state.submitting = false;
    showMessage(error.message || "The reconstruction job could not be submitted.");
    $("#progress-panel").hidden = true;
    $("#capture-panel").hidden = false;
    renderReadiness();
  }
}

function showProgress() {
  $("#capture-panel").hidden = true;
  $("#progress-panel").hidden = false;
  $("#result-section").hidden = true;
  updateProgress({ stage: "queued", progress: 0 });
  $("#progress-panel").focus();
}

async function pollJob() {
  clearTimeout(state.pollTimer);
  if (!state.statusUrl) return;
  try {
    const response = await fetch(state.statusUrl, { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(apiMessage(payload, "Could not read the persistent job."));
    updateProgress(payload);
    if (payload.status === "completed") {
      localStorage.removeItem(STORAGE_KEY);
      state.submitting = false;
      renderResult(payload);
      return;
    }
    if (payload.status === "failed" || payload.status === "cancelled") {
      localStorage.removeItem(STORAGE_KEY);
      state.submitting = false;
      const messages = (payload.errors || []).map((item) => item.message).filter(Boolean);
      throw new Error(messages.join(" ") || `The job ${payload.status}.`);
    }
    state.pollTimer = setTimeout(pollJob, 1000);
  } catch (error) {
    state.submitting = false;
    $("#progress-panel").hidden = true;
    $("#capture-panel").hidden = false;
    showMessage(error.message || "The job status could not be read.");
    renderReadiness();
  }
}

function updateProgress(payload) {
  const stage = payload.stage || "queued";
  const value = Math.max(0, Math.min(100, Number(payload.progress) || 0));
  const copy = stageLabels[stage] || [stage.replaceAll("_", " "), "The local worker is processing this stage."];
  $("#progress-title").textContent = copy[0];
  $("#progress-copy").textContent = copy[1];
  $("#progress-stage").textContent = stage.replaceAll("_", " ").toUpperCase();
  $("#progress-value").textContent = `${value}%`;
  $("#progress-bar").style.width = `${value}%`;
  $("#progress-track").setAttribute("aria-valuenow", String(value));
  const stageOrder = ["validating", "estimating_cameras", "building_dense_cloud", "building_mesh", "repairing_mesh", "exporting", "validating_outputs"];
  const stageGroup = {
    queued: "validating", validating: "validating", extracting_frames: "validating", analyzing_images: "validating",
    estimating_cameras: "estimating_cameras", building_dense_cloud: "building_dense_cloud",
    building_mesh: "building_mesh", texturing: "building_mesh",
    applying_scale: "repairing_mesh", repairing_mesh: "repairing_mesh",
    exporting: "exporting", fitting_cad: "exporting", validating_outputs: "validating_outputs",
  };
  const current = stageOrder.indexOf(stageGroup[stage] || stage);
  $$("#stage-list li").forEach((item) => {
    const index = stageOrder.indexOf(item.dataset.stage);
    item.classList.toggle("active", index === current);
    item.classList.toggle("complete", index >= 0 && current > index);
  });
}

async function cancelJob() {
  if (!state.jobId) return;
  $("#cancel-job").disabled = true;
  try {
    const response = await fetch(`/api/v2/jobs/${state.jobId}/cancel`, { method: "POST" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(apiMessage(payload, "Cancellation failed."));
    updateProgress(payload);
    await pollJob();
  } catch (error) {
    showMessage(error.message || "Cancellation failed.");
  } finally {
    $("#cancel-job").disabled = false;
  }
}

function renderResult(payload) {
  $("#progress-panel").hidden = true;
  $("#capture-panel").hidden = true;
  $("#result-section").hidden = false;
  const report = payload.report || {};
  const metrics = report.metrics || {};
  const scale = report.scale || {};
  const artifacts = payload.artifacts || [];
  const preview = artifacts.find((item) => item.kind === "interactive_preview");
  if (preview) $("#model-preview").src = preview.download_url;
  $("#quality-grade").textContent = `${String(report.quality_class || "unknown").toUpperCase()} RECONSTRUCTION`;
  $("#quality-summary").textContent = `${formatNumber(metrics.registration_percentage, 1)}% cameras registered · ${formatNumber(metrics.reprojection_error_px, 2)} px reprojection error`;
  const dimensions = metrics.bounding_box || [];
  $("#metric-x").textContent = formatNumber(dimensions[0], 3);
  $("#metric-y").textContent = formatNumber(dimensions[1], 3);
  $("#metric-z").textContent = formatNumber(dimensions[2], 3);
  $("#metric-triangles").textContent = formatNumber(metrics.triangles, 0);
  $("#metric-points").textContent = formatNumber(metrics.dense_points, 0);
  $("#metric-cameras").textContent = `${metrics.registered_cameras || 0}/${metrics.accepted_images || 0}`;
  const unit = scale.calibrated ? scale.output_unit : "unknown";
  $$(".metric-unit").forEach((element) => { element.textContent = unit; });
  const scaleBox = $("#scale-status");
  scaleBox.classList.toggle("unknown", !scale.calibrated);
  scaleBox.classList.toggle("known", Boolean(scale.calibrated));
  scaleBox.innerHTML = scale.calibrated
    ? `<b>SCALE CALIBRATED · ${escapeHtml(String(scale.output_unit).toUpperCase())}</b><span>${escapeHtml(scale.warning || "Verify the selected points and measurement uncertainty.")}</span>`
    : "<b>SCALE UNKNOWN</b><span>Dimensions are arbitrary until a two-point measurement is supplied.</span>";
  $("#calibration-card").hidden = Boolean(scale.calibrated);
  renderWarnings(report.warnings || payload.warnings || []);
  renderArtifacts(artifacts);
  const contact = artifacts.find((item) => item.kind === "selected_frame_contact_sheet");
  $("#contact-card").hidden = !contact;
  if (contact) {
    $("#contact-sheet").src = contact.download_url;
    $("#contact-sheet-link").href = contact.download_url;
  }
  state.jobId = payload.id;
  state.statusUrl = payload.status_url;
  $("#result-section").scrollIntoView({ behavior: "smooth", block: "start" });
  $("#result-title").focus({ preventScroll: true });
}

function renderWarnings(warnings) {
  const list = $("#warning-list");
  list.replaceChildren();
  if (!warnings.length) {
    const item = document.createElement("li");
    item.textContent = "No structured warnings were reported. Numerical quality metrics still require inspection.";
    list.appendChild(item);
    return;
  }
  warnings.forEach((warning) => {
    const item = document.createElement("li");
    const code = document.createElement("b");
    code.textContent = String(warning.code || "warning").replaceAll("_", " ").toUpperCase();
    item.appendChild(code);
    item.appendChild(document.createTextNode(` ${warning.message || "Review the report."}`));
    list.appendChild(item);
  });
}

function renderArtifacts(artifacts) {
  const grid = $("#artifact-grid");
  grid.replaceChildren();
  const visible = artifacts.filter((artifact) => artifact.kind !== "texture_resource" && artifact.kind !== "processing_log");
  visible.sort((left, right) => artifactPriority(left) - artifactPriority(right));
  visible.forEach((artifact) => {
    const link = document.createElement("a");
    link.className = `artifact-card ${artifact.kind === "complete_bundle" ? "primary" : ""}`;
    link.href = artifact.download_url;
    link.download = artifact.filename;
    const label = artifactLabel(artifact);
    link.innerHTML = `<span>${escapeHtml(label.eyebrow)}</span><b>${escapeHtml(label.title)}</b><small>${escapeHtml(label.copy)}</small><i>${formatBytes(artifact.size_bytes)} · DOWNLOAD →</i>`;
    grid.appendChild(link);
  });
}

function artifactLabel(artifact) {
  const suffix = String(artifact.filename || "").split(".").pop().toUpperCase();
  const labels = {
    sparse_point_cloud: ["SPARSE CAMERA MODEL", "Sparse point cloud", "Feature landmarks used to register cameras"],
    dense_point_cloud: ["DENSE RECONSTRUCTION", "Dense point cloud", "Surface samples · PLY"],
    triangle_mesh: ["TRIANGLE GEOMETRY", suffix === "OBJ" ? "OBJ mesh" : "Cleaned mesh", artifact.textured ? "Camera texture resources linked" : "Untextured triangle representation"],
    visualization_model: ["INTERACTIVE VISUAL", "GLB model", artifact.textured ? "Embedded UV/material/texture linkage validated" : "Geometry validated · no texture claim"],
    watertight_printable_mesh: ["REPAIR GATE PASSED", "Watertight STL", "Reopened as a watertight manifold mesh"],
    analytic_cad_brep: ["ANALYTIC FIT PASSED", "Fitted STEP CAD", "Scaled B-rep reopened as one valid solid"],
    editable_cad_script: ["EDITABLE REPRODUCTION", "CadQuery script", "Compact box or cylinder construction"],
    selected_frame_contact_sheet: ["VIDEO EVIDENCE", "Selected frame sheet", "Exact frames admitted to reconstruction"],
    reconstruction_report: ["QUALITY + PROVENANCE", "Reconstruction report", "Metrics, scale, warnings, CAD residuals, and timings"],
    reproducibility_manifest: ["REPRODUCIBILITY", "Job manifest", "Input hashes, settings, tool versions, commands, artifact hashes"],
    interactive_preview: ["SELF-CONTAINED VIEWER", "Interactive preview", "Orbit, wireframe, normals, grid, axes, and point picking"],
    complete_bundle: ["EVERY PRODUCED OUTPUT", "Complete ZIP bundle", "Safe relative paths · all available artifacts and reports"],
  };
  const value = labels[artifact.kind] || [suffix || "FILE", artifact.filename, "Validated job artifact"];
  return { eyebrow: value[0], title: value[1], copy: value[2] };
}

function artifactPriority(artifact) {
  const order = ["complete_bundle", "visualization_model", "triangle_mesh", "dense_point_cloud", "sparse_point_cloud", "watertight_printable_mesh", "analytic_cad_brep", "editable_cad_script", "reconstruction_report", "reproducibility_manifest", "selected_frame_contact_sheet", "interactive_preview"];
  const index = order.indexOf(artifact.kind);
  return index < 0 ? 99 : index;
}

async function applyCalibration() {
  if (!state.jobId) return;
  const pointA = [$("#point-a-x"), $("#point-a-y"), $("#point-a-z")].map((input) => Number(input.value));
  const pointB = [$("#point-b-x"), $("#point-b-y"), $("#point-b-z")].map((input) => Number(input.value));
  const distance = Number($("#real-distance").value);
  if (![...pointA, ...pointB, distance].every(Number.isFinite) || distance <= 0) {
    $("#calibration-feedback").textContent = "Enter two finite 3D points and a positive real distance.";
    return;
  }
  $("#apply-calibration").disabled = true;
  $("#calibration-feedback").textContent = "Creating a calibrated reconstruction revision…";
  try {
    const response = await fetch(`/api/v2/jobs/${state.jobId}/calibration`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point_a: pointA, point_b: pointB, real_distance: distance, unit: $("#scale-unit").value, selection_uncertainty: 0 }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(apiMessage(payload, "Calibration failed."));
    state.jobId = payload.id;
    state.statusUrl = payload.status_url;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ id: state.jobId, status_url: state.statusUrl }));
    $("#result-section").hidden = true;
    state.submitting = true;
    showProgress();
    await pollJob();
  } catch (error) {
    $("#calibration-feedback").textContent = error.message || "Calibration failed.";
  } finally {
    $("#apply-calibration").disabled = false;
  }
}

function handleViewerMessage(event) {
  if (event.origin !== window.location.origin || event.source !== $("#model-preview").contentWindow) return;
  if (event.data?.type !== "cadpro-point-picked" || !Array.isArray(event.data.point)) return;
  const point = event.data.point.map(Number);
  if (point.length !== 3 || !point.every(Number.isFinite)) return;
  if (state.pickedPoints.length >= 2) state.pickedPoints = [];
  state.pickedPoints.push(point);
  const prefix = state.pickedPoints.length === 1 ? "a" : "b";
  ["x", "y", "z"].forEach((axis, index) => { $(`#point-${prefix}-${axis}`).value = point[index].toPrecision(10); });
  $("#calibration-feedback").textContent = state.pickedPoints.length === 1 ? "Point A selected. Double-click Point B." : "Points A and B selected. Enter their real separation.";
}

function startAnother() {
  clearTimeout(state.pollTimer);
  localStorage.removeItem(STORAGE_KEY);
  state.jobId = null;
  state.statusUrl = null;
  state.submitting = false;
  state.pickedPoints = [];
  resetFiles();
  $("#model-preview").src = "about:blank";
  $("#result-section").hidden = true;
  $("#progress-panel").hidden = true;
  $("#capture-panel").hidden = false;
  clearMessage();
  renderReadiness();
  $(".studio").scrollIntoView({ behavior: "smooth" });
}

async function loadCapabilities() {
  try {
    const response = await fetch("/api/v2/capabilities", { cache: "no-store" });
    if (!response.ok) throw new Error("Capability check failed.");
    state.capabilities = await response.json();
    const capabilities = state.capabilities.capabilities || {};
    const broadlyReady = capabilities.photo_reconstruction && capabilities.mesh_processing;
    $("#engine-status").classList.toggle("offline", !broadlyReady);
    $("#engine-status").innerHTML = broadlyReady ? "<span></span> Local photogrammetry ready" : "<span></span> Native scan tools missing";
  } catch (_error) {
    state.capabilities = { capabilities: {}, single_image: { available: false, reason: "The capability endpoint could not be reached." } };
    $("#engine-status").classList.add("offline");
    $("#engine-status").innerHTML = "<span></span> Scan service unavailable";
  }
  renderReadiness();
  resumeJob();
}

async function resumeJob() {
  let saved;
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch (_error) { saved = null; }
  if (!saved?.status_url) return;
  state.jobId = saved.id;
  state.statusUrl = saved.status_url;
  state.submitting = true;
  showProgress();
  await pollJob();
}

function apiMessage(payload, fallback) {
  const error = payload?.error;
  if (!error) return fallback;
  const missing = error.details?.missing;
  if (Array.isArray(missing) && missing.length) {
    return `${error.message} ${missing.map((item) => `${item.tool}: ${item.install_hint || item.reason || "unavailable"}`).join(" ")}`;
  }
  return error.message || fallback;
}

function showMessage(message) { $("#message").hidden = false; $("#message").textContent = message; }
function clearMessage() { $("#message").hidden = true; $("#message").textContent = ""; }
function formatNumber(value, digits = 2) { const number = Number(value); return Number.isFinite(number) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(number) : "—"; }
function formatBytes(value) { const bytes = Number(value) || 0; if (bytes >= GIBIBYTE) return `${(bytes / GIBIBYTE).toFixed(2)} GiB`; if (bytes >= MEBIBYTE) return `${(bytes / MEBIBYTE).toFixed(1)} MiB`; if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`; return `${bytes} B`; }
function escapeHtml(value) { return value.replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]); }

$$('.mode-button').forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
  button.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const tabs = $$('.mode-button');
    const current = tabs.indexOf(button);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    setMode(tabs[next].dataset.mode);
    tabs[next].focus();
  });
});

$("#file-input").addEventListener("change", (event) => chooseFiles(event.target.files));
$("#clear-files").addEventListener("click", () => { resetFiles(); renderReadiness(); });
$("#build-button").addEventListener("click", startBuild);
$("#cancel-job").addEventListener("click", cancelJob);
$("#new-model").addEventListener("click", startAnother);
$("#apply-calibration").addEventListener("click", applyCalibration);
$("#use-gpu").addEventListener("change", renderReadiness);
window.addEventListener("message", handleViewerMessage);

const dropZone = $("#drop-zone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (event) => chooseFiles(event.dataTransfer.files));

setMode("photos");
loadCapabilities();

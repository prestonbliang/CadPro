const state = {
  mode: "image",
  files: [],
  thumbnails: new Map(),
  dimensionsValid: null,
  photoCheckProgress: null,
  selectionGeneration: 0,
  selectionController: null,
  aiAvailable: false,
  neuralAvailable: false,
  conceptMeshAvailable: false,
  jobId: null,
  pollTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const fileInput = $("#file-input");
const dropZone = $("#drop-zone");
const buildButton = $("#build-button");
const imageTypes = new Set([
  "image/png", "image/jpeg", "image/webp", "image/bmp", "image/x-bmp",
]);
const videoTypes = new Set([
  "video/mp4", "video/quicktime", "video/x-msvideo", "video/avi", "video/webm",
  "video/x-matroska", "video/mkv", "video/x-m4v",
]);
const MEBIBYTE = 1024 * 1024;
const GIBIBYTE = 1024 * MEBIBYTE;
const MAX_IMAGE_BYTES = 25 * MEBIBYTE;
const MAX_PHOTO_SET_BYTES = 500 * MEBIBYTE;
const MAX_VIDEO_BYTES = 2 * GIBIBYTE;
const MAX_DIMENSION_MM = 1_000_000;
const MAX_IMAGE_EDGE = 8_192;
const MAX_IMAGE_PIXELS = 12_500_000;
const THUMBNAIL_MAX_EDGE = 240;

const modeCopy = {
  image: {
    uploadTitle: "Add one object photo",
    dropTitle: "Drop one photo here",
    dropCopy: "or browse · PNG, JPEG, WebP, or BMP",
    chip: "CHOOSE PHOTO",
    rail: "<b>Fast profile.</b> One square-on image creates a measured profile extrusion. Add more views when shape depth matters.",
  },
  photos: {
    uploadTitle: "Add an ordered photo orbit",
    dropTitle: "Drop 20–50 photos here",
    dropCopy: "or browse · select them in rotation order",
    chip: "CHOOSE PHOTOS",
    rail: "<b>Best measured shape.</b> Use 20–50 evenly spaced views from one complete revolution with a fixed camera.",
  },
  video: {
    uploadTitle: "Add one turntable video",
    dropTitle: "Drop one full-rotation video here",
    dropCopy: "or browse · one steady constant-speed 360° turn",
    chip: "CHOOSE VIDEO",
    rail: "<b>Easiest full orbit.</b> Keep the camera fixed while the object completes exactly one steady revolution.",
  },
};

function setMode(mode) {
  if (!Object.hasOwn(modeCopy, mode)) return;
  const changed = state.mode !== mode;
  state.mode = mode;
  if (changed) resetFiles();
  $$(".mode-button").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    if (active) $("#capture-panel").setAttribute("aria-labelledby", button.id);
  });
  const copy = modeCopy[mode];
  $("#upload-title").textContent = copy.uploadTitle;
  $("#drop-title").textContent = copy.dropTitle;
  $("#drop-copy").textContent = copy.dropCopy;
  $("#browse-chip").textContent = copy.chip;
  $("#rail-note").innerHTML = copy.rail;
  const video = mode === "video";
  fileInput.accept = video
    ? "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska"
    : "image/png,image/jpeg,image/webp,image/bmp";
  fileInput.multiple = mode === "photos";
  $$(".image-setting").forEach((field) => { field.hidden = mode !== "image"; });
  $$(".neural-setting").forEach((field) => { field.hidden = mode !== "image"; });
  if (mode !== "image") $("#neural-predict").checked = false;
  $$(".orbit-setting").forEach((field) => { field.hidden = mode === "image"; });
  $$(".video-setting").forEach((field) => { field.hidden = !video; });
  syncNeuralControls();
  renderSelection();
}

function neuralPredictionSelected() {
  return state.mode === "image" && state.neuralAvailable && $("#neural-predict").checked;
}

function syncNeuralControls() {
  const predicted = neuralPredictionSelected();
  $("#depth-mm").disabled = predicted;
  $("#depth-label").childNodes[0].textContent = predicted
    ? "Extrusion depth · neural prediction "
    : "Extrusion depth ";
}

function resetFiles() {
  state.selectionController?.abort();
  state.selectionController = null;
  state.selectionGeneration += 1;
  state.files = [];
  state.thumbnails = new Map();
  state.dimensionsValid = null;
  state.photoCheckProgress = null;
  fileInput.value = "";
  showMessage("");
}

async function chooseFiles(fileList) {
  state.selectionController?.abort();
  const controller = new AbortController();
  const generation = state.selectionGeneration + 1;
  const mode = state.mode;
  state.selectionGeneration = generation;
  state.selectionController = controller;
  state.files = [];
  state.thumbnails = new Map();
  state.dimensionsValid = null;
  state.photoCheckProgress = null;
  fileInput.value = "";
  showMessage("");

  const incoming = [...fileList];
  const video = mode === "video";
  const photos = mode === "photos";
  const acceptedTypes = video ? videoTypes : imageTypes;
  const rejectSelection = (message) => {
    if (!isCurrentSelection(generation, mode)) return;
    state.selectionController = null;
    showMessage(message);
    renderSelection();
  };

  if (!incoming.length) {
    rejectSelection("Choose a capture file to continue.");
    return;
  }
  if (mode === "image" && incoming.length !== 1) {
    rejectSelection("One-photo mode accepts exactly one image. Choose Photo orbit for 20–50 views.");
    return;
  }
  if (photos && incoming.length > 50) {
    rejectSelection("Use at most 50 ordered photos. Remove duplicate or extra views.");
    return;
  }
  if (video && incoming.length !== 1) {
    rejectSelection("Choose exactly one video containing one complete 360° revolution.");
    return;
  }
  const wrongType = incoming.find((file) => file.type && !acceptedTypes.has(file.type));
  if (wrongType) {
    rejectSelection(`${wrongType.name} is not a supported ${video ? "video" : "image"} file.`);
    return;
  }
  const emptyFile = incoming.find((file) => file.size === 0);
  if (emptyFile) {
    rejectSelection(`${emptyFile.name} is empty. Choose the original capture file.`);
    return;
  }
  if (!video) {
    const oversized = incoming.find((file) => file.size > MAX_IMAGE_BYTES);
    if (oversized) {
      rejectSelection(`${oversized.name} exceeds the 25 MiB per-image limit.`);
      return;
    }
    const totalBytes = incoming.reduce((sum, file) => sum + file.size, 0);
    if (totalBytes > MAX_PHOTO_SET_BYTES) {
      rejectSelection(`The photo set is ${formatFileSize(totalBytes)}. Keep it at or below 500 MiB.`);
      return;
    }
  } else if (incoming[0].size > MAX_VIDEO_BYTES) {
    rejectSelection(`${incoming[0].name} exceeds the 2 GiB video limit.`);
    return;
  }

  state.files = [...incoming];
  state.dimensionsValid = video ? true : null;
  state.photoCheckProgress = !video ? { checked: 0, total: incoming.length } : null;
  renderSelection();
  if (!video) await verifyImageDimensions(incoming, generation, mode, controller.signal);
  if (isCurrentSelection(generation, mode)) state.selectionController = null;
}

function isCurrentSelection(generation, mode) {
  return generation === state.selectionGeneration && mode === state.mode;
}

async function verifyImageDimensions(files, generation, mode, signal) {
  try {
    let expectedDimensions = null;
    for (let index = 0; index < files.length; index += 1) {
      const result = await inspectImage(files[index], signal);
      if (!isCurrentSelection(generation, mode)) return;
      if (expectedDimensions === null) {
        expectedDimensions = [result.width, result.height];
      } else if (result.width !== expectedDimensions[0] || result.height !== expectedDimensions[1]) {
        state.dimensionsValid = false;
        state.photoCheckProgress = null;
        showMessage("Every orbit photo must have identical pixel dimensions. Avoid crops or mixed camera modes.");
        renderSelection();
        return;
      }
      state.thumbnails.set(files[index], result.thumbnail);
      state.photoCheckProgress = { checked: index + 1, total: files.length };
      if ((index + 1) % 4 === 0 || index === files.length - 1) renderSelection();
    }
    state.dimensionsValid = true;
    state.photoCheckProgress = null;
    showMessage("");
  } catch (error) {
    if (error?.name === "AbortError" || !isCurrentSelection(generation, mode)) return;
    state.dimensionsValid = false;
    state.photoCheckProgress = null;
    showMessage(error?.name === "ImageLimitError"
      ? error.message
      : "One or more images could not be decoded. Replace damaged or unsupported files.");
  }
  renderSelection();
}

function inspectImage(file, signal) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    let settled = false;
    const cleanup = () => {
      signal.removeEventListener("abort", abort);
      image.onload = null;
      image.onerror = null;
      image.removeAttribute("src");
      URL.revokeObjectURL(url);
    };
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback(value);
    };
    const abort = () => {
      const error = new Error("Image inspection cancelled");
      error.name = "AbortError";
      finish(reject, error);
    };
    signal.addEventListener("abort", abort, { once: true });
    image.onload = () => {
      try {
        if (signal.aborted) return abort();
        const width = image.naturalWidth;
        const height = image.naturalHeight;
        if (!width || !height) throw new Error("empty image dimensions");
        if (width > MAX_IMAGE_EDGE || height > MAX_IMAGE_EDGE || width * height > MAX_IMAGE_PIXELS) {
          const error = new Error("Use images with at most 12.5 million pixels and 8,192 pixels per side.");
          error.name = "ImageLimitError";
          throw error;
        }
        const scale = Math.min(1, THUMBNAIL_MAX_EDGE / Math.max(width, height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        const context = canvas.getContext("2d", { alpha: false });
        if (!context) throw new Error("thumbnail canvas unavailable");
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = "high";
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        const thumbnail = canvas.toDataURL("image/jpeg", 0.74);
        canvas.width = 1;
        canvas.height = 1;
        finish(resolve, { width, height, thumbnail });
      } catch (error) {
        finish(reject, error);
      }
    };
    image.onerror = () => finish(reject, new Error("decode failed"));
    image.decoding = "async";
    image.src = url;
  });
}

function moveFile(index, direction) {
  const target = index + direction;
  if (target < 0 || target >= state.files.length) return;
  [state.files[index], state.files[target]] = [state.files[target], state.files[index]];
  renderSelection();
}

function dimensionValue(selector) {
  const value = Number($(selector).value);
  return Number.isFinite(value) && value > 0 && value <= MAX_DIMENSION_MM ? value : null;
}

function renderSelection() {
  const count = state.files.length;
  const mode = state.mode;
  const photos = mode === "photos";
  const video = mode === "video";
  const countValid = mode === "image" ? count === 1 : photos ? count >= 20 && count <= 50 : count === 1;
  const captureValid = countValid && state.dimensionsValid === true;
  const width = dimensionValue("#width-mm");
  const depth = dimensionValue("#depth-mm");
  const neural = neuralPredictionSelected();
  const geometryValid = mode === "image" ? neural || depth !== null : true;
  const totalBytes = state.files.reduce((sum, file) => sum + file.size, 0);

  $("#file-counter").textContent = mode === "image"
    ? `${count} / 1 image`
    : photos ? `${count} / 20 minimum` : `${count} / 1 video`;
  $("#selection").hidden = count === 0;
  $("#selection-count").textContent = mode === "image"
    ? (count ? state.files[0].name : "No image loaded")
    : photos ? `${count} ordered views loaded` : (count ? state.files[0].name : "No video loaded");
  $("#selection-size").textContent = `${formatFileSize(totalBytes)} total`;

  if (!video && state.photoCheckProgress) {
    const next = Math.min(state.photoCheckProgress.checked + 1, state.photoCheckProgress.total);
    $("#capture-check").textContent = `Checking image ${next} of ${state.photoCheckProgress.total}`;
  } else if (mode === "image") {
    $("#capture-check").textContent = captureValid ? "One image ready" : (count ? "Checking image" : "Add one clear image");
  } else if (photos) {
    $("#capture-check").textContent = captureValid ? `${count} ordered views ready` : `Add ${Math.max(0, 20 - count)} more photos`;
  } else {
    $("#capture-check").textContent = captureValid ? "One video ready" : "Add one turntable video";
  }

  $("#scale-check").textContent = width ? `${formatNumber(width)} mm measured width` : "Enter a valid measured width";
  $("#geometry-check-title").textContent = mode === "image" ? "Depth" : video ? "Sampling" : "Coverage";
  $("#geometry-check").textContent = mode === "image"
    ? neural
      ? "Trained model will predict extrusion depth"
      : (depth ? `${formatNumber(depth)} mm extrusion` : "Enter a valid extrusion depth")
    : video ? `${$("#view-count").value} evenly sampled views` : "One complete ordered revolution";

  const dots = $$(".quality-dot");
  dots[0].className = `quality-dot ${captureValid ? "good" : "waiting"}`;
  dots[1].className = `quality-dot ${width ? "good" : "waiting"}`;
  dots[2].className = `quality-dot ${geometryValid && (mode !== "photos" || countValid) ? "good" : "waiting"}`;
  buildButton.disabled = !(captureValid && width && geometryValid);

  const strip = $("#thumbnail-strip");
  strip.replaceChildren();
  state.files.forEach((file, index) => {
    const item = document.createElement("div");
    item.className = `thumb ${video ? "video-thumb" : ""}`;
    const thumbnail = state.thumbnails.get(file);
    if (thumbnail) item.style.backgroundImage = `url("${thumbnail}")`;
    if (mode === "image") item.classList.add("single-thumb");
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    item.appendChild(number);
    item.title = file.name;
    if (photos) {
      const controls = document.createElement("div");
      controls.className = "thumb-controls";
      const previous = document.createElement("button");
      previous.type = "button";
      previous.textContent = "←";
      previous.disabled = index === 0;
      previous.setAttribute("aria-label", `Move ${file.name} earlier`);
      previous.addEventListener("click", () => moveFile(index, -1));
      const next = document.createElement("button");
      next.type = "button";
      next.textContent = "→";
      next.disabled = index === state.files.length - 1;
      next.setAttribute("aria-label", `Move ${file.name} later`);
      next.addEventListener("click", () => moveFile(index, 1));
      controls.append(previous, next);
      item.appendChild(controls);
    }
    strip.appendChild(item);
  });
}

function formatFileSize(bytes) {
  if (bytes >= GIBIBYTE) return `${(bytes / GIBIBYTE).toFixed(2)} GiB`;
  return `${(bytes / MEBIBYTE).toFixed(bytes ? 1 : 0)} MiB`;
}

function showMessage(text) {
  const message = $("#message");
  message.textContent = text;
  message.hidden = !text;
}

function errorMessage(payload, fallback) {
  if (typeof payload?.error?.message === "string") return payload.error.message;
  if (typeof payload?.detail === "string") return payload.detail;
  if (typeof payload?.message === "string") return payload.message;
  return fallback;
}

async function startBuild() {
  showMessage("");
  buildButton.disabled = true;
  const form = new FormData();
  if (state.mode === "photos") {
    state.files.forEach((file) => form.append("files", file, file.name));
  } else {
    form.append("file", state.files[0], state.files[0].name);
  }
  form.append("width_mm", $("#width-mm").value);
  if (state.mode === "image") {
    const neuralRequested = neuralPredictionSelected();
    form.append("neural_predict", String(neuralRequested));
    if (!neuralRequested) form.append("depth_mm", $("#depth-mm").value);
  } else {
    form.append("clockwise", String($("#rotation-direction").value === "clockwise"));
  }
  if (state.mode === "video") {
    form.append("views", $("#view-count").value);
    form.append("start_frame", "0");
  }
  form.append("ai_enhance", String(state.aiAvailable && $("#ai-enhance").checked));
  form.append("concept_mesh", String(state.conceptMeshAvailable && $("#concept-mesh").checked));
  form.append("object_hint", $("#object-hint").value.trim());

  showProgress();
  updateProgress(8, "upload", "Uploading capture", "The reconstruction service is securing every source file.");
  try {
    const response = await fetch(`/api/jobs/${state.mode}`, { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(payload, `Upload failed (${response.status}).`));
    state.jobId = payload.id;
    const aiRequested = state.aiAvailable && $("#ai-enhance").checked;
    updateProgress(16, aiRequested ? "research" : "segment",
      aiRequested ? "Analyzing the object" : "Checking every silhouette",
      aiRequested ? "Inspecting representative views and searching for cited reference specifications." : "Verifying framing, dimensions, and clean object boundaries.");
    await pollJob(payload.status_url || `/api/jobs/${payload.id}`);
  } catch (error) {
    failBuild(error.message || "The model could not be built.");
  }
}

function showProgress() {
  $(".work-card").hidden = true;
  $(".mode-switch").hidden = true;
  $("#progress-panel").hidden = false;
  $$(".step-rail li").forEach((item, index) => item.classList.toggle("active", index === 2));
  window.scrollTo({ top: $(".studio").offsetTop - 80, behavior: "smooth" });
}

function updateProgress(value, stage, title, copy) {
  const bounded = Math.max(0, Math.min(100, Math.round(value)));
  $("#progress-bar").style.width = `${bounded}%`;
  $("#progress-value").textContent = `${bounded}%`;
  $("#progress-stage").textContent = stage.toUpperCase();
  if (title) $("#progress-title").textContent = title;
  if (copy) $("#progress-copy").textContent = copy;
  const order = ["upload", "research", "segment", "reconstruct", "export"];
  const activeIndex = Math.max(0, order.indexOf(stage));
  $$(".build-stages li").forEach((item, index) => {
    item.classList.toggle("active", index === activeIndex);
    item.classList.toggle("complete", index < activeIndex || bounded === 100);
  });
}

async function pollJob(statusUrl) {
  for (;;) {
    await new Promise((resolve) => { state.pollTimer = setTimeout(resolve, 650); });
    const response = await fetch(statusUrl, { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(payload, "Could not read reconstruction status."));
    if (payload.status === "failed") throw new Error(errorMessage(payload, "Reconstruction failed."));
    if (payload.status === "completed") {
      updateProgress(100, "export", "CAD integrity passed", "STEP, STL, GLB, preview, and diagnostics are ready.");
      await new Promise((resolve) => setTimeout(resolve, 350));
      await renderResult(payload);
      return;
    }
    const serverProgress = Number(payload.progress);
    const value = Number.isFinite(serverProgress) ? Math.max(16, serverProgress) : 16;
    const stage = payload.stage || (value < 35 ? "segment" : value < 78 ? "reconstruct" : "export");
    const labels = {
      queued: ["Waiting for the geometry engine", "Your capture is queued safely."],
      upload: ["Securing every input", "Keeping this job isolated from every other upload."],
      research: ["Researching likely references", "Analyzing representative views and collecting cited specifications."],
      segment: ["Extracting silhouettes", "Tracing clean object boundaries across the supplied capture."],
      reconstruct: ["Building the CAD solid", "Creating and validating the measurement-driven B-rep geometry."],
      export: ["Writing interoperable geometry", "Round-trip checking STEP and preparing Blender-friendly meshes."],
    };
    updateProgress(value, stage, ...(labels[stage] || labels.reconstruct));
  }
}

async function renderResult(payload) {
  const result = payload.result || {};
  const artifacts = result.artifacts || [];
  const bySuffix = (suffix) => artifacts.find((artifact) => artifact.filename.toLowerCase().endsWith(suffix));
  const byName = (name) => artifacts.find((artifact) => artifact.filename.toLowerCase() === name);
  const step = bySuffix(".step") || bySuffix(".stp");
  const stl = bySuffix(".stl");
  const glb = byName("cadpro-model.glb") || bySuffix(".glb");
  const conceptGlb = byName("cadpro-ai-concept.glb");
  const preview = bySuffix(".preview.html") || bySuffix(".html");
  const report = bySuffix(".report.json") || bySuffix(".json");
  if (!step || !stl || !glb || !preview || !report) throw new Error("The job completed without every required export.");

  const metrics = result.metrics || await fetchReportMetrics(report.download_url);
  const dimensions = metrics.dimensions_mm || metrics.geometry?.dimensions_mm || {};
  const dimensionValues = Array.isArray(dimensions) ? dimensions : [dimensions.x, dimensions.y, dimensions.z];
  $("#metric-x").textContent = formatNumber(dimensionValues[0]);
  $("#metric-y").textContent = formatNumber(dimensionValues[1]);
  $("#metric-z").textContent = formatNumber(dimensionValues[2]);
  $("#metric-volume").textContent = formatNumber(metrics.volume_mm3 ?? metrics.geometry?.volume_mm3, 0);
  $("#metric-faces").textContent = formatNumber(metrics.face_count ?? metrics.geometry?.face_count, 0);
  $("#metric-views").textContent = String(result.input_diagnostics?.length || payload.input_count || state.files.length);
  $("#model-preview").src = preview.download_url;
  setDownload("#download-step", step);
  setDownload("#download-stl", stl);
  setDownload("#download-glb", glb);
  setDownload("#download-report", report);
  $("#download-concept").hidden = !conceptGlb;
  if (conceptGlb) setDownload("#download-concept", conceptGlb);
  $("#research-report-link").href = report.download_url;

  const research = result.enrichment || result.research;
  const hasResearch = research?.status === "completed";
  $("#research-result").hidden = !hasResearch;
  if (hasResearch) {
    const identity = research.object_identity || research.identity || {};
    $("#research-object").textContent = identity.common_name || identity.name || identity.label || research.object_name || "Object research complete";
    $("#research-summary").textContent = research.summary || identity.summary || identity.evidence || "Review the cited candidate specifications and uncertainty notes before using any reference dimension.";
  }
  const prediction = result.neural_prediction;
  const hasPrediction = prediction?.status === "completed";
  $("#prediction-result").hidden = !hasPrediction;
  if (hasPrediction) {
    $("#prediction-depth").textContent = `${formatNumber(prediction.predicted_depth_mm)} mm predicted depth`;
    $("#prediction-summary").textContent = `Learned ratio ${formatNumber(prediction.predicted_depth_ratio, 4)} × measured width · heuristic confidence ${formatNumber(Number(prediction.confidence_score) * 100, 0)}%. Verify this estimate against the physical object.`;
  }
  const optionalFailures = [];
  if (research?.status === "failed") {
    optionalFailures.push("AI/web research did not finish; the local measurement-driven CAD exports still completed.");
  }
  if (result.concept_mesh?.status === "failed") {
    optionalFailures.push("The optional AI concept mesh did not finish; the validated STEP, STL, and deterministic GLB are unaffected.");
  }
  $("#optional-warning").hidden = optionalFailures.length === 0;
  $("#optional-warning-copy").textContent = optionalFailures.join(" ");
  const truth = $("#truth-note");
  if (payload.kind === "image") {
    truth.innerHTML = hasPrediction
      ? "<span>!</span><p><b>Neural profile extrusion</b>The outline and width are measured inputs; depth is a learned estimate. Hidden topology is not recovered and every critical feature still needs verification.</p>"
      : "<span>!</span><p><b>Measured profile extrusion</b>The outline comes from one image; depth comes from your chosen value. AI research cannot reveal or verify hidden geometry.</p>";
  } else {
    truth.innerHTML = "<span>!</span><p><b>Measured visual hull</b>The solid comes from intersected silhouettes. Hidden cavities and concavities that never change an outline still require CAD verification.</p>";
  }

  $("#progress-panel").hidden = true;
  $("#result-section").hidden = false;
  $$(".step-rail li").forEach((item, index) => {
    item.classList.toggle("active", index === 3);
    item.classList.toggle("complete", index < 3);
  });
  $("#result-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function fetchReportMetrics(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) return {};
  const report = await response.json();
  return report.geometry || {};
}

function setDownload(selector, artifact) {
  const link = $(selector);
  link.href = artifact.download_url;
  link.setAttribute("download", artifact.filename);
}

function formatNumber(value, maximumFractionDigits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits }).format(number);
}

function failBuild(message) {
  clearTimeout(state.pollTimer);
  $("#progress-panel").hidden = true;
  $(".work-card").hidden = false;
  $(".mode-switch").hidden = false;
  showMessage(message);
  renderSelection();
  $$(".step-rail li").forEach((item, index) => item.classList.toggle("active", index === 0));
  $(".studio").scrollIntoView({ behavior: "smooth", block: "start" });
}

function startAnotherModel() {
  clearTimeout(state.pollTimer);
  state.jobId = null;
  resetFiles();
  renderSelection();
  $("#result-section").hidden = true;
  $("#research-result").hidden = true;
  $("#prediction-result").hidden = true;
  $("#optional-warning").hidden = true;
  $("#optional-warning-copy").textContent = "";
  $("#download-concept").hidden = true;
  $("#model-preview").src = "about:blank";
  $(".work-card").hidden = false;
  $(".mode-switch").hidden = false;
  $$(".step-rail li").forEach((item, index) => {
    item.classList.toggle("active", index === 0);
    item.classList.remove("complete");
  });
  $(".studio").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadCapabilities() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("health check failed");
    const health = await response.json();
    $("#engine-status").innerHTML = "<span></span> Geometry engine ready";
    const intelligence = health.intelligence || health.ai || {};
    state.aiAvailable = intelligence.available === true;
    $("#ai-enhance").disabled = !state.aiAvailable;
    const neural = health.neural_prediction || {};
    state.neuralAvailable = neural.available === true;
    $("#neural-predict").disabled = !state.neuralAvailable;
    const conceptMesh = health.concept_mesh || {};
    state.conceptMeshAvailable = conceptMesh.available === true;
    $("#concept-mesh").disabled = !state.conceptMeshAvailable;
    const optionalAvailable = state.aiAvailable || state.neuralAvailable || state.conceptMeshAvailable;
    const ready = [
      state.neuralAvailable ? "NEURAL" : "",
      state.aiAvailable ? "RESEARCH" : "",
      state.conceptMeshAvailable ? "MESH" : "",
    ].filter(Boolean);
    $("#ai-availability").textContent = ready.length ? `${ready.join(" + ")} READY` : "LOCAL MODE";
    $("#ai-availability").classList.toggle("available", optionalAvailable);
    const notes = [];
    notes.push(state.neuralAvailable
      ? "A trained local depth checkpoint is ready."
      : "Train a depth model, then set CADPRO_NEURAL_ENABLED=1 and CADPRO_NEURAL_CHECKPOINT on the server.");
    notes.push(state.aiAvailable
      ? "Cited vision/web research is configured."
      : "For cited research, set CADPRO_AI_ENRICHMENT=1 and OPENAI_API_KEY on the server.");
    notes.push(state.conceptMeshAvailable
      ? "The optional non-metric concept-mesh worker is connected."
      : "The validated local STEP path remains fully available without a concept-mesh worker.");
    $("#ai-note").textContent = notes.join(" ");
    syncNeuralControls();
    renderSelection();
  } catch (_error) {
    $("#engine-status").classList.add("offline");
    $("#engine-status").innerHTML = "<span></span> Geometry engine unavailable";
    $("#ai-availability").textContent = "UNAVAILABLE";
    $("#ai-note").textContent = "The server health check failed. Refresh after the geometry service is running.";
  }
}

$$(".mode-button").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const tabs = $$(".mode-button");
    const current = tabs.indexOf(button);
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? tabs.length - 1
        : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    setMode(tabs[next].dataset.mode);
    tabs[next].focus();
  });
});
fileInput.addEventListener("change", () => chooseFiles(fileInput.files));
$("#clear-files").addEventListener("click", () => { resetFiles(); renderSelection(); });
$("#width-mm").addEventListener("input", renderSelection);
$("#depth-mm").addEventListener("input", renderSelection);
$("#neural-predict").addEventListener("change", () => {
  syncNeuralControls();
  renderSelection();
});
$("#view-count").addEventListener("input", (event) => {
  $("#view-output").textContent = event.target.value;
  renderSelection();
});
$("#new-model").addEventListener("click", startAnotherModel);
buildButton.addEventListener("click", startBuild);

["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => chooseFiles(event.dataTransfer.files));

setMode("image");
loadCapabilities();

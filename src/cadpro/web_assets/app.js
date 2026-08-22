const state = {
  mode: "photos",
  files: [],
  thumbnails: new Map(),
  dimensionsValid: null,
  photoCheckProgress: null,
  selectionGeneration: 0,
  selectionController: null,
  jobId: null,
  pollTimer: null,
  syntheticProgress: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const fileInput = $("#file-input");
const dropZone = $("#drop-zone");
const buildButton = $("#build-button");
const imageTypes = new Set(["image/png", "image/jpeg", "image/webp", "image/bmp", "image/x-bmp", "image/tiff"]);
const videoTypes = new Set(["video/mp4", "video/quicktime", "video/x-msvideo", "video/avi", "video/webm", "video/x-matroska", "video/mkv", "video/x-m4v"]);
const MEBIBYTE = 1024 * 1024;
const GIBIBYTE = 1024 * MEBIBYTE;
const MAX_IMAGE_BYTES = 25 * MEBIBYTE;
const MAX_PHOTO_SET_BYTES = 500 * MEBIBYTE;
const MAX_VIDEO_BYTES = 2 * GIBIBYTE;
const THUMBNAIL_MAX_EDGE = 192;

function setMode(mode) {
  state.mode = mode;
  resetFiles();
  $$(".mode-button").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const photos = mode === "photos";
  $("#upload-title").textContent = photos ? "Add your photo orbit" : "Add your turntable video";
  $("#drop-title").textContent = photos ? "Drop 20–50 photos here" : "Drop one full-rotation video here";
  $("#drop-copy").textContent = photos
    ? "or browse files · keep them in rotation order"
    : "or browse a video · one constant-speed 360° turn";
  fileInput.accept = photos
    ? "image/png,image/jpeg,image/webp,image/bmp,image/tiff"
    : "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska";
  fileInput.multiple = photos;
  $$(".video-setting").forEach((field) => { field.hidden = photos; });
  renderSelection();
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
  const photos = mode === "photos";
  const acceptedTypes = photos ? imageTypes : videoTypes;
  const rejectSelection = (message) => {
    if (!isCurrentSelection(generation, mode)) return;
    state.selectionController = null;
    showMessage(message);
    renderSelection();
  };
  const wrongType = incoming.find((file) => file.type && !acceptedTypes.has(file.type));
  if (wrongType) {
    rejectSelection(`${wrongType.name} is not a supported ${photos ? "image" : "video"} file.`);
    return;
  }
  if (photos && incoming.length > 50) {
    rejectSelection("Use at most 50 ordered photos. Remove extra or duplicate views and try again.");
    return;
  }
  if (!photos && incoming.length !== 1) {
    rejectSelection("Choose exactly one video containing one complete 360° revolution.");
    return;
  }
  const emptyFile = incoming.find((file) => file.size === 0);
  if (emptyFile) {
    rejectSelection(`${emptyFile.name} is empty. Choose the original capture file and try again.`);
    return;
  }
  if (photos) {
    const oversized = incoming.find((file) => file.size > MAX_IMAGE_BYTES);
    if (oversized) {
      rejectSelection(`${oversized.name} exceeds the 25 MiB per-photo limit.`);
      return;
    }
    const totalBytes = incoming.reduce((sum, file) => sum + file.size, 0);
    if (totalBytes > MAX_PHOTO_SET_BYTES) {
      rejectSelection(`The photo set is ${formatFileSize(totalBytes)}. Keep the complete set at or below 500 MiB.`);
      return;
    }
  } else if (incoming[0].size > MAX_VIDEO_BYTES) {
    rejectSelection(`${incoming[0].name} exceeds the 2 GiB video limit.`);
    return;
  }

  // Keep validation's rotation snapshot immutable while the visible order remains reorderable.
  state.files = [...incoming];
  state.dimensionsValid = photos ? null : true;
  state.photoCheckProgress = photos && incoming.length ? { checked: 0, total: incoming.length } : null;
  renderSelection();
  if (photos && incoming.length) {
    await verifyPhotoDimensions(incoming, generation, mode, controller.signal);
  }
  if (isCurrentSelection(generation, mode)) state.selectionController = null;
}

function isCurrentSelection(generation, mode) {
  return generation === state.selectionGeneration && mode === state.mode;
}

async function verifyPhotoDimensions(files, generation, mode, signal) {
  try {
    let expectedDimensions = null;
    for (let index = 0; index < files.length; index += 1) {
      const result = await inspectPhoto(files[index], signal);
      if (!isCurrentSelection(generation, mode)) return;
      if (expectedDimensions === null) {
        expectedDimensions = [result.width, result.height];
      } else if (result.width !== expectedDimensions[0] || result.height !== expectedDimensions[1]) {
        state.dimensionsValid = false;
        state.photoCheckProgress = null;
        showMessage("Every photo must have the same pixel dimensions. Avoid crops, screenshots, or mixed camera modes.");
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
    showMessage("One or more photos could not be decoded. Replace damaged or unsupported files.");
  }
  renderSelection();
}

function inspectPhoto(file, signal) {
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
      const error = new Error("Photo inspection cancelled");
      error.name = "AbortError";
      finish(reject, error);
    };
    signal.addEventListener("abort", abort, { once: true });
    image.onload = () => {
      try {
        if (signal.aborted) {
          abort();
          return;
        }
        const width = image.naturalWidth;
        const height = image.naturalHeight;
        if (!width || !height) throw new Error("empty image dimensions");
        const scale = Math.min(1, THUMBNAIL_MAX_EDGE / Math.max(width, height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        const context = canvas.getContext("2d", { alpha: false });
        if (!context) throw new Error("thumbnail canvas unavailable");
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = "high";
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        const thumbnail = canvas.toDataURL("image/jpeg", 0.72);
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

function renderSelection() {
  const count = state.files.length;
  const photos = state.mode === "photos";
  const countValid = photos ? count >= 20 && count <= 50 : count === 1;
  const valid = countValid && state.dimensionsValid === true;
  const totalBytes = state.files.reduce((sum, file) => sum + file.size, 0);
  $("#file-counter").textContent = photos ? `${count} / 20 minimum` : `${count} / 1 video`;
  $("#selection").hidden = count === 0;
  $("#selection-count").textContent = photos ? `${count} ordered views loaded` : (count ? state.files[0].name : "No video loaded");
  $("#selection-size").textContent = `${(totalBytes / MEBIBYTE).toFixed(totalBytes ? 1 : 0)} MiB total`;
  if (photos && state.photoCheckProgress) {
    const next = Math.min(state.photoCheckProgress.checked + 1, state.photoCheckProgress.total);
    $("#count-check").textContent = `Checking view ${next} of ${state.photoCheckProgress.total}`;
  } else if (photos) {
    $("#count-check").textContent = countValid ? `${count} views ready` : `Add ${Math.max(0, 20 - count)} more photos`;
  } else {
    $("#count-check").textContent = countValid ? "One video ready" : "Add one turntable video";
  }
  const dots = $$(".quality-dot");
  dots[0].className = `quality-dot ${valid ? "good" : "waiting"}`;
  const widthValid = Number($("#width-mm").value) > 0;
  dots[1].className = `quality-dot ${widthValid ? "good" : "waiting"}`;
  dots[2].className = `quality-dot ${countValid ? "good" : "waiting"}`;
  buildButton.disabled = !valid || !widthValid;

  const strip = $("#thumbnail-strip");
  strip.replaceChildren();
  state.files.forEach((file, index) => {
    const item = document.createElement("div");
    item.className = "thumb";
    if (photos) {
      const thumbnail = state.thumbnails.get(file);
      if (thumbnail) item.style.backgroundImage = `url("${thumbnail}")`;
    } else {
      item.classList.add("video-thumb");
    }
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
  return `${(bytes / MEBIBYTE).toFixed(1)} MiB`;
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
    form.append("views", $("#view-count").value);
    form.append("start_frame", "0");
  }
  form.append("width_mm", $("#width-mm").value);
  form.append("clockwise", String($("#rotation-direction").value === "clockwise"));
  showProgress();
  updateProgress(8, "upload", "Uploading capture", "The reconstruction service is receiving every view.");
  try {
    const response = await fetch(`/api/jobs/${state.mode}`, { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(payload, `Upload failed (${response.status}).`));
    state.jobId = payload.id;
    state.syntheticProgress = 16;
    updateProgress(16, "segment", "Checking every silhouette", "Verifying framing, dimensions, and a clean background boundary.");
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
  const order = ["upload", "segment", "reconstruct", "export"];
  const activeIndex = Math.max(0, order.indexOf(stage));
  $$(".build-stages li").forEach((item, index) => {
    item.classList.toggle("active", index === activeIndex);
    item.classList.toggle("complete", index < activeIndex || bounded === 100);
  });
}

async function pollJob(statusUrl) {
  for (;;) {
    await new Promise((resolve) => { state.pollTimer = setTimeout(resolve, 750); });
    const response = await fetch(statusUrl, { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(payload, "Could not read reconstruction status."));
    if (payload.status === "failed") throw new Error(errorMessage(payload, "Reconstruction failed."));
    if (payload.status === "completed") {
      updateProgress(100, "export", "CAD integrity passed", "STEP, STL, and GLB exports are ready.");
      await new Promise((resolve) => setTimeout(resolve, 450));
      await renderResult(payload);
      return;
    }
    state.syntheticProgress = Math.min(91, state.syntheticProgress + (state.syntheticProgress < 55 ? 7 : 2));
    const serverProgress = Number(payload.progress);
    const value = Number.isFinite(serverProgress) ? serverProgress : state.syntheticProgress;
    const stage = payload.stage || (value < 35 ? "segment" : value < 78 ? "reconstruct" : "export");
    const labels = {
      queued: ["Waiting for the geometry engine", "Your capture is queued safely."],
      upload: ["Securing every input", "Preserving the supplied rotation order."],
      segment: ["Extracting silhouettes", "Tracing clean object boundaries across the full orbit."],
      reconstruct: ["Intersecting the visual hull", "Building and validating the closed B-rep solid."],
      export: ["Writing interoperable geometry", "Verifying STEP and preparing Blender-friendly meshes."],
    };
    updateProgress(value, stage, ...(labels[stage] || labels.reconstruct));
  }
}

async function renderResult(payload) {
  const result = payload.result || {};
  const artifacts = result.artifacts || [];
  const bySuffix = (suffix) => artifacts.find((artifact) => artifact.filename.toLowerCase().endsWith(suffix));
  const step = bySuffix(".step") || bySuffix(".stp");
  const stl = bySuffix(".stl");
  const glb = bySuffix(".glb");
  const preview = bySuffix(".preview.html") || bySuffix(".html");
  const report = bySuffix(".report.json") || bySuffix(".json");
  if (!step || !stl || !glb || !preview || !report) throw new Error("The job completed without every required export.");

  const metrics = result.metrics || await fetchReportMetrics(report.download_url);
  const dimensions = metrics.dimensions_mm || metrics.geometry?.dimensions_mm || {};
  const dimensionValues = Array.isArray(dimensions)
    ? dimensions
    : [dimensions.x, dimensions.y, dimensions.z];
  $("#metric-x").textContent = formatNumber(dimensionValues[0]);
  $("#metric-y").textContent = formatNumber(dimensionValues[1]);
  $("#metric-z").textContent = formatNumber(dimensionValues[2]);
  $("#metric-volume").textContent = formatNumber(metrics.volume_mm3 ?? metrics.geometry?.volume_mm3, 0);
  $("#metric-faces").textContent = formatNumber(metrics.face_count ?? metrics.geometry?.face_count, 0);
  $("#metric-views").textContent = String(payload.input_count || state.files.length);
  $("#model-preview").src = preview.download_url;
  setDownload("#download-step", step);
  setDownload("#download-stl", stl);
  setDownload("#download-glb", glb);
  setDownload("#download-report", report);

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
  state.syntheticProgress = 0;
  resetFiles();
  renderSelection();
  $("#result-section").hidden = true;
  $("#model-preview").src = "about:blank";
  $(".work-card").hidden = false;
  $(".mode-switch").hidden = false;
  $$(".step-rail li").forEach((item, index) => {
    item.classList.toggle("active", index === 0);
    item.classList.remove("complete");
  });
  $(".studio").scrollIntoView({ behavior: "smooth", block: "start" });
}

$$('.mode-button').forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
fileInput.addEventListener("change", () => chooseFiles(fileInput.files));
$("#clear-files").addEventListener("click", () => { resetFiles(); renderSelection(); });
$("#width-mm").addEventListener("input", (event) => {
  $("#scale-check").textContent = `${event.target.value || 0} mm maximum width`;
  renderSelection();
});
$("#view-count").addEventListener("input", (event) => { $("#view-output").textContent = event.target.value; });
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

setMode("photos");

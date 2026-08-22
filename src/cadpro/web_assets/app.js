const state = {
  file: null,
  thumbnail: null,
  imageSize: null,
  imageValid: false,
  selectionGeneration: 0,
  selectionController: null,
  jobId: null,
  pollTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const fileInput = $("#file-input");
const dropZone = $("#drop-zone");
const buildButton = $("#build-button");
const imageTypes = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/bmp",
  "image/x-bmp",
]);
const MEBIBYTE = 1024 * 1024;
const MAX_IMAGE_BYTES = 25 * MEBIBYTE;
const MAX_DIMENSION_MM = 1_000_000;
const MAX_IMAGE_EDGE = 8_192;
const MAX_IMAGE_PIXELS = 12_500_000;
const THUMBNAIL_MAX_EDGE = 320;

function resetFile() {
  state.selectionController?.abort();
  state.selectionController = null;
  state.selectionGeneration += 1;
  state.file = null;
  state.thumbnail = null;
  state.imageSize = null;
  state.imageValid = false;
  fileInput.value = "";
  showMessage("");
}

async function chooseFiles(fileList) {
  state.selectionController?.abort();
  const controller = new AbortController();
  const generation = state.selectionGeneration + 1;
  state.selectionGeneration = generation;
  state.selectionController = controller;
  state.file = null;
  state.thumbnail = null;
  state.imageSize = null;
  state.imageValid = false;
  fileInput.value = "";
  showMessage("");

  const incoming = [...fileList];
  if (incoming.length !== 1) {
    rejectSelection(generation, "Choose exactly one image.");
    return;
  }

  const file = incoming[0];
  if (file.type && !imageTypes.has(file.type)) {
    rejectSelection(generation, `${file.name} is not a supported image file.`);
    return;
  }
  if (file.size === 0) {
    rejectSelection(generation, `${file.name} is empty. Choose the original image and try again.`);
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    rejectSelection(generation, `${file.name} exceeds the 25 MiB image limit.`);
    return;
  }

  state.file = file;
  renderSelection();
  try {
    const result = await inspectImage(file, controller.signal);
    if (!isCurrentSelection(generation)) return;
    state.thumbnail = result.thumbnail;
    state.imageSize = [result.width, result.height];
    state.imageValid = true;
    state.selectionController = null;
    renderSelection();
  } catch (error) {
    if (error?.name === "AbortError" || !isCurrentSelection(generation)) return;
    state.selectionController = null;
    state.imageValid = false;
    const message = error?.name === "ImageLimitError"
      ? error.message
      : "The image could not be decoded. Use a clear PNG, JPEG, WebP, or BMP file.";
    showMessage(message);
    renderSelection();
  }
}

function rejectSelection(generation, message) {
  if (!isCurrentSelection(generation)) return;
  state.selectionController = null;
  showMessage(message);
  renderSelection();
}

function isCurrentSelection(generation) {
  return generation === state.selectionGeneration;
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
        if (signal.aborted) {
          abort();
          return;
        }
        const width = image.naturalWidth;
        const height = image.naturalHeight;
        if (!width || !height) throw new Error("empty image dimensions");
        if (width > MAX_IMAGE_EDGE || height > MAX_IMAGE_EDGE || width * height > MAX_IMAGE_PIXELS) {
          const error = new Error("Use an image with at most 12.5 million pixels and 8,192 pixels per side.");
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
        const thumbnail = canvas.toDataURL("image/jpeg", 0.78);
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

function dimensionValue(selector) {
  const value = Number($(selector).value);
  return Number.isFinite(value) && value > 0 && value <= MAX_DIMENSION_MM ? value : null;
}

function renderSelection() {
  const hasFile = state.file !== null;
  const width = dimensionValue("#width-mm");
  const depth = dimensionValue("#depth-mm");
  $("#file-counter").textContent = hasFile ? "1 / 1 image" : "0 / 1 image";
  $("#selection").hidden = !hasFile;
  $("#selection-count").textContent = hasFile ? state.file.name : "No image loaded";

  const sizeParts = [];
  if (hasFile) sizeParts.push(`${(state.file.size / MEBIBYTE).toFixed(1)} MiB`);
  if (state.imageSize) sizeParts.push(`${state.imageSize[0]} × ${state.imageSize[1]} px`);
  $("#selection-size").textContent = sizeParts.join(" · ") || "0 MiB";
  $("#image-check").textContent = state.imageValid ? "One image ready" : (hasFile ? "Checking image" : "Add one clear image");
  $("#scale-check").textContent = width ? `${formatNumber(width)} mm profile width` : "Enter a valid width";
  $("#depth-check").textContent = depth ? `${formatNumber(depth)} mm extrusion` : "Enter a valid depth";

  const dots = $$(".quality-dot");
  dots[0].className = `quality-dot ${state.imageValid ? "good" : "waiting"}`;
  dots[1].className = `quality-dot ${width ? "good" : "waiting"}`;
  dots[2].className = `quality-dot ${depth ? "good" : "waiting"}`;
  buildButton.disabled = !(state.imageValid && width && depth);

  const strip = $("#thumbnail-strip");
  strip.replaceChildren();
  if (hasFile) {
    const item = document.createElement("div");
    item.className = "thumb single-thumb";
    if (state.thumbnail) item.style.backgroundImage = `url("${state.thumbnail}")`;
    const label = document.createElement("span");
    label.textContent = "01";
    item.appendChild(label);
    item.title = state.file.name;
    strip.appendChild(item);
  }
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
  if (!state.file || !state.imageValid) return;
  showMessage("");
  buildButton.disabled = true;
  const form = new FormData();
  form.append("file", state.file, state.file.name);
  form.append("width_mm", $("#width-mm").value);
  form.append("depth_mm", $("#depth-mm").value);

  showProgress();
  updateProgress(8, "upload", "Uploading one image", "The reconstruction service is securing your source image.");
  try {
    const response = await fetch("/api/jobs/image", { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(payload, `Upload failed (${response.status}).`));
    state.jobId = payload.id;
    updateProgress(16, "segment", "Checking the outline", "Verifying framing and extracting the visible object profile.");
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
    await new Promise((resolve) => { state.pollTimer = setTimeout(resolve, 500); });
    const response = await fetch(statusUrl, { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(payload, "Could not read reconstruction status."));
    if (payload.status === "failed") throw new Error(errorMessage(payload, "Reconstruction failed."));
    if (payload.status === "completed") {
      updateProgress(100, "export", "CAD integrity passed", "STEP, STL, and GLB exports are ready.");
      await new Promise((resolve) => setTimeout(resolve, 350));
      await renderResult(payload);
      return;
    }

    const serverProgress = Number(payload.progress);
    const value = Number.isFinite(serverProgress) ? Math.max(16, serverProgress) : 16;
    const stage = payload.stage || (value < 40 ? "segment" : value < 80 ? "reconstruct" : "export");
    const labels = {
      queued: ["Waiting for the geometry engine", "Your image is queued safely."],
      upload: ["Securing the source image", "Keeping this job isolated from every other upload."],
      segment: ["Extracting the visible profile", "Tracing the object boundary and visible through-holes."],
      reconstruct: ["Extruding the CAD solid", "Applying your real width and chosen depth."],
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
  $("#metric-views").textContent = "1";
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
  resetFile();
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

fileInput.addEventListener("change", () => chooseFiles(fileInput.files));
$("#clear-files").addEventListener("click", () => { resetFile(); renderSelection(); });
$("#width-mm").addEventListener("input", renderSelection);
$("#depth-mm").addEventListener("input", renderSelection);
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

renderSelection();

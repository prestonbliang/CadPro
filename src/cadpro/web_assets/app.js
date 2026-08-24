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
  generativeMeshAvailable: false,
  textTo3dAvailable: false,
  meshyProvider: false,
  meshProvider: "AI",
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
const MAX_PROMPT_CHARS = 600;
const MIN_MESH_FACES = 100;
const MAX_MESH_FACES = 300_000;

const modeCopy = {
  text: {
    uploadTitle: "Describe a visual 3D asset",
    dropTitle: "",
    dropCopy: "",
    chip: "",
    rail: "<b>Prompt to visual mesh.</b> Describe the shape and materials. The generated asset is non-metric and is not CAD or STEP.",
  },
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
  const text = mode === "text";
  const video = mode === "video";
  fileInput.accept = video
    ? "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska"
    : "image/png,image/jpeg,image/webp,image/bmp";
  fileInput.multiple = mode === "photos";
  dropZone.hidden = text;
  $("#text-prompt-panel").hidden = !text;
  $("#calibration-grid").hidden = text;
  $("#intelligence-panel").hidden = text;
  $$(".image-setting").forEach((field) => { field.hidden = mode !== "image"; });
  $$(".neural-setting").forEach((field) => { field.hidden = mode !== "image"; });
  if (mode !== "image") $("#neural-predict").checked = false;
  $$(".orbit-setting").forEach((field) => { field.hidden = mode === "image" || text; });
  $$(".video-setting").forEach((field) => { field.hidden = !video; });
  setWorkflowCopy(text);
  syncNeuralControls();
  syncMeshSettings();
  renderSelection();
}

function setWorkflowCopy(text) {
  const rail = text
    ? [
      ["Prompt", "Describe the visual asset"],
      ["Configure", "Texture + topology"],
      ["Generate", "Geometry + materials"],
      ["Export", "Visual meshes + report"],
    ]
    : [
      ["Capture", "Photo, orbit, or video"],
      ["Calibrate", "Add one real measurement"],
      ["Reconstruct", "Geometry + optional research"],
      ["Export", "STEP + meshes + report"],
    ];
  rail.forEach(([title, copy], index) => {
    $(`#rail-step-${["one", "two", "three", "four"][index]}-title`).textContent = title;
    $(`#rail-step-${["one", "two", "three", "four"][index]}-copy`).textContent = copy;
  });
  const stageLabels = text
    ? ["Prepare prompt", "Generate mesh", "Texture + PBR", "Optimize topology", "Package exports"]
    : ["Secure capture", "Analyze + research", "Extract silhouettes", "Build solid", "Validate + export"];
  $$(".build-stages li b").forEach((label, index) => { label.textContent = stageLabels[index]; });
  $("#progress-eyebrow").textContent = text ? "VISUAL ASSET GENERATION / LIVE" : "GEOMETRY BUILD / LIVE";
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

function meshSettingsRequested() {
  return state.mode === "text"
    || (state.conceptMeshAvailable && $("#concept-mesh").checked);
}

function meshSettingsValid() {
  const targetFaces = Number($("#mesh-target-faces").value);
  const topology = $("#mesh-topology").value;
  const height = Number($("#mesh-height-m").value);
  const facesValid = Number.isInteger(targetFaces)
    && targetFaces >= MIN_MESH_FACES
    && targetFaces <= MAX_MESH_FACES;
  const heightValid = !$("#mesh-rig").checked
    || (Number.isFinite(height) && height > 0 && height <= 10);
  return facesValid && ["triangle", "quad"].includes(topology) && heightValid;
}

function syncMeshSettings() {
  const text = state.mode === "text";
  const captureMesh = !text
    && state.meshyProvider
    && state.conceptMeshAvailable
    && $("#concept-mesh").checked;
  $("#mesh-settings-panel").hidden = !(text || captureMesh);
  $("#mesh-pbr").disabled = !$("#mesh-texture").checked;
  $("#mesh-height-m").disabled = !$("#mesh-rig").checked;
  $("#mesh-settings-provider").textContent = `${state.meshProvider.toUpperCase()} VISUAL ONLY`;
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
  const text = mode === "text";
  const photos = mode === "photos";
  const video = mode === "video";
  const countValid = mode === "image"
    ? count === 1
    : photos ? count >= 20 && count <= 50 : video ? count === 1 : false;
  const captureValid = countValid && state.dimensionsValid === true;
  const width = dimensionValue("#width-mm");
  const depth = dimensionValue("#depth-mm");
  const neural = neuralPredictionSelected();
  const geometryValid = mode === "image" ? neural || depth !== null : true;
  const totalBytes = state.files.reduce((sum, file) => sum + file.size, 0);

  if (text) {
    const prompt = $("#text-prompt").value;
    const promptValid = prompt.trim().length > 0 && prompt.length <= MAX_PROMPT_CHARS;
    const settingsValid = meshSettingsValid();
    $("#file-counter").textContent = `${prompt.length} / ${MAX_PROMPT_CHARS} characters`;
    $("#text-prompt-count").textContent = `${prompt.length} / ${MAX_PROMPT_CHARS}`;
    $("#selection").hidden = true;
    $("#thumbnail-strip").replaceChildren();
    $("#capture-check-title").textContent = "Prompt";
    $("#capture-check").textContent = promptValid
      ? "Description ready"
      : "Enter 1–600 characters";
    $("#scale-check-title").textContent = "Provider";
    $("#scale-check").textContent = state.textTo3dAvailable
      ? `${state.meshProvider} text-to-3D ready`
      : "Text-to-3D is unavailable";
    $("#geometry-check-title").textContent = "Output";
    $("#geometry-check").textContent = settingsValid
      ? `${formatNumber($("#mesh-target-faces").value, 0)} target faces · non-metric mesh`
      : "Check face target and rig height";
    const dots = $$(".quality-dot");
    dots[0].className = `quality-dot ${promptValid ? "good" : "waiting"}`;
    dots[1].className = `quality-dot ${state.textTo3dAvailable ? "good" : "waiting"}`;
    dots[2].className = `quality-dot ${settingsValid ? "good" : "waiting"}`;
    buildButton.disabled = !(promptValid && state.textTo3dAvailable && settingsValid);
    $("#build-button-label").textContent = "GENERATE VISUAL ASSET";
    $("#action-note").innerHTML = "<span aria-hidden=\"true\">●</span> Text generation returns a non-metric visual mesh, never STEP or manufacturing CAD.";
    return;
  }

  $("#file-counter").textContent = mode === "image"
    ? `${count} / 1 image`
    : photos ? `${count} / 20 minimum` : `${count} / 1 video`;
  $("#selection").hidden = count === 0;
  $("#selection-count").textContent = mode === "image"
    ? (count ? state.files[0].name : "No image loaded")
    : photos ? `${count} ordered views loaded` : (count ? state.files[0].name : "No video loaded");
  $("#selection-size").textContent = `${formatFileSize(totalBytes)} total`;
  $("#capture-check-title").textContent = "Capture";
  $("#scale-check-title").textContent = "Scale";
  $("#build-button-label").textContent = "BUILD CAD MODEL";
  $("#action-note").innerHTML = "<span aria-hidden=\"true\">●</span> Uploads are isolated per job and expire automatically.";

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
  const optionalMeshValid = !meshSettingsRequested() || meshSettingsValid();
  buildButton.disabled = !(captureValid && width && geometryValid && optionalMeshValid);

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

function appendMeshSettings(form) {
  const texture = $("#mesh-texture").checked;
  form.append("mesh_texture", String(texture));
  form.append("mesh_pbr", String(texture && $("#mesh-pbr").checked));
  form.append("mesh_topology", $("#mesh-topology").value);
  form.append("mesh_target_faces", $("#mesh-target-faces").value);
  form.append("mesh_rig", String($("#mesh-rig").checked));
  form.append("mesh_height_m", $("#mesh-height-m").value);
}

async function startBuild() {
  showMessage("");
  const text = state.mode === "text";
  const prompt = text ? $("#text-prompt").value.trim() : "";
  if (text && !state.textTo3dAvailable) {
    showMessage("Text-to-3D visual generation is not available on this server.");
    renderSelection();
    return;
  }
  if (text && (!prompt || prompt.length > MAX_PROMPT_CHARS || !meshSettingsValid())) {
    showMessage("Enter a 1–600 character prompt and check the visual asset settings.");
    renderSelection();
    return;
  }
  buildButton.disabled = true;
  const form = new FormData();
  const conceptRequested = !text
    && state.conceptMeshAvailable
    && $("#concept-mesh").checked;
  if (text) {
    form.append("prompt", prompt);
    appendMeshSettings(form);
  } else if (state.mode === "photos") {
    state.files.forEach((file) => form.append("files", file, file.name));
  } else {
    form.append("file", state.files[0], state.files[0].name);
  }
  if (!text) {
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
    form.append("concept_mesh", String(conceptRequested));
    form.append("object_hint", $("#object-hint").value.trim());
    if (conceptRequested) appendMeshSettings(form);
  }

  showProgress();
  updateProgress(
    8,
    "upload",
    text ? "Submitting your prompt" : "Uploading capture",
    text
      ? "The visual-generation service is validating your description and output settings."
      : "The reconstruction service is securing every source file.",
  );
  try {
    const response = await fetch(`/api/jobs/${state.mode}`, { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(payload, `${text ? "Generation request" : "Upload"} failed (${response.status}).`));
    state.jobId = payload.id;
    const aiRequested = !text && state.aiAvailable && $("#ai-enhance").checked;
    updateProgress(
      16,
      text ? "research" : aiRequested ? "research" : "segment",
      text ? "Starting visual generation" : aiRequested ? "Analyzing the object" : "Checking every silhouette",
      text
        ? "The provider is synthesizing geometry from your prompt before texture and export stages."
        : aiRequested
          ? "Inspecting representative views and searching for cited reference specifications."
          : "Verifying framing, dimensions, and clean object boundaries.",
    );
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
  const stageGroups = state.mode === "text"
    ? [
      ["queued", "upload", "prompt"],
      ["research", "segment", "reconstruct", "generate"],
      ["texture", "texturing"],
      ["remesh", "rig", "rigging"],
      ["export", "complete"],
    ]
    : [["queued", "upload"], ["research"], ["segment"], ["reconstruct"], ["export", "complete"]];
  const matched = stageGroups.findIndex((group) => group.includes(stage));
  const activeIndex = matched >= 0 ? matched : state.mode === "text" ? 1 : 3;
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
    if (payload.status === "failed") throw new Error(errorMessage(payload, state.mode === "text" ? "Visual generation failed." : "Reconstruction failed."));
    if (payload.status === "completed") {
      updateProgress(
        100,
        "export",
        state.mode === "text" ? "Visual asset generated" : "CAD integrity passed",
        state.mode === "text"
          ? "The non-metric AI mesh, interactive preview, and provider report are ready."
          : "STEP, STL, GLB, preview, and diagnostics are ready.",
      );
      await new Promise((resolve) => setTimeout(resolve, 350));
      await renderResult(payload);
      return;
    }
    const serverProgress = Number(payload.progress);
    const value = Number.isFinite(serverProgress) ? Math.max(16, serverProgress) : 16;
    const stage = payload.stage || (value < 35 ? "segment" : value < 78 ? "reconstruct" : "export");
    const labels = state.mode === "text"
      ? {
        queued: ["Waiting for the visual generator", "Your prompt is queued for an available provider slot."],
        upload: ["Preparing the generation request", "Validating prompt and visual-output settings."],
        prompt: ["Preparing the generation request", "Validating prompt and visual-output settings."],
        research: ["Synthesizing the object", "Generating a visual mesh from the text description."],
        segment: ["Synthesizing the object", "Generating a visual mesh from the text description."],
        reconstruct: ["Refining generated geometry", "Completing the provider mesh and requested topology."],
        generate: ["Synthesizing the object", "Generating a visual mesh from the text description."],
        texture: ["Painting surface detail", "Generating textures and requested PBR material channels."],
        texturing: ["Painting surface detail", "Generating textures and requested PBR material channels."],
        remesh: ["Optimizing topology", "Working toward the requested topology and approximate face target."],
        rig: ["Building a humanoid rig", "Creating and validating the optional visual animation skeleton."],
        rigging: ["Building a humanoid rig", "Creating and validating the optional visual animation skeleton."],
        export: ["Packaging visual assets", "Preparing GLB, optional mesh exports, preview, and provider report."],
      }
      : {
        queued: ["Waiting for the geometry engine", "Your capture is queued safely."],
        upload: ["Securing every input", "Keeping this job isolated from every other upload."],
        research: ["Researching likely references", "Analyzing representative views and collecting cited specifications."],
        segment: ["Extracting silhouettes", "Tracing clean object boundaries across the supplied capture."],
        reconstruct: ["Building the CAD solid", "Creating and validating the measurement-driven B-rep geometry."],
        export: ["Writing interoperable geometry", "Round-trip checking STEP and preparing Blender-friendly meshes."],
      };
    const fallback = state.mode === "text" ? labels.generate : labels.reconstruct;
    updateProgress(value, stage, ...(labels[stage] || fallback));
  }
}

async function renderResult(payload) {
  const result = payload.result || {};
  const artifacts = result.artifacts || [];
  const filename = (artifact) => String(artifact?.filename || "").toLowerCase();
  const bySuffix = (suffix) => artifacts.find((artifact) => filename(artifact).endsWith(suffix));
  const byName = (name) => artifacts.find((artifact) => filename(artifact) === name);
  const byKind = (...kinds) => artifacts.find((artifact) => kinds.includes(
    String(artifact?.kind || artifact?.artifact_kind || "").toLowerCase(),
  ));
  const textResult = payload.kind === "text";

  $("#download-step").hidden = textResult;
  $("#download-stl").hidden = textResult;
  $("#download-glb").hidden = textResult;
  $("#download-concept").hidden = true;
  $("#download-generated").hidden = true;
  $("#download-generated-stl").hidden = true;
  $("#download-rigged").hidden = true;
  $("#research-result").hidden = true;
  $("#prediction-result").hidden = true;
  $("#optional-warning").hidden = true;
  $("#optional-warning-copy").textContent = "";

  if (textResult) {
    const generatedGlb = byName("cadpro-ai-asset.glb")
      || byKind("ai_visual_asset", "generative_mesh", "visual_asset")
      || artifacts.find((artifact) => filename(artifact).endsWith(".glb") && !filename(artifact).includes("rigged"));
    const generatedStl = byName("cadpro-ai-asset.stl")
      || byKind("ai_visual_stl", "visual_mesh_stl")
      || bySuffix(".stl");
    const riggedGlb = byName("cadpro-ai-asset.rigged.glb")
      || byKind("rigged_glb", "rigged_visual_asset")
      || artifacts.find((artifact) => filename(artifact).includes("rigged") && filename(artifact).endsWith(".glb"));
    const preview = byName("cadpro-ai-asset.preview.html")
      || byKind("visual_asset_preview", "preview")
      || bySuffix(".preview.html")
      || bySuffix(".html");
    const report = byName("cadpro-ai-asset.report.json")
      || byKind("visual_asset_report", "report")
      || bySuffix(".report.json")
      || bySuffix(".json");
    if (!generatedGlb || !preview || !report) {
      throw new Error("The visual-generation job completed without its GLB, preview, or provider report.");
    }
    $("#result-badge").hidden = true;
    $("#metric-grid").hidden = true;
    $("#result-eyebrow").textContent = "VISUAL GENERATION COMPLETE / 04";
    $("#result-title").textContent = "Your AI visual asset is ready.";
    $("#model-stage-label").innerHTML = "<span></span>NON-METRIC AI VISUAL MESH · DRAG TO ORBIT";
    $("#model-preview").title = "Interactive non-metric AI visual mesh preview";
    $("#model-preview").src = preview.download_url;
    setDownload("#download-generated", generatedGlb);
    $("#download-generated").hidden = false;
    if (generatedStl) {
      setDownload("#download-generated-stl", generatedStl);
      $("#download-generated-stl").hidden = false;
    }
    if (riggedGlb) {
      setDownload("#download-rigged", riggedGlb);
      $("#download-rigged").hidden = false;
    }
    setDownload("#download-report", report);
    $("#download-report").textContent = "Download visual-generation provider report (.json)";
    $("#truth-note").innerHTML = "<span>!</span><p><b>Non-metric AI visual mesh</b>This asset was synthesized from your text. It has no measured dimensions, CAD constraints, B-rep surfaces, tolerances, or STEP file. Inspect geometry, materials, topology, and any rig in a 3D editor before use.</p>";
  } else {
    const step = bySuffix(".step") || bySuffix(".stp");
    const stl = byName("cadpro-model.stl") || bySuffix(".stl");
    const glb = byName("cadpro-model.glb") || bySuffix(".glb");
    const conceptGlb = byName("cadpro-ai-concept.glb") || byName("cadpro-ai-asset.glb");
    const preview = byName("cadpro-model.preview.html") || bySuffix(".preview.html") || bySuffix(".html");
    const report = byName("cadpro-model.report.json") || bySuffix(".report.json") || bySuffix(".json");
    if (!step || !stl || !glb || !preview || !report) {
      throw new Error("The job completed without every required CAD export.");
    }
    $("#result-badge").hidden = false;
    $("#metric-grid").hidden = false;
    $("#result-eyebrow").textContent = "RECONSTRUCTION COMPLETE / 04";
    $("#result-title").textContent = "Your model is ready.";
    $("#model-stage-label").innerHTML = "<span></span>VALIDATED SOLID · DRAG TO ORBIT";
    $("#model-preview").title = "Interactive reconstructed model preview";
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
    $("#download-step").hidden = false;
    $("#download-stl").hidden = false;
    $("#download-glb").hidden = false;
    setDownload("#download-report", report);
    $("#download-report").textContent = "Download reconstruction + research report (.json)";
    $("#download-concept").hidden = !conceptGlb;
    if (conceptGlb) {
      setDownload("#download-concept", conceptGlb);
      $("#download-concept b").textContent = state.meshyProvider
        ? "TEXTURED AI VISUAL ASSET GLB"
        : "AI CONCEPT GLB";
    }
    $("#research-report-link").href = report.download_url;
  }

  const research = result.enrichment || result.research;
  const hasResearch = research?.status === "completed";
  $("#research-result").hidden = textResult || !hasResearch;
  if (hasResearch) {
    const identity = research.object_identity || research.identity || {};
    $("#research-object").textContent = identity.common_name || identity.name || identity.label || research.object_name || "Object research complete";
    $("#research-summary").textContent = research.summary || identity.summary || identity.evidence || "Review the cited candidate specifications and uncertainty notes before using any reference dimension.";
  }
  const prediction = result.neural_prediction;
  const hasPrediction = prediction?.status === "completed";
  $("#prediction-result").hidden = textResult || !hasPrediction;
  if (hasPrediction) {
    $("#prediction-depth").textContent = `${formatNumber(prediction.predicted_depth_mm)} mm predicted depth`;
    $("#prediction-summary").textContent = `Learned ratio ${formatNumber(prediction.predicted_depth_ratio, 4)} × measured width · heuristic confidence ${formatNumber(Number(prediction.confidence_score) * 100, 0)}%. Verify this estimate against the physical object.`;
  }
  const optionalFailures = [];
  if (!textResult && research?.status === "failed") {
    optionalFailures.push("AI/web research did not finish; the local measurement-driven CAD exports still completed.");
  }
  if (!textResult && result.concept_mesh?.status === "failed") {
    optionalFailures.push("The optional AI concept mesh did not finish; the validated STEP, STL, and deterministic GLB are unaffected.");
  }
  $("#optional-warning").hidden = optionalFailures.length === 0;
  $("#optional-warning-copy").textContent = optionalFailures.join(" ");
  const truth = $("#truth-note");
  if (textResult) {
    // The text-only truth label is set with its visual exports above.
  } else if (payload.kind === "image") {
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
  $("#text-prompt").value = "";
  renderSelection();
  $("#result-section").hidden = true;
  $("#research-result").hidden = true;
  $("#prediction-result").hidden = true;
  $("#optional-warning").hidden = true;
  $("#optional-warning-copy").textContent = "";
  $("#download-concept").hidden = true;
  $("#download-generated").hidden = true;
  $("#download-generated-stl").hidden = true;
  $("#download-rigged").hidden = true;
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
    const generativeMesh = health.generative_mesh || {};
    state.generativeMeshAvailable = generativeMesh.available === true;
    state.textTo3dAvailable = state.generativeMeshAvailable && generativeMesh.text_to_3d === true;
    const intelligence = health.intelligence || health.ai || {};
    state.aiAvailable = intelligence.available === true;
    $("#ai-enhance").disabled = !state.aiAvailable;
    const neural = health.neural_prediction || {};
    state.neuralAvailable = neural.available === true;
    $("#neural-predict").disabled = !state.neuralAvailable;
    const conceptMesh = health.concept_mesh || {};
    state.conceptMeshAvailable = conceptMesh.available === true
      || (state.generativeMeshAvailable && generativeMesh.image_to_3d === true);
    const provider = generativeMesh.provider || conceptMesh.provider || "AI";
    state.meshProvider = String(provider).trim() || "AI";
    state.meshyProvider = state.meshProvider.toLowerCase().includes("meshy");
    $("#engine-status").innerHTML = state.textTo3dAvailable
      ? "<span></span> Geometry + visual generation ready"
      : "<span></span> Geometry engine ready";
    $("#concept-mesh").disabled = !state.conceptMeshAvailable;
    if (state.meshyProvider) {
      $("#concept-mesh-title").textContent = "Textured AI visual asset";
      $("#concept-mesh-copy").textContent = "Generate a separate non-metric Meshy visual GLB from representative views, with optional textures, PBR materials, topology targeting, and humanoid rigging. It never replaces validated STEP.";
    } else {
      $("#concept-mesh-title").textContent = "High-detail AI concept mesh";
      $("#concept-mesh-copy").textContent = "Ask a separately configured image-to-3D worker for an extra GLB based on one representative view. This visual mesh is non-metric and never replaces the validated STEP model.";
    }
    const optionalAvailable = state.aiAvailable
      || state.neuralAvailable
      || state.conceptMeshAvailable
      || state.textTo3dAvailable;
    const ready = [
      state.textTo3dAvailable ? "GENERATE" : "",
      state.neuralAvailable ? "NEURAL" : "",
      state.aiAvailable ? "RESEARCH" : "",
      state.conceptMeshAvailable ? "MESH" : "",
    ].filter(Boolean);
    $("#ai-availability").textContent = ready.length ? `${ready.join(" + ")} READY` : "LOCAL MODE";
    $("#ai-availability").classList.toggle("available", optionalAvailable);
    const notes = [];
    notes.push(state.textTo3dAvailable
      ? `${state.meshProvider} text-to-3D visual generation is ready.`
      : "Text-to-3D visual generation is not configured on this server.");
    notes.push(state.neuralAvailable
      ? "A trained local depth checkpoint is ready."
      : "Train a depth model, then set CADPRO_NEURAL_ENABLED=1 and CADPRO_NEURAL_CHECKPOINT on the server.");
    notes.push(state.aiAvailable
      ? "Cited vision/web research is configured."
      : "For cited research, set CADPRO_AI_ENRICHMENT=1 and OPENAI_API_KEY on the server.");
    notes.push(state.conceptMeshAvailable
      ? `The optional non-metric ${state.meshProvider} visual-mesh provider is connected.`
      : "The validated local STEP path remains fully available without a concept-mesh worker.");
    $("#ai-note").textContent = notes.join(" ");
    syncNeuralControls();
    syncMeshSettings();
    renderSelection();
  } catch (_error) {
    state.generativeMeshAvailable = false;
    state.textTo3dAvailable = false;
    $("#engine-status").classList.add("offline");
    $("#engine-status").innerHTML = "<span></span> Geometry engine unavailable";
    $("#ai-availability").textContent = "UNAVAILABLE";
    $("#ai-note").textContent = "The server health check failed. Refresh after the geometry service is running.";
    syncMeshSettings();
    renderSelection();
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
$("#text-prompt").addEventListener("input", renderSelection);
$("#neural-predict").addEventListener("change", () => {
  syncNeuralControls();
  renderSelection();
});
$("#concept-mesh").addEventListener("change", () => {
  syncMeshSettings();
  renderSelection();
});
$("#mesh-texture").addEventListener("change", () => {
  syncMeshSettings();
  renderSelection();
});
$("#mesh-pbr").addEventListener("change", renderSelection);
$("#mesh-topology").addEventListener("change", renderSelection);
$("#mesh-target-faces").addEventListener("input", renderSelection);
$("#mesh-rig").addEventListener("change", () => {
  syncMeshSettings();
  renderSelection();
});
$("#mesh-height-m").addEventListener("input", renderSelection);
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

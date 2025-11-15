// Central UI script for the OPENCORE Analyzer. Handles prompt templates, uploads, batch flows,
// debug telemetry, token-mode routing and export utilities.

const PROMPT_TEMPLATES = [
  {
    id: "trichomes_maturity",
    label: "Trichome-Reifegrad",
    description: "Analysiert Klar/Milchig/Amber und das Erntefenster.",
    prompt:
      "Analysiere bitte die Trichome auf Klar/Milchig/Amber, beschreibe die Verteilung und schätze das optimale Erntefenster.",
  },
  {
    id: "bud_health_mold",
    label: "Bud-Health / Schimmel",
    description: "Prüft Buds auf Schimmel, Fäulnis und generelle Gesundheit.",
    prompt:
      "Untersuche das Bild auf Schimmel, Fäulnis oder andere Probleme der Bud-Gesundheit und beschreibe die Risiken.",
  },
  {
    id: "pest_detection",
    label: "Pest-Detection",
    description: "Sucht nach sichtbaren Schädlingen oder Schadmustern.",
    prompt:
      "Analysiere das Bild auf sichtbare Schädlinge oder typische Schadbilder und beschreibe die Befunde.",
  },
  {
    id: "bag_appeal",
    label: "Bag Appeal",
    description: "Bewertet Trim, Struktur, Frost und Farbspiel.",
    prompt:
      "Bewerte das visuelle Erscheinungsbild (Bag Appeal) der Buds: Trim, Struktur, Trichomdichte, Farbspiel, Gesamteindruck.",
  },
];

const STORAGE_KEYS = {
  THEME: "opencore_theme",
  TEMPLATES: "opencore_custom_templates",
  API_SETTINGS: "opencore_settings",
};

const state = {
  files: [],
  previews: [],
  customTemplates: [],
  debugEnabled: false,
  theme: "dark",
  lastResult: null,
  activeTab: "summary",
};

const dom = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheDom();
  initTheme();
  initTemplates();
  bindEvents();
  loadModels();
  applyDebugFromQuery();
  updatePreviewGrid();
  renderResultPlaceholder();
});

function cacheDom() {
  dom.form = document.getElementById("analyze-form");
  dom.modelSelect = document.getElementById("model-select");
  dom.modelStatus = document.getElementById("model-status");
  dom.prompt = document.getElementById("prompt");
  dom.templateSelect = document.getElementById("template-select");
  dom.templateList = document.getElementById("customTemplateList");
  dom.saveTemplateBtn = document.getElementById("saveTemplateBtn");
  dom.templateModal = document.getElementById("templateModal");
  dom.templateName = document.getElementById("templateName");
  dom.templateDescription = document.getElementById("templateDescription");
  dom.confirmTemplate = document.getElementById("confirmTemplate");
  dom.cancelTemplate = document.getElementById("cancelTemplate");
  dom.imageInput = document.getElementById("image");
  dom.dropzone = document.getElementById("dropzone");
  dom.previewContainer = document.getElementById("imagePreviewContainer");
  dom.runStatus = document.getElementById("runStatus");
  dom.analyzeBtn = document.getElementById("analyzeBtn");
  dom.resultTabs = document.getElementById("result-tabs");
  dom.resultDisplay = document.getElementById("result-display");
  dom.resultJson = document.getElementById("result-json");
  dom.debugPanel = document.getElementById("debugPanel");
  dom.debugToggle = document.getElementById("debugToggle");
  dom.themeToggle = document.getElementById("themeToggle");
  dom.apiSettingsBtn = document.getElementById("apiSettingsBtn");
  dom.apiModal = document.getElementById("apiModal");
  dom.apiBaseUrl = document.getElementById("apiBaseUrl");
  dom.apiToken = document.getElementById("apiToken");
  dom.apiModeEnabled = document.getElementById("apiModeEnabled");
  dom.saveApiSettings = document.getElementById("saveApiSettings");
  dom.resetApiSettings = document.getElementById("resetApiSettings");
  dom.closeApiModal = document.getElementById("closeApiModal");
  dom.jsonFullscreenBtn = document.getElementById("jsonFullscreenBtn");
  dom.jsonFullscreen = document.getElementById("jsonFullscreen");
  dom.jsonFullscreenOutput = document.getElementById("jsonFullscreenOutput");
  dom.closeJsonFullscreen = document.getElementById("closeJsonFullscreen");
  dom.jsonDownloadBtn = document.getElementById("jsonDownloadBtn");
  dom.pdfExportBtn = document.getElementById("pdfExportBtn");
  dom.shareBtn = document.getElementById("shareBtn");
  dom.toast = document.getElementById("toast");
  dom.imageModal = document.getElementById("imageModal");
  dom.modalImage = document.getElementById("modalImage");
  dom.closeImageModal = document.getElementById("closeImageModal");
  dom.zoomSlider = document.getElementById("zoomSlider");
  dom.fileStatus = document.getElementById("fileStatus");
}

function initTheme() {
  const saved = localStorage.getItem(STORAGE_KEYS.THEME);
  state.theme = saved === "light" ? "light" : "dark";
  applyTheme();
  dom.themeToggle.addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    applyTheme();
    localStorage.setItem(STORAGE_KEYS.THEME, state.theme);
  });
}

function applyTheme() {
  document.body.classList.toggle("theme-light", state.theme === "light");
}

function initTemplates() {
  const raw = localStorage.getItem(STORAGE_KEYS.TEMPLATES);
  if (raw) {
    try {
      state.customTemplates = JSON.parse(raw) ?? [];
    } catch (error) {
      state.customTemplates = [];
    }
  }
  renderTemplateOptions();
  renderCustomTemplateList();
}

function renderTemplateOptions() {
  if (!dom.templateSelect) return;
  dom.templateSelect.innerHTML = '<option value="">Template auswählen …</option>';
  PROMPT_TEMPLATES.forEach((tpl) => {
    const option = document.createElement("option");
    option.value = tpl.id;
    option.textContent = `${tpl.label} — ${tpl.description}`;
    dom.templateSelect.appendChild(option);
  });
  if (state.customTemplates.length) {
    const divider = document.createElement("option");
    divider.disabled = true;
    divider.textContent = "──── Eigene Templates ────";
    dom.templateSelect.appendChild(divider);
    state.customTemplates.forEach((tpl) => {
      const option = document.createElement("option");
      option.value = tpl.id;
      option.textContent = `${tpl.label} (Custom)`;
      dom.templateSelect.appendChild(option);
    });
  }
}

function renderCustomTemplateList() {
  if (!dom.templateList) return;
  dom.templateList.innerHTML = "";
  state.customTemplates.forEach((tpl) => {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.justifyContent = "space-between";
    row.style.gap = "0.5rem";
    const info = document.createElement("div");
    info.innerHTML = `<strong>${tpl.label}</strong><br /><small>${tpl.description || ""}</small>`;
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "pill-button";
    removeBtn.textContent = "Löschen";
    removeBtn.addEventListener("click", () => removeCustomTemplate(tpl.id));
    row.append(info, removeBtn);
    dom.templateList.appendChild(row);
  });
}

function removeCustomTemplate(id) {
  state.customTemplates = state.customTemplates.filter((tpl) => tpl.id !== id);
  persistCustomTemplates();
  renderTemplateOptions();
  renderCustomTemplateList();
  showToast("Template entfernt.");
}

function persistCustomTemplates() {
  localStorage.setItem(STORAGE_KEYS.TEMPLATES, JSON.stringify(state.customTemplates));
}

function bindEvents() {
  dom.templateSelect.addEventListener("change", handleTemplateSelect);
  dom.saveTemplateBtn.addEventListener("click", () => openModal(dom.templateModal));
  dom.cancelTemplate.addEventListener("click", () => closeModal(dom.templateModal));
  dom.confirmTemplate.addEventListener("click", saveCustomTemplate);
  dom.form.addEventListener("submit", handleAnalyzeSubmit);
  dom.imageInput.addEventListener("change", handleFileChange);
  setupDropzone();
  dom.debugToggle.addEventListener("change", () => {
    state.debugEnabled = dom.debugToggle.checked;
  });
  dom.apiSettingsBtn.addEventListener("click", openApiModal);
  dom.closeApiModal.addEventListener("click", () => closeModal(dom.apiModal));
  dom.saveApiSettings.addEventListener("click", saveApiSettings);
  dom.resetApiSettings.addEventListener("click", resetApiSettings);
  dom.jsonFullscreenBtn.addEventListener("click", openJsonFullscreen);
  dom.closeJsonFullscreen.addEventListener("click", () => closeModal(dom.jsonFullscreen));
  dom.jsonDownloadBtn.addEventListener("click", downloadJsonReport);
  dom.pdfExportBtn.addEventListener("click", generatePdfReport);
  dom.shareBtn.addEventListener("click", shareReport);
  dom.closeImageModal.addEventListener("click", () => closeModal(dom.imageModal));
  dom.zoomSlider.addEventListener("input", handleZoomChange);
  window.addEventListener("keyup", (event) => {
    if (event.key === "Escape") {
      [dom.templateModal, dom.apiModal, dom.jsonFullscreen, dom.imageModal].forEach(closeModal);
    }
  });
}

function applyDebugFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const debug = params.get("debug");
  if (debug && ["1", "true"].includes(debug.toLowerCase())) {
    dom.debugToggle.checked = true;
    state.debugEnabled = true;
  }
}

function handleTemplateSelect() {
  const value = dom.templateSelect.value;
  if (!value) return;
  const builtIn = PROMPT_TEMPLATES.find((tpl) => tpl.id === value);
  const custom = state.customTemplates.find((tpl) => tpl.id === value);
  const template = builtIn || custom;
  if (template && dom.prompt) {
    dom.prompt.value = template.prompt;
  }
}

function saveCustomTemplate() {
  const label = dom.templateName.value.trim();
  const description = dom.templateDescription.value.trim();
  const prompt = dom.prompt.value.trim();
  if (!label || !prompt) {
    showToast("Name und Prompt sind erforderlich.");
    return;
  }
  const entry = {
    id: `custom_${Date.now()}`,
    label,
    description,
    prompt,
  };
  state.customTemplates.push(entry);
  persistCustomTemplates();
  renderTemplateOptions();
  renderCustomTemplateList();
  dom.templateName.value = "";
  dom.templateDescription.value = "";
  closeModal(dom.templateModal);
  showToast("Template gespeichert.");
}

function setupDropzone() {
  ["dragenter", "dragover"].forEach((eventName) => {
    dom.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dom.dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dom.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dom.dropzone.classList.remove("dragover");
    });
  });
  dom.dropzone.addEventListener("drop", (event) => {
    const files = Array.from(event.dataTransfer.files || []);
    if (!files.length) return;
    state.files = state.files.concat(files);
    updatePreviewGrid();
  });
}

function handleFileChange(event) {
  const files = Array.from(event.target.files || []);
  state.files = files;
  updatePreviewGrid();
}

function updatePreviewGrid() {
  dom.previewContainer.innerHTML = "";
  if (!state.files.length) {
    dom.fileStatus.textContent = "Keine Dateien ausgewählt.";
    return;
  }
  dom.fileStatus.textContent = `${state.files.length} Datei(en) ausgewählt.`;
  state.files.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const item = document.createElement("div");
      item.className = "preview-item";
      item.innerHTML = `<img src="${event.target.result}" alt="Preview" /><small>${file.name || `Bild ${index + 1}`}</small>`;
      item.addEventListener("click", () => openImageModal(event.target.result));
      dom.previewContainer.appendChild(item);
    };
    reader.readAsDataURL(file);
  });
}

function openImageModal(src) {
  dom.modalImage.src = src;
  dom.zoomSlider.value = "100";
  dom.modalImage.style.transform = "scale(1)";
  openModal(dom.imageModal);
}

function handleZoomChange() {
  const scale = Number(dom.zoomSlider.value) / 100;
  dom.modalImage.style.transform = `scale(${scale})`;
}

function loadModels() {
  if (!dom.modelSelect) return;
  dom.modelSelect.disabled = true;
  dom.modelSelect.innerHTML = "<option>Modelle werden geladen …</option>";
  apiRequest("/tm-models")
    .then((payload) => {
      const models = payload.models || [];
      const hasBuiltin = payload.has_builtin;
      const defaultId = payload.default_model_id;
      dom.modelSelect.innerHTML = "";
      if (hasBuiltin) {
        const option = document.createElement("option");
        option.value = "builtin";
        option.textContent = "OPENCORE Referenz (TEACHABLE_MODEL_PATH)";
        dom.modelSelect.appendChild(option);
      }
      models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = `${model.name} (${model.type})`;
        dom.modelSelect.appendChild(option);
      });
      if (defaultId && models.some((model) => model.id === defaultId)) {
        dom.modelSelect.value = defaultId;
      } else if (hasBuiltin) {
        dom.modelSelect.value = "builtin";
      } else if (models.length) {
        dom.modelSelect.value = models[0].id;
      }
      dom.modelStatus.textContent = "Die Auswahl bestimmt das aktive Teachable-Machine-Paket.";
    })
    .catch((error) => {
      dom.modelStatus.textContent = `Modelle konnten nicht geladen werden (${error.message}).`;
    })
    .finally(() => {
      dom.modelSelect.disabled = false;
    });
}

async function handleAnalyzeSubmit(event) {
  event.preventDefault();
  if (!dom.prompt.value.trim()) {
    showToast("Prompt ist erforderlich.");
    return;
  }
  if (!state.files.length) {
    showToast("Bitte mindestens ein Bild hinzufügen.");
    return;
  }
  dom.runStatus.textContent = "Analyse läuft …";
  dom.analyzeBtn.disabled = true;
  const formData = new FormData();
  formData.append("prompt", dom.prompt.value.trim());
  const modelValue = dom.modelSelect.value;
  if (modelValue) {
    formData.append("model_id", modelValue);
  }
  if (state.debugEnabled) {
    formData.append("debug", "1");
  }
  const endpoint = state.files.length > 1 ? "/api/opencore/analyze-batch" : "/analyze";
  if (state.files.length > 1) {
    state.files.forEach((file) => formData.append("files[]", file));
  } else {
    formData.append("image", state.files[0]);
  }
  const clientStart = performance.now();
  try {
    const payload = await apiRequest(endpoint, { method: "POST", body: formData });
    const elapsed = Math.round(performance.now() - clientStart);
    handleResult(payload, elapsed);
    dom.runStatus.textContent = `Fertig (${elapsed} ms)`;
  } catch (error) {
    dom.runStatus.textContent = "Fehler";
    dom.resultDisplay.innerHTML = `<p class="disclaimer">Fehler: ${error.message}</p>`;
    dom.resultJson.textContent = JSON.stringify({ status: "error", message: error.message }, null, 2);
    dom.debugPanel.classList.add("active");
    dom.debugPanel.textContent = `Client-Fehler: ${error.message}`;
  } finally {
    dom.analyzeBtn.disabled = false;
  }
}

function handleResult(payload, elapsedMs) {
  const normalized = normalizeResult(payload);
  state.lastResult = normalized;
  renderResultTabs(normalized);
  renderJson(normalized);
  renderDebug(normalized.debug, elapsedMs);
}

function normalizeResult(payload) {
  if (payload && Array.isArray(payload.items)) {
    return payload;
  }
  const wrapper = {
    status: payload.status || "ok",
    summary: { text: payload.summary?.text || payload.gpt_response || "Report erstellt." },
    items: [
      {
        image_id: state.files[0]?.name || "upload",
        analysis: payload,
      },
    ],
    teachable_model: payload.teachable_model || payload.items?.[0]?.analysis?.teachable_model,
    debug: payload.debug,
  };
  return wrapper;
}

function renderResultTabs(result) {
  dom.resultTabs.innerHTML = "";
  const tabs = [];
  tabs.push({ key: "summary", label: "Gesamt-Report" });
  result.items.forEach((item, index) => {
    tabs.push({ key: `item-${index}`, label: item.image_id || `Bild ${index + 1}` });
  });
  tabs.forEach((tab, index) => {
    const button = document.createElement("button");
    button.textContent = tab.label;
    button.classList.toggle("active", index === 0);
    button.addEventListener("click", () => {
      state.activeTab = tab.key;
      document.querySelectorAll(".tab-bar button").forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");
      renderTabContent(result, tab.key);
    });
    dom.resultTabs.appendChild(button);
  });
  state.activeTab = "summary";
  renderTabContent(result, "summary");
}

function renderTabContent(result, key) {
  if (key === "summary") {
    dom.resultDisplay.innerHTML = `<p>${result.summary?.text || "Keine Zusammenfassung vorhanden."}</p>`;
  } else {
    const index = Number(key.split("-")[1]);
    const item = result.items[index];
    if (!item) {
      dom.resultDisplay.innerHTML = `<p class="disclaimer">Kein Eintrag gefunden.</p>`;
    } else {
      dom.resultDisplay.innerHTML = `
        <h3>${item.image_id}</h3>
        <p><strong>Modell:</strong> ${item.analysis?.teachable_model?.name || "OPENCORE"}</p>
        <pre style="white-space: pre-wrap; font-family: 'Space Mono', monospace;">${
          item.analysis?.gpt_response || ""
        }</pre>
      `;
    }
  }
}

function renderJson(result) {
  dom.resultJson.textContent = JSON.stringify(result, null, 2);
}

function renderDebug(debug, clientMs) {
  if (!debug && !state.debugEnabled) {
    dom.debugPanel.classList.remove("active");
    dom.debugPanel.textContent = "";
    return;
  }
  dom.debugPanel.classList.add("active");
  const timings = debug?.timings || {};
  dom.debugPanel.innerHTML = `
    <strong>Request-ID:</strong> ${debug?.request_id || "n/a"}<br />
    <strong>Modell:</strong> ${debug?.model_name || "OPENCORE"} (${debug?.model_version || ""})<br />
    <strong>Timings:</strong> Modell ${timings.model_ms || "?"} ms · LLM ${timings.llm_ms || "?"} ms · Gesamt ${timings.total_ms ||
    "?"} ms<br />
    <strong>Client:</strong> ${clientMs || "?"} ms<br />
    <strong>Prompt:</strong> ${debug?.prompt_preview || "-"}
  `;
}

function openModal(element) {
  element.classList.remove("hidden");
}

function closeModal(element) {
  element.classList.add("hidden");
}

function openApiModal() {
  const settings = loadApiSettings();
  dom.apiBaseUrl.value = settings.baseUrl || "";
  dom.apiToken.value = settings.token || "";
  dom.apiModeEnabled.checked = Boolean(settings.enabled);
  openModal(dom.apiModal);
}

function loadApiSettings() {
  const raw = localStorage.getItem(STORAGE_KEYS.API_SETTINGS);
  if (!raw) return { enabled: false };
  try {
    return JSON.parse(raw) ?? { enabled: false };
  } catch (error) {
    return { enabled: false };
  }
}

function saveApiSettings() {
  const settings = {
    baseUrl: dom.apiBaseUrl.value.trim().replace(/\/$/, ""),
    token: dom.apiToken.value.trim(),
    enabled: dom.apiModeEnabled.checked,
  };
  localStorage.setItem(STORAGE_KEYS.API_SETTINGS, JSON.stringify(settings));
  closeModal(dom.apiModal);
  showToast("API-Einstellungen gespeichert.");
}

function resetApiSettings() {
  localStorage.removeItem(STORAGE_KEYS.API_SETTINGS);
  dom.apiBaseUrl.value = "";
  dom.apiToken.value = "";
  dom.apiModeEnabled.checked = false;
  showToast("API-Einstellungen zurückgesetzt.");
}

function openJsonFullscreen() {
  if (!state.lastResult) {
    showToast("Keine Analyse vorhanden.");
    return;
  }
  dom.jsonFullscreenOutput.textContent = JSON.stringify(state.lastResult, null, 2);
  openModal(dom.jsonFullscreen);
}

function downloadJsonReport() {
  if (!state.lastResult) {
    showToast("Keine Analyse vorhanden.");
    return;
  }
  const blob = new Blob([JSON.stringify(state.lastResult, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "opencore-report.json";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function generatePdfReport() {
  if (!state.lastResult) {
    showToast("Keine Analyse vorhanden.");
    return;
  }
  const { jsPDF } = window.jspdf || {};
  if (!jsPDF) {
    showToast("PDF-Bibliothek nicht geladen.");
    return;
  }
  const doc = new jsPDF();
  doc.setFont("helvetica", "");
  doc.setFontSize(14);
  doc.text("OPENCORE Analyzer Report", 14, 20);
  doc.setFontSize(11);
  doc.text(`Datum: ${new Date().toLocaleString()}`, 14, 30);
  doc.text(`Modell: ${state.lastResult.teachable_model?.name || "OPENCORE"}`, 14, 38);
  doc.setFontSize(12);
  doc.text("Summary:", 14, 48);
  doc.setFontSize(11);
  const summary = doc.splitTextToSize(state.lastResult.summary?.text || "", 180);
  doc.text(summary, 14, 56);
  let offset = 56 + summary.length * 6;
  state.lastResult.items.forEach((item, idx) => {
    doc.setFontSize(12);
    doc.text(`Bild ${idx + 1}: ${item.image_id}`, 14, offset);
    offset += 6;
    doc.setFontSize(10);
    const text = doc.splitTextToSize(item.analysis?.gpt_response || "", 180);
    doc.text(text, 14, offset);
    offset += text.length * 5 + 4;
    if (offset > 260) {
      doc.addPage();
      offset = 20;
    }
  });
  doc.save("opencore-report.pdf");
}

async function shareReport() {
  if (!state.lastResult) {
    showToast("Keine Analyse vorhanden.");
    return;
  }
  try {
    const payload = await apiRequest("/api/opencore/share", {
      method: "POST",
      body: { payload: state.lastResult },
    });
    const url = `${window.location.origin}${payload.url}`;
    await navigator.clipboard?.writeText(url).catch(() => {});
    showToast(`Share-Link erstellt: ${url}`);
  } catch (error) {
    showToast(`Share fehlgeschlagen: ${error.message}`);
  }
}

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("active");
  setTimeout(() => dom.toast.classList.remove("active"), 3200);
}

async function apiRequest(path, options = {}) {
  const settings = loadApiSettings();
  const headers = options.headers ? { ...options.headers } : {};
  const isFormData = options.body instanceof FormData;
  let url = path;
  if (settings.enabled && settings.baseUrl && !path.startsWith("http")) {
    url = `${settings.baseUrl}${path}`;
  }
  const fetchOptions = {
    method: options.method || "GET",
    headers,
    body: undefined,
  };
  if (settings.enabled && settings.token) {
    headers.Authorization = `Bearer ${settings.token}`;
  }
  if (options.body) {
    if (isFormData) {
      fetchOptions.body = options.body;
    } else {
      headers["Content-Type"] = "application/json";
      fetchOptions.body = JSON.stringify(options.body);
    }
  }
  const response = await fetch(url, fetchOptions);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  if (response.headers.get("content-type")?.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

function renderResultPlaceholder() {
  dom.resultDisplay.innerHTML = `<p class="disclaimer">Noch keine Analyse durchgeführt.</p>`;
}

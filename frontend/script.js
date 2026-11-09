/* ═══════════════════════════════════════════════════════════════════════
   Edge-AI Plant Disease Diagnosis — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════ */

const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("fileInput");
const previewArea = document.getElementById("previewArea");
const previewImg  = document.getElementById("previewImg");
const clearBtn    = document.getElementById("clearBtn");
const predictBtn  = document.getElementById("predictBtn");
const resultCard  = document.getElementById("resultCard");
const loader      = document.getElementById("loader");

let selectedFile = null;

/* ── Drag & Drop + Click ─────────────────────────────────────────────── */
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    handleFile(file);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── File Handling ───────────────────────────────────────────────────── */
function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewArea.classList.remove("hidden");
    dropzone.classList.add("hidden");
    predictBtn.disabled = false;
    resultCard.classList.add("hidden");  // hide old results
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewArea.classList.add("hidden");
  dropzone.classList.remove("hidden");
  predictBtn.disabled = true;
  resultCard.classList.add("hidden");
});

/* ── Prediction ──────────────────────────────────────────────────────── */
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  // Show loader
  loader.classList.remove("hidden");
  predictBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    const data = await response.json();
    displayResults(data);
  } catch (err) {
    alert("Prediction failed: " + err.message);
    console.error(err);
  } finally {
    loader.classList.add("hidden");
    predictBtn.disabled = false;
  }
});

/* ── Display Results ─────────────────────────────────────────────────── */
function displayResults(data) {
  const { predictions, top_class, top_confidence, is_healthy } = data;

  // Status badge
  const statusEl = document.getElementById("resultStatus");
  statusEl.textContent = is_healthy ? "✓ Healthy" : "⚠ Disease Detected";
  statusEl.className = "result-status " + (is_healthy ? "healthy" : "diseased");

  // Top class & confidence
  document.getElementById("topClass").textContent = top_class;
  document.getElementById("confidenceText").textContent =
    `Confidence: ${top_confidence}%`;

  // Animate confidence bar
  const bar = document.getElementById("confidenceBar");
  bar.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      bar.style.width = top_confidence + "%";
    });
  });

  // Top-5 list
  const listEl = document.getElementById("top5List");
  listEl.innerHTML = "";

  predictions.forEach((pred, i) => {
    const item = document.createElement("div");
    item.className = "top5-item";
    item.innerHTML = `
      <span class="top5-rank">#${i + 1}</span>
      <span class="top5-name">${pred.class}</span>
      <div class="top5-bar-wrap">
        <div class="top5-bar" style="width: 0%"></div>
      </div>
      <span class="top5-pct">${pred.confidence}%</span>
    `;
    listEl.appendChild(item);

    // Animate bar
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        item.querySelector(".top5-bar").style.width = pred.confidence + "%";
      });
    });
  });

  // Show result card with scroll
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}


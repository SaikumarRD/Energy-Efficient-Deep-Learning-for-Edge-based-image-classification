# 🌿 TerraLeaf Studio: Energy-Efficient Edge-AI Plant Pathology Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16%2B-005CED.svg)](https://onnxruntime.ai/)
[![Edge AI](https://img.shields.io/badge/Domain-Edge_AI_%26_IoT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Project_Complete_%E2%9C%93-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

> **Academic / Capstone Project**: Research and implementation of model compression techniques (Structured Pruning, Knowledge Distillation, and Post-Training INT8 Quantization) to deploy state-of-the-art plant disease classification models onto resource-constrained Edge and IoT devices.

---

## 📌 Executive Summary & Key Highlights

- **⭐ Champion Architecture (`KD_MobileNetV3_Small`)**:
  - **15.0× Footprint Reduction**: Shrunk from **90.16 MB** (ResNet-50) down to **6.01 MB**.
  - **~9.5× Inference Speedup**: Slashed CPU latency from **94.91 ms** to **10.02 ms** (~100 FPS throughput on 4-thread CPU).
  - **9.0× Energy Efficiency**: Reduced energy consumption from **0.545 Wh** to **0.060 Wh** per 500 inferences.
  - **Accuracy Retained**: Retained **99.66% Top-1 Accuracy** and **99.66% Macro-F1** across 23 plant disease classes (only 0.17% deviation from heavy baseline teacher).
- **🔬 Interactive Research Presentation Studio (`/report`)**:
  - Master unified comparison charts: Pareto frontier (Accuracy vs. Latency bubble chart), Dual-Axis latency & energy graph, and Model Size vs. Computational Complexity (GMACs / FLOPs).
  - Individual 5-axis radar telemetry for all 5 architectures (*Diagnostic Accuracy*, *Memory Compactness*, *Inference Speed*, *Energy Efficiency*, *Edge Viability*).
  - Budget decision framework for agricultural deployment (Drone Flight, Raspberry Pi 4/5, Edge Gateway).
  - Print & PDF presentation export.
- **🚁 Autonomous Drone & Field Multi-Image Survey**:
  - Batch image staging & drag-and-drop ingestion with 1-click drone flight specimens.
  - High-throughput parallel inference (~100 FPS) generating canopy health indices and outbreak distribution metrics.
  - Interactive grid gallery with localized Kannada labels and click-to-inspect transition to single-leaf focus.
- **🔍 Single Leaf Focus & Clinical Pathology**:
  - Clean uncropped viewport (`object-fit: contain`).
  - Dynamic AI lesion detection markers revealed only post-analysis (`①`, `②`, `③` numbered pinpoints with glassmorphic tooltips).
  - Minimalist Clinical Pathology Summary cards (*Visual Hallmarks*, *Target Tissue*, *Disease Driver*, *Agronomic Threat*).
  - Localized **Kannada (ಕನ್ನಡ)** agricultural translations for rural Karnataka farmers.
- **🌓 Dark / Light Mode Toggle**: Clean, modern, distraction-free solid theme with persistent preference storage.
- **🖥️ 1-Click Desktop Launcher**: Direct desktop shortcut and automated launcher script (`run_app.bat`).

---

## 🔬 Empirical Research Benchmark

Modern deep learning architectures (e.g., ResNet-50) achieve exceptional accuracy on agricultural disease diagnosis tasks. However, deploying them directly to rural or remote agricultural IoT environments faces major bottlenecks:
- **Compute & Memory Limitations**: Edge hardware (microcontrollers, Raspberry Pi, drones) have strict RAM and storage limits.
- **Latency Constraints**: Farmers and agricultural drones require rapid, on-device inference without depending on high-bandwidth cloud connectivity.
- **Energy Footprint**: Battery-operated edge nodes require minimal energy consumption per inference.

| Model Architecture | Compression Strategy | Top-1 Acc (%) | Macro F1 (%) | Model Size | Parameters | Complexity (GMACs) | Edge CPU Latency | Energy (Wh / 500 inf) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline ResNet-50** | Full Precision (FP32) | 99.83% | 99.75% | 90.16 MB | 23.56 M | 4.132 | 94.91 ms | 0.5449 Wh |
| **2. Pruned ResNet-50** | 25% Channel Pruning | 99.47% | 99.38% | 50.93 MB | 13.28 M | 2.397 | 49.82 ms | 0.2824 Wh |
| **3. Quantized ResNet-50** | Post-Training Dynamic INT8 | 98.68% | 98.73% | 22.62 MB | 23.56 M | 4.132 | 53.10 ms | 0.3626 Wh |
| **4. KD MobileNetV3-Small** ⭐ | **Knowledge Distillation (FP32)** | **99.66%** | **99.66%** | **6.01 MB** | **1.54 M** | **0.061** | **10.02 ms** | **0.0602 Wh** |
| **5. KD MobileNetV3 + INT8** | Distillation + Dynamic INT8 | 9.15% | 3.24% | 1.63 MB | 1.54 M | 0.061 | 11.73 ms | 0.0886 Wh |

### 💡 Core Research Insights:
1. **Knowledge Distillation Outperformed Standalone Quantization & Pruning**: Distilling ResNet-50 into MobileNetV3-Small produced the best Pareto trade-off, achieving a **15× compression factor** and **9.5× CPU speedup** with virtually zero accuracy sacrifice.
2. **Pruning Halves Latency on Dense Architectures**: 25% structured channel pruning cut ResNet-50 inference time in half (~95 ms down to ~50 ms) while preserving 99.47% accuracy.
3. **PTQ Collapse on Ultra-Small Architectures**: While dynamic INT8 quantization compressed ResNet-50 by 4× without issue, applying Post-Training Quantization (PTQ) to lightweight depthwise-separable convolutions caused severe accuracy collapse (9.15%), proving that Quantization-Aware Training (QAT) is required when quantizing sub-2M parameter models.

---

## 📂 Repository Layout

```text
├── app.py                               # FastAPI backend with multi-model switcher & batch API
├── baseline_and_benchmark.py            # Baseline ResNet-50 training & benchmark pipeline
├── step3_pruning.py                     # Structured L2 channel pruning implementation
├── step4_distillation.py                # Teacher-Student distillation training loop
├── step5_quantization_and_comparison.py # ONNX INT8 dynamic quantization & evaluator
├── dataset_setup.py                     # Deterministic data loaders & preprocessing pipeline
├── run_app.bat                          # 1-Click launcher script for Windows
├── final_research_comparison.csv        # Empirical metric comparison export
├── kd_mobilenetv3_fp32.onnx             # Champion distilled student model (~6 MB)
├── baseline_resnet50_fp32.onnx          # Baseline teacher ONNX model (~90 MB)
├── quantized_resnet50_int8.onnx         # 4x compressed INT8 ResNet-50 (~22 MB)
├── quantized_kd_mobilenetv3_int8.onnx   # Quantized student artifact (~1.6 MB)
├── frontend/                            # Full-Stack Web Studio
│   ├── index.html                       # Single leaf focus & Drone multi-image survey studio
│   └── report.html                      # Interactive presentation & Chart.js research report
├── requirements.txt                     # Pinned project dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Clone the repository and initialize the Python environment:

```bash
git clone https://github.com/SaikumarRD/Energy-Efficient-Deep-Learning-for-Edge-based-image-classification.git
cd Energy-Efficient-Deep-Learning-for-Edge-based-image-classification

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Studio (FastAPI + HTML5/JS)

Start the local server:
```bash
uvicorn app:app --reload --port 8000
```
Open your browser to:
- **Leaf Inspection & Drone Survey Studio**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Research Report**: [http://localhost:8000/report](http://localhost:8000/report)

*(On Windows, you can also double-click `run_app.bat` or the desktop shortcut `TerraLeaf Studio` to start the app and open the browser automatically).*

---

## 🌾 Platform Features Overview

### 1. 🔍 Single Leaf Focus Mode
- **Uncropped Image Chamber**: Full view without edge clipping.
- **Dynamic Visual Pinpoints**: Numbered lesion pins (`①`, `②`, `③`) appear only after laser scanning completes, equipped with glassmorphic hover/click tooltips.
- **Clinical Pathology Summary**: Concise visual hallmarks, target foliar tissue, environmental drivers, and agronomic threat.
- **Bilingual Accessibility**: Kannada (ಕನ್ನಡ) labels alongside English scientific names.

### 2. 🚁 Drone Multi-Image Survey Mode
- **Batch Processing**: Drag and drop multi-frame flight folders or click `⚡ Load 6 Drone Flight Specimens`.
- **High-Throughput Telemetry**: Process batches at ~100 FPS on CPU, reporting aggregate Canopy Health Index, Outbreak Severity, and per-leaf latency.
- **Direct Chamber Drill-down**: Click `Inspect in Chamber →` on any card in the batch to load it into the Single Leaf chamber for granular pinpoint inspection.

### 3. 📊 Academic Presentation Studio (`/report`)
- **Interactive Chart.js Telemetry**: Switch between Pareto Bubble Frontier, Dual-Axis Latency/Energy, and Size/FLOPs bar charts.
- **5-Axis Radar Profiles**: View individual radar signatures for each model architecture.
- **Hardware Budget Framework**: Clear trade-off recommendations for drone payloads, edge micro-controllers, and gateways.
- **1-Click PDF / Print Export**: Formatted printable report for academic review.

---

## 📦 Model Weights & Releases

The lightweight production model (`kd_mobilenetv3_fp32.onnx`, **6.01 MB**) and quantized models are stored directly in this repository for zero-setup execution. 

Full PyTorch training checkpoints (`.pth`) and standalone ONNX binaries are also available on the [GitHub Releases Page](https://github.com/SaikumarRD/Energy-Efficient-Deep-Learning-for-Edge-based-image-classification/releases).

---

## 💻 Standalone Edge Deployment (No PyTorch Needed)

To run inference on an edge device (e.g., Raspberry Pi 4/5 or edge VM), only ONNX Runtime is required:

```bash
pip install onnxruntime numpy Pillow
```

Run single-leaf diagnosis in ~10 ms:
```python
import onnxruntime as ort, numpy as np
from PIL import Image

sess = ort.InferenceSession("kd_mobilenetv3_fp32.onnx", providers=["CPUExecutionProvider"])
img = Image.open("sample_leaf.jpg").convert("RGB").resize((224, 224))
arr = (np.array(img, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
tensor = np.expand_dims(np.transpose(arr, (2, 0, 1)), 0)

logits = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
probs = np.exp(logits) / np.sum(np.exp(logits))
print(f"Top Class Index: {np.argmax(probs)} ({np.max(probs)*100:.1f}%)")
```

---

## 📜 Authors & Citation
- **Project**: *Energy-Efficient Deep Learning for Edge-Based Image Classification*
- **Stack**: PyTorch, ONNX Runtime, FastAPI, Chart.js, CodeCarbon
- **License**: MIT

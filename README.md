# 🌿 Edge-AI Plant Disease Diagnosis: Edge Optimization & Model Compression

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16%2B-005CED.svg)](https://onnxruntime.ai/)
[![Edge AI](https://img.shields.io/badge/Domain-Edge_AI_%26_IoT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Backend_%26_Optimization_Complete-success.svg)]()

> **Academic / Capstone Project**: Research and implementation of model compression techniques (Structured Pruning, Knowledge Distillation, and Post-Training INT8 Quantization) to deploy state-of-the-art plant disease classification models onto resource-constrained Edge and IoT devices.

---

## 📌 Project Status

- [x] **Backend & Data Pipeline**: Deterministic dataset preprocessing, augmentation, and data loaders.
- [x] **Baseline Model**: Full-precision ResNet-50 trained and evaluated.
- [x] **Structured Channel Pruning**: 25% channel pruning on ResNet-50.
- [x] **Knowledge Distillation**: Teacher-Student distillation transferring representations from ResNet-50 into MobileNetV3-Small.
- [x] **Quantization & ONNX Export**: Post-Training INT8 Quantization for Edge CPU deployment.
- [x] **Comparative Research Benchmarking**: Evaluation across Accuracy, F1-Score, Parameters, FLOPs/GMACs, Edge CPU Latency, and Energy Consumption (CodeCarbon).
- [x] **Interactive Web UI & REST API**: Full-stack application (FastAPI + HTML5/CSS3/JavaScript) serving the compressed ONNX model (`KD_MobileNetV3`) for real-time edge leaf diagnosis.
- [ ] **Edge Hardware Deployment** *(Planned)*: Testing on Raspberry Pi / Intel OpenVINO edge kit.

---

## 🎯 Motivation & Problem Statement

Modern deep learning architectures (e.g., ResNet-50) achieve exceptional accuracy on agricultural disease diagnosis tasks. However, deploying them directly to rural or remote agricultural IoT environments faces major bottlenecks:
- **Compute & Memory Limitations**: Edge hardware (microcontrollers, Raspberry Pi, drones) have strict RAM and storage limits.
- **Latency Constraints**: Farmers and agricultural drones require rapid, on-device inference without depending on high-bandwidth cloud connectivity.
- **Energy Footprint**: Battery-operated edge nodes require minimal energy consumption per inference.

This project addresses these challenges by systematically compressing and benchmarking deep vision architectures using **Structured Pruning**, **Knowledge Distillation**, and **INT8 Quantization**.

---

## 🔬 Experimental Results & Research Comparison

All models were evaluated under identical edge CPU runtime conditions with locked seeds (`seed=42`) on the Plant Disease Dataset:

| Model | Top-1 Acc (%) | Macro F1 (%) | Size (MB) | Params (M) | GMACs | Edge CPU Latency (ms/img) | Energy (Wh / 500 inf) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline ResNet50 (FP32)** | 99.83% | 99.75% | 90.16 MB | 23.56 M | 4.132 | 94.91 ms | 0.5449 Wh |
| **2. Pruned ResNet50 (25% Ch)** | 99.47% | 99.38% | 50.93 MB | 13.28 M | 2.397 | 49.82 ms | 0.2824 Wh |
| **3. Quantized ResNet50 (INT8)** | 98.68% | 98.73% | 22.62 MB | 23.56 M | 4.132 | 53.10 ms | 0.3626 Wh |
| **4. KD MobileNetV3-Small (FP32)** ⭐ | **99.66%** | **99.66%** | **6.01 MB** | **1.54 M** | **0.061** | **10.02 ms** | **0.0602 Wh** |
| **5. KD MobileNetV3 + INT8** | 9.15% | 3.24% | 1.63 MB | 1.54 M | 0.061 | 11.73 ms | 0.0886 Wh |

### 💡 Key Findings
1. **Knowledge Distillation is the Star Technique**: `KD_MobileNetV3_Small` delivers a **15x reduction in storage size** (6.01 MB vs 90.16 MB), **~9.5x latency speedup** (10.02 ms vs 94.91 ms), and **9x energy savings**, while sacrificing only **0.17%** top-1 accuracy!
2. **Pruning Cuts Latency in Half**: 25% channel pruning successfully slashed inference time from ~95 ms to ~50 ms on CPU with negligible accuracy drop (< 0.4%).
3. **Quantization Nuance on Lightweight Models**: While Post-Training Quantization (PTQ) compressed ResNet-50 smoothly down to 22.62 MB, aggressive PTQ on small depthwise-separable architectures (MobileNetV3) led to severe quantization noise, indicating that Quantization-Aware Training (QAT) is required for ultra-small models.

---

## 📂 Repository Structure

```text
├── app.py                               # FastAPI backend serving ONNX inference
├── frontend/                            # Web UI (HTML5, modern CSS3, vanilla JS)
│   ├── index.html                       # Dashboard layout & disease diagnosis UI
│   ├── style.css                        # Modern dark-mode responsive styling
│   └── script.js                        # Drag-and-drop & API integration
├── dataset_setup.py                     # Data loaders, augmentations & seed reproducibility
├── baseline_and_benchmark.py            # Baseline ResNet-50 training & benchmark pipeline
├── step3_pruning.py                     # Structured channel pruning implementation
├── step4_distillation.py                # Teacher (ResNet50) -> Student (MobileNetV3) distillation
├── step5_quantization_and_comparison.py # ONNX INT8 dynamic/static quantization & evaluation
├── final_research_comparison.csv        # Comprehensive benchmark metrics table
├── requirements.txt                     # Pinned dependencies
├── .gitignore                           # Git ignore rules for checkpoints & cache
└── README.md                            # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
Ensure the Plant Disease dataset is placed in `./plant_disease_data` or configure the Hugging Face dataset source in `dataset_setup.py`.

```bash
python dataset_setup.py
```

### 3. Running the Pipeline
Each stage of the optimization pipeline can be executed sequentially:

- **Train/Benchmark Baseline**:
  ```bash
  python baseline_and_benchmark.py
  ```
- **Apply Structured Pruning**:
  ```bash
  python step3_pruning.py
  ```
- **Train Student via Knowledge Distillation**:
  ```bash
  python step4_distillation.py
  ```
- **Perform Quantization & Generate Research Comparison**:
  ```bash
  python step5_quantization_and_comparison.py
  ```

The comparative metrics will be saved in `final_research_comparison.csv`.

### 4. Running the Web Application
To launch the interactive leaf diagnosis web application powered by FastAPI and the compressed ONNX model:

```bash
uvicorn app:app --reload --port 8000
```
Then open your browser and navigate to **`http://localhost:8000`**.

---

## 🗺️ Roadmap & Next Steps

- [ ] **Phase 1: Web Interface**: Build an intuitive Streamlit / Gradio frontend for real-time leaf image diagnosis and disease confidence scores.
- [ ] **Phase 2: REST API**: Wrap the ONNX runtime model in a lightweight FastAPI microservice.
- [ ] **Phase 3: Hardware In-the-Loop Validation**: Deploy and record live power/thermal metrics on an edge device (e.g., Raspberry Pi 4 / Coral Edge TPU).

---

## 📜 Model Weights Note
Due to file size limits on GitHub, raw `.pth` and `.onnx` model weights are excluded from Git version control. Trained checkpoints and ONNX artifacts can be regenerated using the reproduction scripts above, or downloaded from the upcoming GitHub Releases.

---

## 🤝 Authors & Acknowledgments
- Developed as part of **Module 3 Project (AI & IoT)**.
- Built using PyTorch, ONNX Runtime, and CodeCarbon.


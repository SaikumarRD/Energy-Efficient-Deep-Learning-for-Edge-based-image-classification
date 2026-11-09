import os
import time
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import f1_score
from codecarbon import EmissionsTracker
from tqdm import tqdm

# Import dataset loaders and builders from previous steps
from dataset_setup import (
    test_loader, edge_benchmark_loader, num_classes, seed_everything
)
from baseline_and_benchmark import build_baseline_model
from step4_distillation import build_student_model

# 1. Configuration
BASELINE_PTH = "baseline_resnet50.pth"
PRUNED_PTH = "pruned_resnet50.pth"
KD_STUDENT_PTH = "kd_mobilenetv3_small.pth"

BASELINE_ONNX_FP32 = "baseline_resnet50_fp32.onnx"
BASELINE_ONNX_INT8 = "quantized_resnet50_int8.onnx"
KD_ONNX_FP32 = "kd_mobilenetv3_fp32.onnx"
KD_ONNX_INT8 = "quantized_kd_mobilenetv3_int8.onnx"

# 2. Export PyTorch Model to ONNX and Quantize to INT8
def export_and_quantize_onnx(pytorch_model, fp32_onnx_path, int8_onnx_path):
    pytorch_model.to("cpu")
    pytorch_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

    # Export to FP32 ONNX
    if not os.path.exists(fp32_onnx_path):
        torch.onnx.export(
            pytorch_model, dummy_input, fp32_onnx_path,
            input_names=["input"], output_names=["output"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )

    # Apply INT8 Quantization (converts 32-bit weights to 8-bit integers)
    if not os.path.exists(int8_onnx_path):
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            weight_type=QuantType.QUInt8
        )
    print(f"Created INT8 Quantized Model: {int8_onnx_path}")

# 3. Benchmark ONNX INT8 Model on Edge CPU
def benchmark_onnx_int8(onnx_path, model_name, params_m, gmacs, test_loader, edge_loader):
    print(f"\n========== Benchmarking on Edge CPU: {model_name} ==========")
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    # Force single-threaded/4-thread CPU execution to simulate a low-cost Edge CPU
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # A. Evaluate Test Set Accuracy & F1
    all_preds, all_labels = [], []
    for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
        ort_inputs = {input_name: images.numpy()}
        outputs = session.run(None, ort_inputs)[0]
        preds = np.argmax(outputs, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = (sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)) * 100
    f1 = f1_score(all_labels, all_preds, average="macro") * 100

    # B. Edge CPU Single-Image Latency & Energy (500 images, batch_size=1)
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_np})

    tracker = EmissionsTracker(project_name=model_name, log_level="error", save_to_file=False)
    latencies = []
    tracker.start()
    for images, _ in edge_loader:
        img_np = images.numpy()
        start_t = time.perf_counter()
        _ = session.run(None, {input_name: img_np})
        end_t = time.perf_counter()
        latencies.append((end_t - start_t) * 1000)
    tracker.stop()

    energy_wh = tracker.final_emissions_data.energy_consumed * 1000
    avg_latency_ms = sum(latencies) / len(latencies)

    res = {
        "Model": model_name,
        "Top-1 Acc (%)": round(acc, 2),
        "Macro F1 (%)": round(f1, 2),
        "Size (MB)": round(size_mb, 2),
        "Params (M)": params_m,
        "GMACs": gmacs,
        "Edge CPU Latency (ms/img)": round(avg_latency_ms, 2),
        "Edge CPU Energy (Wh/500)": round(energy_wh, 4)
    }
    for k, v in res.items():
        print(f"{k:<26}: {v}")
    return res

# 4. Measure True Edge CPU Latency & Energy for PyTorch Models (for Apples-to-Apples Table)
def measure_pytorch_edge_cpu(model, model_name, acc, f1, size_mb, params_m, gmacs, edge_loader):
    print(f"\nMeasuring Edge CPU Latency & Energy for: {model_name}...")
    model.to("cpu")
    model.eval()
    torch.set_num_threads(4)

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    tracker = EmissionsTracker(project_name=model_name, log_level="error", save_to_file=False)
    latencies = []
    tracker.start()
    with torch.no_grad():
        for images, _ in edge_loader:
            start_t = time.perf_counter()
            _ = model(images)
            end_t = time.perf_counter()
            latencies.append((end_t - start_t) * 1000)
    tracker.stop()

    energy_wh = tracker.final_emissions_data.energy_consumed * 1000
    avg_latency_ms = sum(latencies) / len(latencies)

    return {
        "Model": model_name,
        "Top-1 Acc (%)": acc,
        "Macro F1 (%)": f1,
        "Size (MB)": size_mb,
        "Params (M)": params_m,
        "GMACs": gmacs,
        "Edge CPU Latency (ms/img)": round(avg_latency_ms, 2),
        "Edge CPU Energy (Wh/500)": round(energy_wh, 4)
    }

if __name__ == "__main__":
    seed_everything(42)

    # Load all trained PyTorch models
    baseline_model = build_baseline_model(num_classes)
    baseline_model.load_state_dict(torch.load(BASELINE_PTH, map_location="cpu", weights_only=True))

    pruned_model = torch.load(PRUNED_PTH, map_location="cpu", weights_only=False)

    student_model = build_student_model(num_classes)
    student_model.load_state_dict(torch.load(KD_STUDENT_PTH, map_location="cpu", weights_only=True))

    # 1. Quantize both Baseline ResNet-50 and Distilled MobileNetV3-Small to INT8
    print("Exporting and Quantizing models to INT8...")
    export_and_quantize_onnx(baseline_model, BASELINE_ONNX_FP32, BASELINE_ONNX_INT8)
    export_and_quantize_onnx(student_model, KD_ONNX_FP32, KD_ONNX_INT8)

    # 2. Collect Unified Edge CPU Comparison Table
    summary_rows = []

    # Row 1: Baseline ResNet-50 (FP32)
    summary_rows.append(measure_pytorch_edge_cpu(
        baseline_model, "1. Baseline_ResNet50 (FP32)", 
        99.83, 99.75, 90.16, 23.56, 4.132, edge_benchmark_loader
    ))

    # Row 2: Pruned ResNet-50 (25% Structural Pruning)
    summary_rows.append(measure_pytorch_edge_cpu(
        pruned_model, "2. Pruned_ResNet50 (25% Ch)", 
        99.47, 99.38, 50.93, 13.28, 2.397, edge_benchmark_loader
    ))

    # Row 3: Quantized ResNet-50 (INT8)
    summary_rows.append(benchmark_onnx_int8(
        BASELINE_ONNX_INT8, "3. Quantized_ResNet50 (INT8)", 
        23.56, 4.132, test_loader, edge_benchmark_loader
    ))

    # Row 4: Knowledge Distillation (MobileNetV3-Small FP32)
    summary_rows.append(measure_pytorch_edge_cpu(
        student_model, "4. KD_MobileNetV3_Small (FP32)", 
        99.66, 99.66, 6.01, 1.54, 0.061, edge_benchmark_loader
    ))

    # Row 5: Combined KD + INT8 Quantization (Ultimate Edge Model)
    summary_rows.append(benchmark_onnx_int8(
        KD_ONNX_INT8, "5. KD_MobileNetV3 + INT8 (Combo)", 
        1.54, 0.061, test_loader, edge_benchmark_loader
    ))

    # 3. Print and Save Final Research Table
    df = pd.DataFrame(summary_rows)
    print("\n" + "="*95)
    print("FINAL REPRODUCIBLE ACCURACY-LATENCY-MEMORY-ENERGY COMPARISON (EDGE CPU)")
    print("="*95)
    print(df.to_string(index=False))
    df.to_csv("final_research_comparison.csv", index=False)
    print("\nSaved full comparison table to 'final_research_comparison.csv'!")
"""
Standalone Edge-AI Inference CLI
Run single-image leaf disease diagnosis with latency benchmarking.

Usage:
    python inference.py --image path/to/leaf.jpg
    python inference.py --image path/to/leaf.jpg --model baseline_resnet50_fp32.onnx
"""

import argparse
import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image

# ── Class Labels ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Apple — Apple Scab",
    "Apple — Black Rot",
    "Apple — Cedar Apple Rust",
    "Apple — Healthy",
    "Corn (Maize) — Cercospora / Gray Leaf Spot",
    "Corn (Maize) — Common Rust",
    "Corn (Maize) — Northern Leaf Blight",
    "Corn (Maize) — Healthy",
    "Pepper (Bell) — Bacterial Spot",
    "Pepper (Bell) — Healthy",
    "Potato — Early Blight",
    "Potato — Late Blight",
    "Potato — Healthy",
    "Tomato — Bacterial Spot",
    "Tomato — Early Blight",
    "Tomato — Late Blight",
    "Tomato — Leaf Mold",
    "Tomato — Septoria Leaf Spot",
    "Tomato — Spider Mites (Two-Spotted)",
    "Tomato — Target Spot",
    "Tomato — Yellow Leaf Curl Virus",
    "Tomato — Mosaic Virus",
    "Tomato — Healthy",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """Load and preprocess image matching the training pipeline."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)


def run_inference(model_path: str, image_path: str, num_threads: int = 4):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Initialize ONNX Runtime Session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    tensor = preprocess_image(image_path)

    # Warm-up pass
    _ = session.run(None, {input_name: tensor})

    # Timed inference pass
    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})[0][0]
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Softmax probabilities
    exp_logits = np.exp(outputs - np.max(outputs))
    probs = exp_logits / exp_logits.sum()

    top5_indices = probs.argsort()[-5:][::-1]

    # Display results
    print("\n" + "=" * 60)
    print("🌿 Edge-AI Single-Image Diagnosis Result")
    print("=" * 60)
    print(f"Model Path       : {model_path}")
    print(f"Model Size       : {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
    print(f"Target Image     : {image_path}")
    print(f"Inference Latency: {latency_ms:.2f} ms (CPU, {num_threads} threads)")
    print("-" * 60)
    print("Top-5 Predictions:")
    for rank, idx in enumerate(top5_indices, 1):
        bar_len = int(probs[idx] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  #{rank} {CLASS_NAMES[idx]:<38} {bar} {probs[idx] * 100:6.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-AI Plant Disease Diagnosis CLI")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input leaf image file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kd_mobilenetv3_fp32.onnx",
        help="Path to the ONNX model file (default: kd_mobilenetv3_fp32.onnx)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for inference (default: 4)",
    )

    args = parser.parse_args()
    run_inference(args.model, args.image, args.threads)


"""
FastAPI Backend — Edge-AI Plant Disease Diagnosis
Serves the best ONNX model (KD MobileNetV3-Small FP32) via REST API.
"""
import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
ONNX_MODEL_PATH = "kd_mobilenetv3_fp32.onnx"
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

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

# ── Load ONNX Model ───────────────────────────────────────────────────────────
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
session = ort.InferenceSession(
    ONNX_MODEL_PATH, sess_options, providers=["CPUExecutionProvider"]
)
INPUT_NAME = session.get_inputs()[0].name

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Edge-AI Plant Disease Diagnosis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and convert PIL image to ONNX-ready NCHW tensor."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0           # HWC [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD                # ImageNet normalize
    arr = arr.transpose(2, 0, 1)                              # HWC → CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)     # Add batch dim → NCHW


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an uploaded leaf image and return top-5 disease predictions."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    tensor = preprocess(image)

    # Run ONNX inference
    logits = session.run(None, {INPUT_NAME: tensor})[0][0]    # shape: (23,)

    # Softmax for probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    # Top-5 predictions
    top5_idx = probs.argsort()[-5:][::-1]
    predictions = [
        {"class": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 2)}
        for i in top5_idx
    ]

    return {
        "predictions": predictions,
        "top_class": predictions[0]["class"],
        "top_confidence": predictions[0]["confidence"],
        "is_healthy": "healthy" in predictions[0]["class"].lower(),
    }


# ── Serve static frontend ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


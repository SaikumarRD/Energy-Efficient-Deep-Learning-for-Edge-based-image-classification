import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import f1_score
from thop import profile
from codecarbon import EmissionsTracker
from tqdm import tqdm

# Import your prepared loaders and metadata from Step 1
from dataset_setup import (
    train_loader, val_loader, test_loader, 
    edge_benchmark_loader, num_classes, seed_everything
)

# 1. Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10               # Adjust to 15-20 if you want higher convergence
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "baseline_resnet50.pth"

# 2. Build the Heavy Baseline CNN (ResNet-50)
def build_baseline_model(num_classes):
    # Load pretrained ResNet-50 (~25.5M parameters, ~98 MB in FP32)
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # Replace the final 1000-class head with our 23-class plant disease head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# 3. Training & Validation Loop
def train_baseline(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.to(device)
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100

        # --- Validation Phase ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = (val_correct / val_total) * 100
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODEL_SAVE_PATH)

    model.load_state_dict(best_weights)
    return model

# 4. Unified Efficiency Benchmark Function (Reusable for Pruning, KD, and Quantization!)
def benchmark_model(model, model_name, weight_path, test_loader, edge_loader, device):
    print(f"\n========== Benchmarking: {model_name} ==========")
    torch.cuda.empty_cache()  # <-- Clears GPU memory before benchmarking
    model.to(device)
    model.eval()

    # A. Model Size on Disk (MB)
    size_mb = os.path.getsize(weight_path) / (1024 * 1024)

    # B. Parameters & MACs (FLOPs) using a single 224x224 dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    macs, params = profile(copy.deepcopy(model), inputs=(dummy_input,), verbose=False)
    gmacs = macs / 1e9
    mparams = params / 1e6

    # C. Test Accuracy & Macro F1-Score
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = (sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)) * 100
    f1 = f1_score(all_labels, all_preds, average="macro") * 100

    # D. Edge Inference Latency (batch_size=1) & Energy Consumption (CodeCarbon)
    # Warm-up passes so hardware clocks stabilize
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    tracker = EmissionsTracker(project_name=model_name, log_level="error", save_to_file=False)
    latencies = []
    
    tracker.start()
    with torch.no_grad():
        for images, _ in edge_loader:  # 500 single images
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_t = time.perf_counter()
            
            _ = model(images)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_t = time.perf_counter()
            latencies.append((end_t - start_t) * 1000) # Convert to ms
    emissions_kg = tracker.stop()
    
    # Energy consumed in Watt-hours (Wh) from tracker data
    energy_kwh = tracker.final_emissions_data.energy_consumed
    energy_wh = energy_kwh * 1000 
    avg_latency_ms = sum(latencies) / len(latencies)

    # Print Clean Summary Table
    results = {
        "Model": model_name,
        "Top-1 Acc (%)": round(acc, 2),
        "Macro F1 (%)": round(f1, 2),
        "Size (MB)": round(size_mb, 2),
        "Params (M)": round(mparams, 2),
        "GMACs": round(gmacs, 3),
        "Latency (ms/img)": round(avg_latency_ms, 2),
        "Energy (Wh / 500 imgs)": round(energy_wh, 4)
    }
    for k, v in results.items():
        print(f"{k:<25}: {v}")
    return results

# 5. Execute Training and Benchmarking
if __name__ == "__main__":
    seed_everything(42)
    baseline_model = build_baseline_model(num_classes)
    
    # Train or load existing checkpoint
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Training Baseline ResNet-50 on {DEVICE}...")
        baseline_model = train_baseline(
            baseline_model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE
        )
    else:
        print(f"Loading existing weights from {MODEL_SAVE_PATH}...")
        baseline_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # Run the full benchmark suite
    benchmark_model(
        model=baseline_model,
        model_name="Baseline_ResNet50_FP32",
        weight_path=MODEL_SAVE_PATH,
        test_loader=test_loader,
        edge_loader=edge_benchmark_loader,
        device=DEVICE
    )
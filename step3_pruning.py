import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
from tqdm import tqdm

# Import dataset loaders and baseline functions
from dataset_setup import (
    train_loader, val_loader, test_loader, 
    edge_benchmark_loader, num_classes, seed_everything
)
from baseline_and_benchmark import build_baseline_model, benchmark_model

# 1. Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_PATH = "baseline_resnet50.pth"
PRUNED_SAVE_PATH = "pruned_resnet50.pth"
PRUNING_RATIO = 0.25      # 25% channel removal (~44% GMACs & size reduction without breaking skip connections)
FINETUNE_EPOCHS = 2       # Only 2 epochs needed with BN recalibration!
LEARNING_RATE = 1e-4

# 2. Apply Safe Structural Channel Pruning
def prune_model(model, pruning_ratio, device):
    model.to(device)
    model.eval()
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    # L2-Norm importance across coupled channels
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")

    # Protect the stem (conv1, bn1) and the final classification head (fc)
    ignored_layers = [model.conv1, model.bn1, model.fc]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )

    pruner.step()
    return model

# 3. Recalibrate BatchNorm & Fine-Tune
def finetune_pruned_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)

    # Step A: Fast BatchNorm Statistics Recalibration (Fixes the 5.65% collapse!)
    print("Recalibrating BatchNorm statistics after channel slicing...")
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= 100:  # 100 batches is plenty to realign BN mean/variance
                break
            _ = model(images.to(device))

    # Step B: Standard FP32 Fine-Tuning (Avoids GTX 1650 FP16 gradient underflow)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Fine-Tune Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        # Validation check
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
        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {(correct/total)*100:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model, PRUNED_SAVE_PATH)

    return torch.load(PRUNED_SAVE_PATH, map_location=device, weights_only=False)

if __name__ == "__main__":
    seed_everything(42)

    if not os.path.exists(PRUNED_SAVE_PATH):
        print("1. Loading trained Baseline ResNet-50...")
        model = build_baseline_model(num_classes)
        model.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE, weights_only=True))

        print(f"2. Physically pruning {int(PRUNING_RATIO*100)}% of convolutional channels...")
        pruned_model = prune_model(model, PRUNING_RATIO, DEVICE)

        print("3. Fine-tuning pruned model to recover accuracy...")
        pruned_model = finetune_pruned_model(
            pruned_model, train_loader, val_loader, FINETUNE_EPOCHS, LEARNING_RATE, DEVICE
        )
    else:
        print(f"Loading existing pruned model from {PRUNED_SAVE_PATH}...")
        pruned_model = torch.load(PRUNED_SAVE_PATH, map_location=DEVICE, weights_only=False)

    # 4. Benchmark Pruned Model
    benchmark_model(
        model=pruned_model,
        model_name="Pruned_ResNet50_25pct",
        weight_path=PRUNED_SAVE_PATH,
        test_loader=test_loader,
        edge_loader=edge_benchmark_loader,
        device=DEVICE
    )
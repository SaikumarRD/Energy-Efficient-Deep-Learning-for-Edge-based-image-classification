import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# Import dataset loaders and baseline functions
from dataset_setup import (
    train_loader, val_loader, test_loader, 
    edge_benchmark_loader, num_classes, seed_everything
)
from baseline_and_benchmark import build_baseline_model, benchmark_model

# 1. Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_PATH = "baseline_resnet50.pth"
STUDENT_SAVE_PATH = "kd_mobilenetv3_small.pth"
EPOCHS = 8                # MobileNetV3 is tiny; each epoch takes only ~2-3 minutes!
LEARNING_RATE = 3e-4
TEMPERATURE = 4.0         # Softens probability distributions to reveal inter-class similarity
ALPHA = 0.3               # 30% Hard Ground-Truth Loss + 70% Soft Teacher Logit Loss

# 2. Build Lightweight Student CNN (MobileNetV3-Small)
def build_student_model(num_classes):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    student = models.mobilenet_v3_small(weights=weights)
    
    # Replace the final classifier layer for 23 plant disease classes
    in_features = student.classifier[3].in_features
    student.classifier[3] = nn.Linear(in_features, num_classes)
    return student

# 3. Knowledge Distillation Loss Function (Hinton et al.)
def distillation_loss(student_logits, teacher_logits, labels, temp, alpha):
    # Hard target loss (Standard Cross-Entropy with true labels)
    hard_loss = F.cross_entropy(student_logits, labels, label_smoothing=0.05)
    
    # Soft target loss (KL-Divergence between softened student & teacher distributions)
    soft_student = F.log_softmax(student_logits / temp, dim=1)
    soft_teacher = F.softmax(teacher_logits / temp, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temp ** 2)
    
    return alpha * hard_loss + (1.0 - alpha) * soft_loss

# 4. Knowledge Distillation Training Loop
def train_kd(teacher, student, train_loader, val_loader, epochs, lr, temp, alpha, device):
    teacher.to(device)
    teacher.eval()  # Teacher stays strictly frozen in eval mode
    
    student.to(device)
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_weights = copy.deepcopy(student.state_dict())

    for epoch in range(epochs):
        student.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"KD Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Get frozen teacher logits without tracking gradients
            with torch.no_grad():
                teacher_logits = teacher(images)

            # Forward pass through student
            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, temp, alpha)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(student_logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = (correct / total) * 100

        # Validation Phase
        student.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100
        print(f"Epoch [{epoch+1}/{epochs}] KD Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(student.state_dict())
            torch.save(best_weights, STUDENT_SAVE_PATH)

    student.load_state_dict(best_weights)
    return student

if __name__ == "__main__":
    seed_everything(42)

    student_model = build_student_model(num_classes)

    if not os.path.exists(STUDENT_SAVE_PATH):
        print("1. Loading frozen Teacher (Baseline ResNet-50)...")
        teacher_model = build_baseline_model(num_classes)
        teacher_model.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True))

        print("2. Distilling knowledge into Student (MobileNetV3-Small)...")
        student_model = train_kd(
            teacher_model, student_model, train_loader, val_loader, 
            EPOCHS, LEARNING_RATE, TEMPERATURE, ALPHA, DEVICE
        )
    else:
        print(f"Loading existing distilled student from {STUDENT_SAVE_PATH}...")
        student_model.load_state_dict(torch.load(STUDENT_SAVE_PATH, map_location=DEVICE, weights_only=True))

    # 3. Benchmark Distilled Student Model
    benchmark_model(
        model=student_model,
        model_name="KD_MobileNetV3_Small",
        weight_path=STUDENT_SAVE_PATH,
        test_loader=test_loader,
        edge_loader=edge_benchmark_loader,
        device=DEVICE
    )
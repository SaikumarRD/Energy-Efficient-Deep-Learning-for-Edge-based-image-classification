import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datasets import load_dataset

# 1. Reproducibility: Lock all random seeds across runs
SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# 2. Configuration and Hyperparameters
DATA_DIR = "./plant_disease_data"  # Or "Gunjan12r/plant-disease-detection"
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_WORKERS = 2                    # Set to 0 on Windows if multiprocessing fails

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 3. Define Image Augmentation & Preprocessing Pipelines
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# 4. Load Dataset and Class Metadata
raw_dataset = load_dataset(DATA_DIR)
class_names = raw_dataset["train"].features["label"].names
num_classes = len(class_names)

def apply_train_transforms(batch):
    batch["pixel_values"] = [train_transforms(img.convert("RGB")) for img in batch["image"]]
    return batch

def apply_eval_transforms(batch):
    batch["pixel_values"] = [eval_transforms(img.convert("RGB")) for img in batch["image"]]
    return batch

train_data = raw_dataset["train"].with_transform(apply_train_transforms)
val_data = raw_dataset["validation"].with_transform(apply_eval_transforms)
test_data = raw_dataset["test"].with_transform(apply_eval_transforms)

# 5. Collate Function & Worker Seed Generator
def collate_fn(batch):
    images = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return images, labels

g = torch.Generator()
g.manual_seed(SEED)

# 6. Standard Training & Evaluation DataLoaders
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, generator=g
)

val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, 
    num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
)

test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, 
    num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
)

# 7. Edge Compression & Benchmarking Specific Loaders
# Calibration subset (200 validation samples) required for INT8 Static Quantization
calib_subset = Subset(val_data, range(200))
calib_loader = DataLoader(
    calib_subset, batch_size=1, shuffle=False, collate_fn=collate_fn
)

# Single-image loader (batch_size=1) to measure real-world edge latency & energy draw
edge_benchmark_loader = DataLoader(
    Subset(test_data, range(500)), batch_size=1, shuffle=False, collate_fn=collate_fn
)

if __name__ == "__main__":
    print(f"Classes: {num_classes} | Train batches: {len(train_loader)} | Calib samples: {len(calib_loader)}")
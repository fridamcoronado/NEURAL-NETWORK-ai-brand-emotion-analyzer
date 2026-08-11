# 1. Data Preparation

This stage takes the raw labeled dataset (stored as a zip file in Google Drive) and turns it into training-ready PyTorch `DataLoader`s: it unpacks the images, checks class balance, splits into train/validation/test, and defines the preprocessing and augmentation pipelines.

## 1.1 Mount Drive and unpack the dataset

```python
from google.colab import drive
drive.mount('/content/drive')

!rm -rf /content/data
!mkdir -p /content/data/raw
!unzip -q -o "/content/drive/MyDrive/PROYECTOS PORTAFOLIO GITHUB/dataset.zip" -d /content/data/raw
!ls /content/data/raw/dataset
```

The dataset is organized in one folder per class:

```
dataset/
├── energy_emotion/
├── innovation_future/
├── luxury_elegance/
└── trust_comfort/
```

## 1.2 Imports, seed, and device

```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
```

A fixed seed makes the train/val/test split reproducible across runs.

## 1.3 Checking class balance

Before splitting, it's worth confirming the classes are reasonably balanced — an imbalanced dataset would need class weighting or oversampling, which this one doesn't.

```python
DATA_DIR = "/content/data/raw/dataset"

CLASSES = ["energy_emotion", "innovation_future", "luxury_elegance", "trust_comfort"]
NUM_CLASSES = len(CLASSES)

def build_file_list(data_dir, classes):
    samples = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), idx))
    return samples

samples = build_file_list(DATA_DIR, CLASSES)
print(f"Total images found: {len(samples)}")

from collections import Counter
counts = Counter([CLASSES[label] for _, label in samples])
print(counts)
```

**Result:** 1,989 images total — 497 / 490 / 492 / 510 per class, close enough to balanced that no weighting was needed.

## 1.4 Stratified train / validation / test split

A 70 / 15 / 15 split, stratified so every split keeps the same class proportions.

```python
paths = [s[0] for s in samples]
labels = [s[1] for s in samples]

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths, labels, test_size=0.30, stratify=labels, random_state=SEED
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
)

print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
```

**Result:** Train 1,392 · Validation 298 · Test 299.

## 1.5 Preprocessing and data augmentation

Augmentation is applied **only to the training split** — validation and test always see the "clean" evaluation transform, so metrics reflect real generalization, not augmented inputs.

```python
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

A quick visual sanity check — plotting several augmented variants of one image per class — helps confirm the augmentation looks reasonable (not too aggressive, not distorting the ad beyond recognition):

```python
import matplotlib.pyplot as plt

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverses Normalize() so the image can be displayed correctly."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)

def show_augmentation_samples(data_dir, classes, transform, n_variants=6):
    fig, axes = plt.subplots(len(classes), n_variants + 1, figsize=(3 * (n_variants + 1), 3 * len(classes)))

    for row, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        first_image_name = sorted(os.listdir(cls_dir))[0]
        image_path = os.path.join(cls_dir, first_image_name)
        original = Image.open(image_path).convert("RGB")

        axes[row, 0].imshow(original.resize((224, 224)))
        axes[row, 0].set_title(f"{cls}\n(original)", fontsize=9)
        axes[row, 0].axis("off")

        for col in range(1, n_variants + 1):
            augmented = transform(original)
            augmented = denormalize(augmented).permute(1, 2, 0).numpy()
            axes[row, col].imshow(augmented)
            axes[row, col].set_title(f"variant {col}", fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

show_augmentation_samples(DATA_DIR, CLASSES, train_transform, n_variants=6)
```

## 1.6 Dataset and DataLoaders

```python
class AdCampaignDataset(Dataset):
    def __init__(self, paths, labels, transform, num_classes):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        target = self.labels[idx]
        return image, target


train_ds = AdCampaignDataset(train_paths, train_labels, train_transform, NUM_CLASSES)
val_ds = AdCampaignDataset(val_paths, val_labels, eval_transform, NUM_CLASSES)
test_ds = AdCampaignDataset(test_paths, test_labels, eval_transform, NUM_CLASSES)

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
```

Only the training loader shuffles — validation and test order doesn't matter for evaluation.

**Next:** [2. Model architecture & training →](02_model_training.md)

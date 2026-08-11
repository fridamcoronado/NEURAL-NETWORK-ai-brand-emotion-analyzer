# 2. Model Architecture & Training

Two-phase transfer learning on `EfficientNetB0`: first train only a new classification head with the backbone frozen (feature extraction), then unfreeze the last convolutional block and fine-tune at a much lower learning rate.

## 2.1 Building the model

```python
def build_model(num_classes, dropout=0.3):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    backbone = models.efficientnet_b0(weights=weights)

    for param in backbone.parameters():
        param.requires_grad = False

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )
    return backbone
```

- Starts from ImageNet-pretrained weights.
- Freezes the entire backbone by default — every layer starts with `requires_grad = False`.
- Replaces the original 1000-class ImageNet head with a small custom head sized for this problem: `Dropout → Linear(256) → ReLU → Dropout → Linear(4)`.

## 2.2 Helper functions

```python
def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"Total parameters:     {total:,}")
    print(f"Frozen parameters:    {frozen:,}  ({frozen/total:.1%})")
    print(f"Trainable parameters: {trainable:,}  ({trainable/total:.1%})")


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


criterion = nn.CrossEntropyLoss()
```

`run_epoch` doubles as both the training loop (when an `optimizer` is passed) and the evaluation loop (when it isn't) — `torch.set_grad_enabled` and `model.train()/model.eval()` switch behavior accordingly, so the same function drives both training and validation passes each epoch.

## 2.3 Phase 1 — Feature Extraction

Only the new classification head is trained; the pretrained backbone stays frozen.

```python
model = build_model(NUM_CLASSES).to(DEVICE)
count_parameters(model)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

EPOCHS_PHASE1 = 20
best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(EPOCHS_PHASE1):
    train_loss = run_epoch(model, train_loader, criterion, optimizer)
    val_loss = run_epoch(model, val_loader, criterion)
    print(f"[Phase 1] Epoch {epoch+1}/{EPOCHS_PHASE1} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_feature_extraction.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping (phase 1)")
            break
```

**Model size:** 4,336,512 total parameters — 4,007,548 frozen (92.4%), 328,964 trainable (7.6%).

**Result:** ran the full 20 epochs without triggering early stopping; best checkpoint at epoch 19 (`val_loss = 0.381`).

## 2.4 Phase 2 — Fine-Tuning

Load the best Phase 1 weights, then unfreeze the last convolutional block (`model.features[-1:]`) plus the classifier head, and continue training at a much lower learning rate so the pretrained features aren't destroyed.

```python
model.load_state_dict(torch.load("best_feature_extraction.pt"))

for param in model.parameters():
    param.requires_grad = False
for param in model.features[-1:].parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
count_parameters(model)

EPOCHS_PHASE2 = 15
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS_PHASE2):
    train_loss = run_epoch(model, train_loader, criterion, optimizer)
    val_loss = run_epoch(model, val_loader, criterion)
    print(f"[Phase 2] Epoch {epoch+1}/{EPOCHS_PHASE2} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_fine_tuned_partial.pt")
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print("Early stopping (phase 2)")
            break
```

**Trainable parameters in this phase:** 741,124 (17.1%) — 3,595,388 (82.9%) remain frozen.

**Result:** early stopping triggered at epoch 13; best checkpoint at epoch 8 (`val_loss = 0.361`).

## 2.5 Evaluating both checkpoints on the test set

```python
def evaluate_checkpoint(checkpoint_path, classes):
    m = build_model(len(classes)).to(DEVICE)
    m.load_state_dict(torch.load(checkpoint_path))
    m.eval()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)
            probs = torch.softmax(m(images), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_preds = np.argmax(all_probs, axis=1)

    print(f"\n=== {checkpoint_path} ===")
    print(classification_report(all_targets, all_preds, target_names=classes, digits=3))
    return all_preds, all_targets


print("PHASE 1 (frozen backbone):")
preds_p1, targets_p1 = evaluate_checkpoint("best_feature_extraction.pt", CLASSES)

print("\nPHASE 2 (fine-tuned):")
preds_p2, targets_p2 = evaluate_checkpoint("best_fine_tuned_partial.pt", CLASSES)
```

| Metric | Phase 1 | Phase 2 |
|---|---|---|
| Accuracy | 86.6% | **88.3%** |
| F1 (macro) | 0.866 | **0.882** |
| F1 (weighted) | 0.866 | **0.883** |

Fine-tuning the last block gives a consistent improvement across every class — see the main [README](../README.md) for the full per-class breakdown.

## 2.6 Saving the final weights to Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy(
    "best_fine_tuned_partial.pt",
    "/content/drive/MyDrive/PROYECTOS PORTAFOLIO GITHUB"
)
print("Model saved to Google Drive.")
```

This `.pt` file only contains the learned **weights** — not the architecture. Reloading it in a new session requires rebuilding the same `build_model()` first, which is exactly what the inference notebook does.

**Next:** [3. Loading saved weights & the Gradio demo →](03_inference_and_demo.md)

# AI Brand Emotion Analyzer

A computer vision system that classifies the dominant **brand attribute** communicated by an advertising image — `Luxury & Elegance`, `Energy & Emotion`, `Trust & Comfort`, or `Innovation & Future` — using transfer learning on a convolutional neural network.

Instead of detecting objects or basic emotions, this model reads marketing positioning from composition, color palette, lighting, and photographic style — the same signals a creative director reads when they look at an ad.

**[Try the live demo →](#)** *(replace with your Hugging Face Spaces / Gradio link)*

![Demo screenshot](assets/demo_screenshot.png)

---

## Results

| Metric | Phase 1 (Feature Extraction) | Phase 2 (Fine-Tuning) |
|---|---|---|
| Test accuracy | 86.6% | **88.3%** |
| F1 (macro) | 0.866 | **0.882** |
| F1 (weighted) | 0.866 | **0.883** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Energy & Emotion | 0.844 | 0.867 | 0.855 |
| Innovation & Future | 0.909 | 0.822 | 0.863 |
| Luxury & Elegance | 0.882 | 0.905 | 0.893 |
| Trust & Comfort | 0.900 | 0.935 | 0.917 |

Evaluated on a held-out test set of 299 images, never seen during training.

![Training curves](assets/training_curves.png)

---

## How it works

```
image  →  preprocessing  →  EfficientNetB0 (CNN)  →  predict()  →  Gradio demo
```

1. **Dataset** — 1,989 advertising images across 4 balanced classes (~500 per class), manually labeled by dominant brand attribute.
2. **Preprocessing** — resize to 224×224, ImageNet normalization; augmentation (random crop, flip, rotation, color jitter) applied only to the training split.
3. **Model** — `EfficientNetB0` pretrained on ImageNet, adapted with a custom classification head (`Dropout → Linear(256) → ReLU → Dropout → Linear(4)`).
4. **Training strategy** — two-phase transfer learning:
   - *Phase 1:* backbone frozen, only the new head is trained (328K trainable params, 7.6% of the model).
   - *Phase 2:* the last convolutional block is unfrozen and fine-tuned at a much lower learning rate (741K trainable params, 17.1%).
   - Early stopping (patience = 5) on validation loss in both phases.
5. **Inference** — a lightweight, separate notebook rebuilds the architecture and loads the saved weights (no retraining needed to reproduce results).
6. **Demo** — a Gradio interface takes an uploaded ad image and returns a probability distribution across the four brand attributes.

## Why EfficientNetB0 + transfer learning?

With only ~500 images per class, training a CNN from scratch doesn't provide enough examples to learn low-level features (edges, textures, color) reliably. Transfer learning reuses ImageNet-pretrained representations and only adapts the final layers — EfficientNetB0 gives a strong accuracy-to-size trade-off (~5M parameters) compared to heavier backbones like ResNet50, with lower overfitting risk on a small dataset.

## Project structure

```
notebooks/
  01_training_pipeline.ipynb   # data prep, augmentation, 2-phase training, evaluation
  02_inference_demo.ipynb      # loads saved weights only — no training, runs in seconds
assets/                        # charts and screenshots used in this README
requirements.txt
```

## Running it yourself

```bash
pip install -r requirements.txt
```

- To retrain: open `notebooks/01_training_pipeline.ipynb` in Colab (GPU runtime recommended), point `DATA_DIR` to your own labeled dataset.
- To just try the trained model: open `notebooks/02_inference_demo.ipynb`, it downloads/loads the saved weights and launches the Gradio demo — no training required.

## Limitations & next steps

- ~500 images/class is workable for transfer learning but modest relative to the diversity of global advertising — expect some brittleness on out-of-distribution styles.
- "Luxury" or "Trust" are marketing concepts, not objective labels — the labeling criteria should be documented and ideally reviewed by more than one annotator.
- The model may partly learn the visual style of the specific brands in the dataset rather than the attribute in the abstract — a natural next step is testing on ads from brands not seen during training.
- Planned: confusion matrix + qualitative review of misclassified examples to guide a targeted improvement (Grad-CAM visualizations are a natural fit here).

## Tech stack

`PyTorch` · `torchvision` · `EfficientNetB0` · `scikit-learn` · `Gradio` · `Google Colab (GPU)`

---

*Part of my ML/CV portfolio — built to practice the full pipeline: data preparation, transfer learning, evaluation, and deployment of a working demo.*

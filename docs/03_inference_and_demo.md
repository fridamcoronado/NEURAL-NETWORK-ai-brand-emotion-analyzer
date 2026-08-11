# 3. Loading Saved Weights & the Gradio Demo

This is the code that runs in **`notebooks/02_inference_demo.ipynb`** — a lightweight, separate notebook that only rebuilds the architecture and loads the saved weights. It never retrains anything, so it starts in seconds and survives Colab disconnects without losing hours of training.

> **Why a separate notebook?** `torch.save(model.state_dict(), ...)` only saves the learned numbers (weights), not the architecture that produced them. To use those weights again, the same model class/function has to be redefined first — but that doesn't mean retraining, just re-declaring the (empty) architecture before loading the numbers into it.

## 3.1 Setup — mount Drive, imports, constants

```python
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

CLASSES = ["energy_emotion", "innovation_future", "luxury_elegance", "trust_comfort"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

`eval_transform` must exactly match the one used to evaluate the model during training (see [1. Data Preparation](01_data_preparation.md)) — any mismatch here silently degrades prediction quality.

## 3.2 Re-declaring the architecture

Same `build_model` function used during training — nothing retrains, this just recreates the empty structure the saved weights fit into.

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

## 3.3 Loading the trained weights

```python
fresh_model = build_model(NUM_CLASSES).to(DEVICE)

fresh_model.load_state_dict(
    torch.load(
        "/content/drive/MyDrive/PROYECTOS PORTAFOLIO GITHUB/best_fine_tuned_partial.pt",
        map_location=DEVICE
    )
)
fresh_model.eval()

print("Model loaded")
```

`map_location=DEVICE` makes this safe to run whether or not a GPU is available — weights trained on GPU still load correctly on CPU-only sessions.

**Sanity check** used while first validating this approach: loading the same checkpoint into two independently-built models and confirming they produce identical outputs on a batch of real images.

```python
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(DEVICE)
    pred1 = model(images)
    pred2 = fresh_model(images)

print(torch.allclose(pred1, pred2))  # True
```

## 3.4 Single-image prediction

```python
def predict(pil_image):
    fresh_model.eval()
    x = eval_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(fresh_model(x), dim=1)[0]

    label = int(probs.argmax())
    confidence = float(probs[label])
    return CLASSES[label], confidence


img = Image.open("/content/drive/MyDrive/PROYECTOS PORTAFOLIO GITHUB/ejemplo_anuncio.jpg")
label, confidence = predict(img)

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2%}")
```

**Example result:** `luxury_elegance`, 95.4% confidence.

## 3.5 Gradio demo

```python
import gradio as gr
from PIL import Image


def predict_gradio(pil_image):
    fresh_model.eval()

    # Apply the exact same preprocessing used during training/evaluation
    x = eval_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(fresh_model(x), dim=1)[0]

    # Build the dict Gradio's Label component expects
    results = {
        CLASSES[i]: float(probs[i])
        for i in range(len(CLASSES))
    }
    return results


example_paths = [
    "/content/drive/MyDrive/PROYECTOS PORTAFOLIO GITHUB/ejemplo_anuncio.jpg",
    "/content/drive/MyDrive/PROYECTOS PORTAFOLIO GITHUB/ejemplo_anuncio2.jpg",
    "/content/drive/MyDrive/PROYECTOS PORTAFOLIO GITHUB/ejemplo_anuncio3.5.jpg",
    "/content/drive/MyDrive/PROYECTOS PORTAFOLIO GITHUB/ejemplo_anuncio4.5.jpg",
]

demo = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=example_paths,
    title="AI Brand Emotion Analyzer",
    description="""
    Upload an advertising image and the CNN predicts
    the main brand perception communicated:
    Luxury & Elegance, Trust & Comfort, Energy & Emotion, Innovation & Future.
    Below are some examples of images to try:
    """
)

demo.launch()
```

Run every cell above, in order, before this one — `fresh_model` has to exist first (see §3.3) or this last cell raises `NameError: name 'fresh_model' is not defined`.

**← Back to:** [README](../README.md) · [1. Data Preparation](01_data_preparation.md) · [2. Model Training](02_model_training.md)

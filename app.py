import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# The weights file lives in the root of this Space (uploaded alongside app.py)
fresh_model = build_model(NUM_CLASSES).to(DEVICE)
fresh_model.load_state_dict(
    torch.load("best_fine_tuned_partial.pt", map_location=DEVICE)
)
fresh_model.eval()


def predict_gradio(pil_image):
    fresh_model.eval()
    x = eval_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(fresh_model(x), dim=1)[0]

    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


example_paths = [
    "examples/ejemplo_anuncio.jpg",
    "examples/ejemplo_anuncio2.jpg",
    "examples/ejemplo_anuncio3.5.jpg",
    "examples/ejemplo_anuncio4.5.jpg",
]

demo = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    examples=example_paths,
    title="AI Brand Emotion Analyzer",
    description="""
    Upload an advertising image and the CNN predicts the main brand attribute
    communicated: Luxury & Elegance, Trust & Comfort, Energy & Emotion, or
    Innovation & Future. EfficientNetB0 fine-tuned via 2-phase transfer learning
    — 88.3% test accuracy. Try one of the examples below, or upload your own ad.
    """,
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

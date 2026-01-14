from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import io
import os

app = FastAPI()

# ===== Config =====
DEVICE = torch.device("cpu")
IMG_SIZE = (400, 300)
CLASS_NAMES = ['A', 'Lu', 'W']

# ===== Model Definition =====
def get_model():
    model = EfficientNet.from_name('efficientnet-b3')  # IMPORTANT: no internet download
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model._fc.in_features, 3)
    )
    return model

# ===== Load Model =====
MODEL_PATH = "models/Fold1_Best.pth"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Touchless EfficientNet model loaded successfully")

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ===== Prediction =====
def predict(img: Image.Image):
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return CLASS_NAMES[pred.item()], float(conf.item())

# ===== API Endpoint =====
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    pred_class, confidence = predict(img)

    # Return structure compatible with your Streamlit UI
    return {
        "success": True,
        "classification": {
            "predicted_class": pred_class,
            "confidence": confidence
        },
        "num_cores": 0,
        "num_deltas": 0,
        "ridge_counts": [],
        "quality": {"status": "OK"},
        "overlay_base64": None
    }

# ===== Root Test =====
@app.get("/")
def root():
    return {"status": "Touchless Fingerprint API Running"}

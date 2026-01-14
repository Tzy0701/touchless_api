from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import io
import glob
import os

app = FastAPI()

DEVICE = torch.device("cpu")   # Render free tier = CPU only
IMG_SIZE = (400,300)
CLASS_NAMES = ['A','Lu','W']

# ===== Load Models once at startup =====
def get_model():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Sequential(nn.Dropout(0.5),
                              nn.Linear(model._fc.in_features, 3))
    return model

models = []
model = get_model()
model.load_state_dict(torch.load("Fold1_Best.pth", map_location="cpu"))
model.eval()

print(f"✅ Loaded {len(models)} ensemble models")

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===== Prediction Function =====
def ensemble_predict(img: Image.Image):
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        probs_sum = None
        for m in models:
            out = m(img_t)
            probs = F.softmax(out, dim=1)

            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs

        avg_probs = probs_sum / len(models)
        conf, pred = torch.max(avg_probs, 1)

    predicted_class = CLASS_NAMES[pred.item()]
    confidence = conf.item()

    return predicted_class, confidence

# ===== API Endpoint =====
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    pred_class, confidence = ensemble_predict(img)

    return {
        "success": True,
        "classification": {
            "predicted_class": pred_class,
            "confidence": confidence
        }
    }

# ===== Root test =====
@app.get("/")
def root():
    return {"status": "Fingerprint API Running"}

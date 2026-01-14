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
DEVICE = torch.device("cpu")   # Render free = CPU only
IMG_SIZE = (400, 300)
CLASS_NAMES = ['A', 'Lu', 'W']

# ===== Load Single Model at Startup =====
def get_model():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model._fc.in_features, 3)
    )
    return model

MODEL_PATH = os.path.join("models", "Fold1_Best.pth")

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

# ===== Prediction Function =====
def predict(img: Image.Image):
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[pred.item()]
    confidence = float(conf.item())

    return predicted_class, confidence

# ===== API Endpoint =====
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    pred_class, confidence = predict(img)

    # Return JSON compatible with your Streamlit summary page
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

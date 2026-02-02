import io
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image

from src.config import CLASSIFIER_PATH, AUTOENCODER_PATH, OUTPUT_DIR
from src.model import build_classifier, ConvAutoencoder
from src.dataset import VAL_TRANSFORM
from src.autoencoder import AE_TRANSFORM

app = FastAPI(title="Manufacturing Defect Detector")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = None
autoencoder = None
ae_threshold = None


@app.on_event("startup")
def load_models():
    global classifier, autoencoder, ae_threshold

    classifier = build_classifier(freeze_backbone=False)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))
    classifier.to(device).eval()

    autoencoder = ConvAutoencoder()
    autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=device, weights_only=True))
    autoencoder.to(device).eval()

    threshold_path = OUTPUT_DIR / "ae_threshold.npy"
    ae_threshold = float(np.load(threshold_path))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    cls_tensor = VAL_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = classifier(cls_tensor).squeeze()
        prob = torch.sigmoid(logit).item()

    ae_tensor = AE_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = autoencoder(ae_tensor)
        recon_error = ((ae_tensor - recon) ** 2).mean().item()

    label = "FAIL" if prob >= 0.5 else "PASS"
    is_novel = recon_error > ae_threshold

    return {
        "label": label,
        "defect_probability": round(prob, 4),
        "reconstruction_error": round(recon_error, 6),
        "anomaly_threshold": round(ae_threshold, 6),
        "novel_defect_suspected": is_novel,
    }

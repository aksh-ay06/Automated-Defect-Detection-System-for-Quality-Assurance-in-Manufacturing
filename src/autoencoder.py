import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

from src.config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    AUTOENCODER_PATH, ANOMALY_THRESHOLD_PERCENTILE, OUTPUT_DIR,
)
from src.model import ConvAutoencoder
from src.dataset import DefectDataset


AE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def get_good_only_loader(data_dir: Path = DATA_DIR):
    paths = sorted((data_dir / "good").glob("*.[jJ][pP][gG]"))
    labels = [0] * len(paths)
    ds = DefectDataset(paths, labels, AE_TRANSFORM)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


def train_autoencoder(num_epochs: int = 50, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_good_only_loader()

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for images, _ in loader:
            images = images.to(device)
            recon = model(images)
            loss = criterion(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg = np.mean(losses)
        if (epoch + 1) % 10 == 0:
            print(f"AE Epoch {epoch+1}/{num_epochs}  loss={avg:.6f}")

    torch.save(model.state_dict(), AUTOENCODER_PATH)
    threshold = compute_threshold(model, loader, device)
    np.save(OUTPUT_DIR / "ae_threshold.npy", threshold)
    print(f"Autoencoder saved. Anomaly threshold: {threshold:.6f}")
    return model, threshold


def compute_threshold(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            recon = model(images)
            mse = ((images - recon) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            errors.extend(mse)
    return float(np.percentile(errors, ANOMALY_THRESHOLD_PERCENTILE))


def score_anomaly(model, image: Image.Image, device) -> float:
    model.eval()
    tensor = AE_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(tensor)
        mse = ((tensor - recon) ** 2).mean().item()
    return mse


if __name__ == "__main__":
    train_autoencoder()

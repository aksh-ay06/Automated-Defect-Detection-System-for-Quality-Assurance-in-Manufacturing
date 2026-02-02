import torch
import torch.nn as nn
import numpy as np

from src.config import (
    NUM_EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE,
    COST_FN, COST_FP, CLASSIFIER_PATH,
)
from src.dataset import build_loaders
from src.model import build_classifier


def cost_sensitive_weight(labels: torch.Tensor) -> torch.Tensor:
    w = torch.where(labels == 1, COST_FN, COST_FP)
    return w


def train_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders()

    model = build_classifier().to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            per_sample_loss = criterion(logits, labels)
            weights = cost_sensitive_weight(labels)
            loss = (per_sample_loss * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images).squeeze(1)
                per_sample_loss = criterion(logits, labels)
                weights = cost_sensitive_weight(labels)
                loss = (per_sample_loss * weights).mean()
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), CLASSIFIER_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(CLASSIFIER_PATH, weights_only=True))
    return model, test_loader, device


if __name__ == "__main__":
    model, test_loader, device = train_classifier()
    print(f"Training complete. Model saved to {CLASSIFIER_PATH}")

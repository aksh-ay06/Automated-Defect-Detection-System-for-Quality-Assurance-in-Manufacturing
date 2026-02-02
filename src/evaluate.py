import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, average_precision_score,
    classification_report,
)

from src.config import COST_FP, COST_FN, PLOT_DIR


def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels, dtype=int)


def plot_cost_sensitive_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cost_matrix = np.array([[0, COST_FP], [COST_FN, 0]])
    weighted_cm = cm * cost_matrix

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Good", "Defective"], yticklabels=["Good", "Defective"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")

    sns.heatmap(weighted_cm, annot=True, fmt=".1f", cmap="Reds", ax=axes[1],
                xticklabels=["Good", "Defective"], yticklabels=["Good", "Defective"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title(f"Cost-Sensitive Matrix (FP={COST_FP}, FN={COST_FN})")

    plt.tight_layout()
    save_path = save_path or PLOT_DIR / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    total_cost = weighted_cm.sum()
    print(f"Total misclassification cost: {total_cost:.1f}")
    return cm, weighted_cm


def plot_precision_recall_curve(y_true, y_probs, save_path=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_path or PLOT_DIR / "precision_recall_curve.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return ap


def run_evaluation(model, test_loader, device, threshold=0.5):
    probs, labels = collect_predictions(model, test_loader, device)
    preds = (probs >= threshold).astype(int)

    print(classification_report(labels, preds, target_names=["Good", "Defective"]))

    cm, wcm = plot_cost_sensitive_confusion_matrix(labels, preds)
    ap = plot_precision_recall_curve(labels, probs)

    print(f"Average Precision: {ap:.3f}")
    return {"probs": probs, "labels": labels, "preds": preds, "ap": ap}

"""Test-set evaluation: Accuracy/Precision/Recall/F1 + confusion matrix (Ch3-F.1/F.2)."""
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from dataset import collate_fn

CLASS_NAMES = ["Negative", "Neutral", "Positive"]  # labels 0, 1, 2


@torch.no_grad()
def predict(model, dataset, batch_size: int = 32, device: str = "cpu"):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_preds, all_labels = [], []
    for token_ids, lengths, feats, labels in loader:
        token_ids, lengths = token_ids.to(device), lengths.to(device)
        if feats is not None:
            feats = feats.to(device)
        logits = model(token_ids, lengths, extra_features=feats)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds)


def macro_averaged_accuracy(cm: np.ndarray) -> float:
    """Ch3-F.1.a's accuracy formula: per-class one-vs-rest (tp+tn)/total, averaged."""
    total = cm.sum()
    per_class = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        per_class.append((tp + tn) / total)
    return float(np.mean(per_class))


def evaluate_model(model, dataset, batch_size: int = 32, device: str = "cpu") -> dict:
    y_true, y_pred = predict(model, dataset, batch_size, device)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),      # correct / total
        "macro_averaged_accuracy": macro_averaged_accuracy(cm),  # Ch3-F.1.a's definition
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(w_p),  # SOP3 compares weighted F1 to 0.84
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f1),
        "per_class": classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "n_test": int(len(y_true)),
    }


def plot_confusion_matrix(cm, class_names, out_path: str, title: str = "Confusion Matrix") -> None:
    """Saves a labeled confusion matrix heatmap (counts) as a PNG."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm), display_labels=class_names
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix image to {out_path}")


def save_metrics(metrics: dict, out_dir: str, title: str = "Confusion Matrix",
                 filename: str = "metrics.json", plot_name: str = "confusion_matrix.png") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir}/{filename}")

    headline = {
        k: round(metrics[k], 4)
        for k in ["accuracy", "macro_averaged_accuracy", "macro_precision",
                  "macro_recall", "macro_f1", "weighted_f1"]
    }
    print(json.dumps(headline, indent=2))
    print("Confusion matrix (rows=true, cols=pred):", metrics["class_names"])
    for row in metrics["confusion_matrix"]:
        print(row)

    plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["class_names"],
        os.path.join(out_dir, plot_name), title=title,
    )

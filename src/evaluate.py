import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from model import BiLSTMClassifier
from dataset import TaglishReviewDataset, collate_fn

# Matches the FiReCS label encoding used throughout preprocessing/training
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


@torch.no_grad()
def run_inference(model, loader, device):
    """Runs the model over a DataLoader and returns (true_labels, predicted_labels)."""
    model.eval()
    all_preds, all_labels = [], []
    for x, lengths, y in loader:
        x = x.to(device)
        logits = model(x, lengths)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    return all_labels, all_preds


def evaluate_model(
    model_path,
    test_df,
    ft_model,
    text_col="clean_review",
    label_col="label",
    input_dim=300,
    hidden_dim=128,
    num_classes=3,
    dropout=0.3,
    batch_size=32,
    model_name="baseline",
    results_dir="../results",
    device=None,
):
    """
    Loads trained weights, runs the model on the test set, and computes the
    Analysis-stage metrics from the architecture diagram: predicted sentiment
    labels, accuracy, precision, recall, F1-score, and a confusion matrix.
    Saves a text report + confusion matrix plot to results_dir.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMClassifier(input_dim, hidden_dim, num_classes, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_ds = TaglishReviewDataset(test_df[text_col], test_df[label_col], ft_model)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    y_true, y_pred = run_inference(model, test_loader, device)

    # --- Predicted sentiment (diagram: "Predicted sentiment: Positive, Neutral, Negative") ---
    predicted_labels = [LABEL_NAMES[p] for p in y_pred]
    true_labels = [LABEL_NAMES[t] for t in y_true]

    # --- Analyze performance: Accuracy, Macro Precision, Macro Recall, Macro F1-score ---
    # Per Ch3-F.1 (p.61-62): macro averaging gives every class equal weight,
    # appropriate here since FiReCS's 3 classes are roughly balanced.
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=[LABEL_NAMES[i] for i in range(num_classes)], zero_division=0
    )

    print(f"\n=== {model_name} — Test Set Results ===")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1-score:  {f1:.4f}")
    print("\nPer-class report:")
    print(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[LABEL_NAMES[i] for i in range(num_classes)])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} — Confusion Matrix (Test Set)")
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    fig_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    report_path = os.path.join(results_dir, f"{model_name}_test_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{model_name} — Test Set Results\n")
        f.write(f"Accuracy:       {accuracy:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall:    {recall:.4f}\n")
        f.write(f"Macro F1-score:  {f1:.4f}\n\n")
        f.write("Per-class report:\n")
        f.write(report)

    print(f"\nSaved confusion matrix -> {fig_path}")
    print(f"Saved report -> {report_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }


if __name__ == "__main__":
    from feature_extraction import load_fasttext_model

    test_df = pd.read_csv("../data/processed/test.csv")
    ft_model = load_fasttext_model("tl")

    evaluate_model(
        model_path="../models/baseline_bilstm.pt",
        test_df=test_df,
        ft_model=ft_model,
        model_name="baseline",
    )
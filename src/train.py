import copy
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from model import BiLSTMClassifier
from dataset import TaglishReviewDataset, collate_fn


def train_bilstm(
    train_df,
    val_df,
    ft_model,
    text_col="clean_review",
    label_col="label",
    input_dim=300,
    hidden_dim=128,
    num_classes=3,
    dropout=0.3,
    lr=0.001,
    batch_size=32,
    max_epochs=50,
    patience=5,
    device=None,
):
    """
    Trains the baseline Bi-LSTM per Ch3-C.4.2 hyperparameters:
    Adam (lr=0.001), CrossEntropyLoss, batch_size=32, hidden_dim=128,
    dropout=0.3, max 50 epochs with early stopping (patience=5 on val loss).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = TaglishReviewDataset(train_df[text_col], train_df[label_col], ft_model)
    val_ds = TaglishReviewDataset(val_df[text_col], val_df[label_col], ft_model)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMClassifier(input_dim, hidden_dim, num_classes, dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, max_epochs + 1):
        # --- train ---
        model.train()
        running_loss = 0.0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ds)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # --- early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    from feature_extraction import load_fasttext_model

    train_df = pd.read_csv("../data/processed/train.csv")
    val_df = pd.read_csv("../data/processed/val.csv")

    ft_model = load_fasttext_model("tl")

    model, history = train_bilstm(train_df, val_df, ft_model)

    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/baseline_bilstm.pt")
    print("Saved to ../models/baseline_bilstm.pt")

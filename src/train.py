"""Training loop with early stopping, per Ch3-C.4.2 hyperparameters."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import collate_fn

GRAD_CLIP_NORM = 5.0  # applied to both models


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """One pass over `loader`. Trains if an optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for token_ids, lengths, feats, labels in loader:
            token_ids = token_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            if feats is not None:
                feats = feats.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(token_ids, lengths, extra_features=feats)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_n += labels.size(0)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / total_n, total_correct / total_n, macro_f1


def train_model(
    model,
    train_dataset,
    val_dataset,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    max_epochs: int = 50,
    patience: int = 5,
    device: str = "cpu",
    seed: int = 42,
    monitor: str = "val_loss",
    verbose: bool = True,
):
    """Adam/AdamW + CrossEntropyLoss, early stopping on `monitor` (val_loss or
    val_f1). Returns (best_model_state, history)."""
    if monitor not in ("val_loss", "val_f1"):
        raise ValueError("monitor must be 'val_loss' or 'val_f1'")

    set_seed(seed)
    model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # weight_decay > 0 -> AdamW, else plain Adam
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_score = float("inf") if monitor == "val_loss" else -float("inf")
    best_state, best_epoch = None, 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = _run_epoch(model, val_loader, criterion, None, device)
        epoch_time = time.time() - t0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "seconds": epoch_time,
            }
        )
        if verbose:
            print(
                f"epoch {epoch:2d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| val_loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | {epoch_time:.1f}s"
            )

        score = val_loss if monitor == "val_loss" else val_f1
        improved = score < best_score if monitor == "val_loss" else score > best_score

        if improved:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    if verbose:
        print(f"Best epoch: {best_epoch} ({monitor}={best_score:.4f})")
    return best_state, history


def time_pilot(model, train_dataset, val_dataset, batch_size: int = 32, device: str = "cpu"):
    """Runs 2 real epochs purely to measure wall-clock time per epoch."""
    print("Running 2-epoch timing pilot...")
    _, history = train_model(
        model, train_dataset, val_dataset, batch_size=batch_size,
        max_epochs=2, patience=99, device=device, verbose=True,
    )
    avg = sum(h["seconds"] for h in history) / len(history)
    print(f"\nAverage epoch time: {avg:.1f}s")
    print(f"Estimated 15-epoch run: {avg * 15 / 60:.1f} min")
    print(f"Estimated 30 independent 15-epoch runs: {avg * 15 * 30 / 60:.1f} min")
    return history

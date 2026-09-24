import quiet  # noqa: F401  (must be imported before torch/calamancy)
import argparse
import json
import os

import numpy as np
import torch

from data_prep import MAX_LEN, build_feature_arrays, load_all_splits, truncate
from dataset import ReviewDataset
from evaluate import evaluate_model, save_metrics
from feature_extraction import load_fasttext_model
from features import build_tagset, save_tagsets
from model import FUSIONS, BiLSTMClassifier
from tokenization import DEFAULT_TAGGER
from train import set_seed, time_pilot, train_model
from vocab import build_embedding_matrix, build_vocab, encode_tokens, save_vocab

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data", "processed")
MODEL_DIR = os.path.join(_ROOT, "models")
# Design choices are scored on validation, here. Final thesis runs go to results/final/.
RESULTS_DIR = os.path.join(_ROOT, "results", "development")
FINAL_DIR = os.path.join(_ROOT, "results", "final")

# C0 / manuscript settings -- folder names are built relative to this.
DEFAULT_REG = {"dropout": 0.3, "input_dropout": 0.0, "weight_decay": 0.0, "lr": 0.001}
# Phase 3 winner; used when no flag overrides it.
CHOSEN_REG = {**DEFAULT_REG, "lr": 0.0005}


def reg_name(reg: dict) -> str:
    """Folder name for a config, e.g. 'dropout0.5'; '' for the C0 defaults."""
    return "_".join(f"{k}{reg[k]:g}" for k in DEFAULT_REG if reg[k] != DEFAULT_REG[k])


def reg_label(reg: dict) -> str:
    """Human-readable name for printing."""
    if reg == CHOSEN_REG:
        return f"{reg_name(reg)} (chosen in Phase 3)"
    return reg_name(reg) or "manuscript settings (C0)"


def final_settings_error(reg: dict) -> str | None:
    """Error message if a test-split run isn't using the frozen (chosen) settings."""
    if reg != CHOSEN_REG:
        return (f"!!! TEST split refused: training settings {reg_label(reg)} are not the frozen "
                f"settings {reg_label(CHOSEN_REG)}.\n    Test-set runs may only use the design "
                f"chosen on validation (drop the --dropout/--input-dropout/--weight-decay/--lr flags).")
    return None


def results_dir(tagger: str, reg: dict | None = None, split: str = "val") -> str:
    """test -> results/final/<tagger>/. val -> results/development/<tagger>/
    (or .../regularization/<config>/ for non-default settings)."""
    if split == "test":
        return os.path.join(FINAL_DIR, tagger)
    name = reg_name(reg) if reg else ""
    base = os.path.join(RESULTS_DIR, tagger)
    return os.path.join(base, "regularization", name) if name else base


def add_reg_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dropout", type=float, default=CHOSEN_REG["dropout"],
                   help="dropout before the dense layer")
    p.add_argument("--input-dropout", type=float, default=CHOSEN_REG["input_dropout"],
                   help="locked (variational) dropout on each token's fused input")
    p.add_argument("--weight-decay", type=float, default=CHOSEN_REG["weight_decay"],
                   help="> 0 switches the optimizer to AdamW with this decay")
    p.add_argument("--lr", type=float, default=CHOSEN_REG["lr"],
                   help="learning rate (0.0005 chosen in Phase 3; the manuscript's 0.001 is C0)")


def reg_from_args(args) -> dict:
    return {"dropout": args.dropout, "input_dropout": args.input_dropout,
            "weight_decay": args.weight_decay, "lr": args.lr}


def artifacts_dir(tagger: str) -> str:
    """vocab/embeddings/tagsets depend on the tokenization, so key them by tagger."""
    return os.path.join(MODEL_DIR, tagger)


def model_label(model: str, fusion: str) -> str:
    """Results folder name. 'proposed' alone means the manuscript's concat fusion."""
    return model if model == "baseline" or fusion == "concat" else f"{model}-{fusion}"


def metrics_filename(split: str) -> str:
    return "metrics_val.json" if split == "val" else "metrics.json"


def build_datasets(use_features: bool, tagger: str = DEFAULT_TAGGER):
    """Returns (train_ds, val_ds, test_ds, embedding_matrix, feature_dim)."""
    print(f"=== Loading splits (tagger: {tagger}) ===")
    splits = load_all_splits(DATA_DIR, tagger)
    art = artifacts_dir(tagger)
    os.makedirs(art, exist_ok=True)

    print("=== Building vocab + embedding matrix ===")
    all_tokens = [t for df in splits.values() for t in df["tokens"]]
    vocab = build_vocab(all_tokens)
    ft_model = load_fasttext_model("tl")
    embedding_matrix = build_embedding_matrix(vocab, ft_model)
    save_vocab(vocab, os.path.join(art, "vocab.json"))
    np.save(os.path.join(art, "embedding_matrix.npy"), embedding_matrix)
    print(f"  vocab size: {len(vocab)}")

    feature_dim, features = 0, {name: None for name in splits}
    if use_features:
        print("=== Building POS/dependency one-hot features ===")
        pos_set = build_tagset(t for df in splits.values() for t in df["pos"])
        dep_set = build_tagset(t for df in splits.values() for t in df["dep"])
        save_tagsets(pos_set, dep_set, os.path.join(art, "tagsets.json"))
        feature_dim = len(pos_set) + len(dep_set)
        print(f"  POS tags: {len(pos_set)}  dep labels: {len(dep_set)}  fused width: {feature_dim}")
        features = {
            name: build_feature_arrays(df, pos_set, dep_set) for name, df in splits.items()
        }

    datasets = {}
    for name, df in splits.items():
        seqs = [truncate(encode_tokens(t, vocab)) for t in df["tokens"]]
        datasets[name] = ReviewDataset(seqs, df["label"].tolist(), features[name])

    return datasets["train"], datasets["val"], datasets["test"], embedding_matrix, feature_dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "proposed"], default="baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pilot", action="store_true", help="2-epoch timing pilot only")
    p.add_argument("--monitor", choices=["val_loss", "val_f1"], default="val_loss",
                   help="early-stopping criterion (Ch3-C.4.2 specifies val_loss)")
    p.add_argument("--tagger", default=DEFAULT_TAGGER,
                   help="calamanCy pipeline for tokens + tags (applies to BOTH models)")
    p.add_argument("--fusion", choices=FUSIONS, default="projection",
                   help="how the proposed model fuses calamanCy tags with FastText")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="val for development (default); test only for the final evaluation")
    p.add_argument("--save-checkpoint", action="store_true",
                   help="save model weights (~20MB); off by default so 30-run sweeps stay small")
    add_reg_args(p)
    args = p.parse_args()
    reg = reg_from_args(args)
    if args.split == "test" and final_settings_error(reg):
        print(final_settings_error(reg))
        return

    use_features = args.model == "proposed"
    train_ds, val_ds, test_ds, embedding_matrix, feature_dim = build_datasets(use_features, args.tagger)

    set_seed(args.seed)
    model = BiLSTMClassifier(
        embedding_matrix, hidden_size=128, dropout=reg["dropout"],
        freeze_embeddings=True, extra_feature_dim=feature_dim, fusion=args.fusion,
        input_dropout=reg["input_dropout"],
    )
    label = model_label(args.model, args.fusion)
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"=== Model: {label} | LSTM input width: {model.lstm.input_size} "
          f"| trainable params: {trainable:,} | seed: {args.seed} | tagger: {args.tagger} ===")
    print(f"=== Training settings: {reg_label(reg)} ===")

    if args.pilot:
        time_pilot(model, train_ds, val_ds)
        return

    best_state, history = train_model(
        model, train_ds, val_ds, seed=args.seed, monitor=args.monitor,
        lr=reg["lr"], weight_decay=reg["weight_decay"],
    )
    model.load_state_dict(best_state)

    out_dir = os.path.join(results_dir(args.tagger, reg, args.split), label, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    if args.save_checkpoint:
        torch.save(model.state_dict(), os.path.join(artifacts_dir(args.tagger), f"{label}_seed{args.seed}.pt"))

    split_label = "validation" if args.split == "val" else "held-out TEST"
    print(f"=== Evaluating on {split_label} set ===")
    metrics = evaluate_model(model, val_ds if args.split == "val" else test_ds)
    metrics["config"] = {
        "model": args.model, "fusion": model.fusion, "split": args.split,
        "seed": args.seed, "monitor": args.monitor, "tagger": args.tagger,
        "feature_dim": feature_dim, "max_len": MAX_LEN,
        "hidden_size": 128, **reg, "batch_size": 32,
        "epochs_run": len(history), "trainable_params": trainable,
    }
    metrics["history"] = history
    suffix = "_val" if args.split == "val" else ""
    save_metrics(
        metrics, out_dir,
        title=f"{label} — {split_label} set (seed {args.seed})\n{args.tagger}",
        filename=metrics_filename(args.split), plot_name=f"confusion_matrix{suffix}.png",
    )

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

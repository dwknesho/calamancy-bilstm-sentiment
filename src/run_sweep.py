"""
Main script: trains the baseline and proposed models over several random seeds
and compares them -- per-seed differences, seeds won, and a paired t-test. On the
validation split it applies the pre-set pass/fail bar; on the test split it also
runs a one-sample t-test against the 0.84 benchmark. The baseline can be reused
from stored results instead of retrained. Per-run metrics and a summary JSON are saved.
"""
import quiet  # noqa: F401  (must be imported before torch/calamancy)
import argparse
import json
import math
import os
import statistics
import time

from scipy import stats

from dataset import ReviewDataset
from evaluate import evaluate_model
from model import FUSIONS, BiLSTMClassifier
from run_experiment import (add_reg_args, build_datasets, final_settings_error, metrics_filename,
                            model_label, reg_from_args, reg_label, results_dir)
from tokenization import DEFAULT_TAGGER
from train import set_seed, train_model

BENCHMARK_F1 = 0.84  # Cosme & De Leon (2024) benchmark
DEV_SEEDS = [42, 7, 123, 2024, 31337]
# Starts at 1000 so it never overlaps a dev seed (42, 7, 123, 2024, 31337).
FINAL_SEED_START = 1000
FINAL_N = 30
BAR_WIN_FRACTION = 0.8  # pre-registered: beat baseline mean AND win >= 80% of seeds (4/5)
METRICS = ("macro_f1", "accuracy", "weighted_f1")


def strip_features(ds: ReviewDataset) -> ReviewDataset:
    """Baseline view of the same data: identical sequences, no features."""
    return ReviewDataset(ds.sequences, ds.labels, None)


def run_one(model_name, fusion, datasets, embedding_matrix, feature_dim, seed, monitor, tagger,
            split, verbose, reg):
    train_ds, val_ds, test_ds = datasets
    set_seed(seed)
    model = BiLSTMClassifier(
        embedding_matrix, hidden_size=128, dropout=reg["dropout"],
        freeze_embeddings=True, extra_feature_dim=feature_dim, fusion=fusion,
        input_dropout=reg["input_dropout"],
    )
    best_state, history = train_model(
        model, train_ds, val_ds, seed=seed, monitor=monitor, verbose=verbose,
        lr=reg["lr"], weight_decay=reg["weight_decay"],
    )
    model.load_state_dict(best_state)

    metrics = evaluate_model(model, val_ds if split == "val" else test_ds)
    metrics["config"] = {
        "model": model_name, "fusion": model.fusion, "seed": seed, "monitor": monitor,
        "split": split, "feature_dim": feature_dim, "epochs_run": len(history), "tagger": tagger,
        **reg,
    }
    metrics["history"] = history

    out_dir = os.path.join(results_dir(tagger, reg, split), model_label(model_name, fusion), f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, metrics_filename(split)), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def load_stored(tagger, label, seeds, split, reg=None):
    """Reads previously saved per-seed metrics. Returns (values, missing_seeds)."""
    vals = {k: [] for k in METRICS}
    missing = []
    for s in seeds:
        path = os.path.join(results_dir(tagger, reg, split), label, f"seed_{s}", metrics_filename(split))
        if not os.path.exists(path):
            missing.append(s)
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        for k in METRICS:
            vals[k].append(m[k])
    return vals, missing


def summarize(name, values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"  {name:<12} mean={mean:.4f}  sd={sd:.4f}  min={min(values):.4f}  max={max(values):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--n", type=int, default=None,
                   help="number of seeds (<=5: dev seeds; >5: fresh seeds 1000..; test default: 30 fresh)")
    p.add_argument("--resume", action="store_true",
                   help="reuse seeds that already have saved results (continue an interrupted run)")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="val for development decisions (default); test only for the final run")
    p.add_argument("--monitor", choices=["val_loss", "val_f1"], default="val_loss")
    p.add_argument("--models", nargs="+", choices=["baseline", "proposed"],
                   default=["baseline", "proposed"])
    p.add_argument("--fusion", choices=FUSIONS, default="projection",
                   help="how the proposed model fuses calamanCy tags with FastText")
    p.add_argument("--tagger", default=DEFAULT_TAGGER)
    p.add_argument("--no-epoch-log", action="store_true", help="hide per-epoch progress lines")
    add_reg_args(p)
    args = p.parse_args()
    reg = reg_from_args(args)
    config = reg_label(reg)

    if args.split == "test" and final_settings_error(reg):
        print(final_settings_error(reg))
        return

    # Dev sweeps reuse the same 5 seeds so results pair up across runs.
    # Final round (test, or --n > 5) uses fresh, unused seeds.
    if args.seeds:
        seeds = args.seeds
    elif args.split == "test" or (args.n and args.n > len(DEV_SEEDS)):
        n = args.n or FINAL_N
        seeds = list(range(FINAL_SEED_START, FINAL_SEED_START + n))
    else:
        seeds = DEV_SEEDS[: args.n or len(DEV_SEEDS)]
    if args.split == "test" and set(seeds) & set(DEV_SEEDS):
        print(f"!!! TEST split refused: seeds {sorted(set(seeds) & set(DEV_SEEDS))} were used in "
              f"development. Use fresh seeds.")
        return

    split_label = "VALIDATION" if args.split == "val" else "TEST"
    split_tag = "val" if args.split == "val" else "test"
    prop_label = model_label("proposed", args.fusion)
    out = results_dir(args.tagger, reg, args.split)
    print(f"=== Sweep on {split_label} set | tagger {args.tagger} | fusion {args.fusion} "
          f"| models {args.models} | {len(seeds)} seeds: {seeds} ===")
    print(f"=== Training settings: {config} ===")
    print(f"=== Results folder: {out} ===")
    if args.split == "test":
        print("!!! FINAL ROUND on the TEST set. If it gets interrupted, run the SAME command "
              "again with --resume added.")

    # If the baseline isn't being re-run, confirm its stored results exist BEFORE training.
    stored_baseline = None
    if "baseline" not in args.models:
        stored_baseline, missing = load_stored(args.tagger, "baseline", seeds, args.split, reg)
        if missing:
            print(f"\n!!! No stored baseline {split_tag} results for seeds {missing} under "
                  f"{args.tagger} with training settings {config}.\n"
                  f"    Run with --models baseline first.")
            return
        print("Pairing against stored baseline results (baseline is deterministic).")

    prop_train, prop_val, prop_test, embedding_matrix, feature_dim = build_datasets(True, args.tagger)
    data = {
        "proposed": (prop_train, prop_val, prop_test),
        "baseline": tuple(strip_features(d) for d in (prop_train, prop_val, prop_test)),
    }

    labels = [model_label(m, args.fusion) for m in args.models]
    results = {lbl: {k: [] for k in METRICS} for lbl in labels}
    total = len(seeds) * len(args.models)
    k = 0
    for seed in seeds:
        for m, lbl in zip(args.models, labels):
            k += 1
            fd = feature_dim if m == "proposed" else 0
            print(f"\n--- run {k}/{total}: {lbl} | seed {seed} ---")
            t0 = time.time()
            saved = os.path.join(out, lbl, f"seed_{seed}", metrics_filename(args.split))
            if args.resume and os.path.exists(saved):
                with open(saved, encoding="utf-8") as f:
                    met = json.load(f)
                for key in METRICS:
                    results[lbl][key].append(met[key])
                print(f">>> already done, loaded saved result: macro_f1={met['macro_f1']:.4f}")
                continue
            met = run_one(m, args.fusion, data[m], embedding_matrix, fd, seed, args.monitor,
                          args.tagger, args.split, verbose=not args.no_epoch_log, reg=reg)
            for key in METRICS:
                results[lbl][key].append(met[key])
            print(f">>> {split_label} macro_f1={met['macro_f1']:.4f}  acc={met['accuracy']:.4f}"
                  f"  ({time.time() - t0:.0f}s)")

    base_key = "baseline"
    if stored_baseline is not None:
        base_key = "baseline (stored)"
        results = {base_key: stored_baseline, **results}

    print("\n" + "=" * 70)
    print(f"{split_label} SUMMARY | tagger {args.tagger} | fusion {args.fusion} | "
          f"{len(seeds)} seeds: {seeds}")
    print(f"training settings: {config}")
    print("=" * 70)
    for lbl, vals in results.items():
        print(f"{lbl}:")
        for key, v in vals.items():
            summarize(key, v)

    if base_key in results and prop_label in results:
        b = results[base_key]["macro_f1"]
        p_ = results[prop_label]["macro_f1"]
        diffs = [x - y for x, y in zip(p_, b)]
        wins = sum(1 for d in diffs if d > 0)
        print(f"\nPaired comparison ({prop_label} - baseline), {split_label} macro F1"
              f"{' [PRIMARY, Ch3-F.3]' if args.split == 'test' else ''}:")
        print("  per seed: " + "  ".join(f"{s}:{d:+.4f}" for s, d in zip(seeds, diffs)))
        print(f"  mean difference: {statistics.mean(diffs):+.4f}")
        print(f"  {prop_label} wins {wins}/{len(diffs)} seeds")
        if len(diffs) > 1:
            t, pv = stats.ttest_rel(p_, b)
            one_tailed = pv / 2 if t > 0 else 1 - pv / 2
            half = stats.t.ppf(0.975, len(diffs) - 1) * statistics.stdev(diffs) / math.sqrt(len(diffs))
            mean_d = statistics.mean(diffs)
            print(f"  95% CI of the difference: [{mean_d - half:+.4f}, {mean_d + half:+.4f}]")
            print(f"  paired t-test (two-sided): t={t:.3f}, p={pv:.4f}")
            print(f"  one-tailed p (proposed > baseline): {one_tailed:.4f}")
            if args.split == "val":
                print("  NOTE: with few seeds this is indicative only.")
            bw, pw = results[base_key]["weighted_f1"], results[prop_label]["weighted_f1"]
            tw, pvw = stats.ttest_rel(pw, bw)
            print(f"  (secondary) weighted F1 difference {statistics.mean(pw) - statistics.mean(bw):+.4f}, "
                  f"paired t={tw:.3f}, p={pvw:.4f}")

        if args.split == "val":
            need = math.ceil(BAR_WIN_FRACTION * len(diffs))
            mean_ok = statistics.mean(p_) > statistics.mean(b)
            wins_ok = wins >= need
            verdict = "PASS" if (mean_ok and wins_ok) else "FAIL"
            print(f"\nPre-registered bar: mean above baseline AND wins >= {need}/{len(diffs)}")
            print(f"  mean {statistics.mean(p_):.4f} vs baseline {statistics.mean(b):.4f} -> "
                  f"{'yes' if mean_ok else 'no'}")
            print(f"  wins {wins}/{len(diffs)} -> {'yes' if wins_ok else 'no'}")
            print(f"  RESULT: {verdict}")
            print("\n(The 0.84 benchmark test is skipped on validation -- it is a test-set number.)")
    if args.split == "test" and len(seeds) > 1:
        print(f"\nOne-sample t-test vs Cosme & De Leon (2024) benchmark, weighted F1 = {BENCHMARK_F1} (Ch3-F.4):")
        for lbl, vals in results.items():
            w = vals["weighted_f1"]
            t1, pv1 = stats.ttest_1samp(w, BENCHMARK_F1)
            print(f"  {lbl:<20} mean weighted F1 = {statistics.mean(w):.4f} "
                  f"({statistics.mean(w) - BENCHMARK_F1:+.4f} vs benchmark)  t={t1:.3f}, two-sided p={pv1:.4g}")

    os.makedirs(out, exist_ok=True)
    suffix = "" if args.fusion == "concat" else f"_{args.fusion}"
    fname = f"sweep_summary_{split_tag}{suffix}.json"
    with open(os.path.join(out, fname), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "split": args.split, "monitor": args.monitor,
                   "tagger": args.tagger, "fusion": args.fusion, "models": args.models,
                   "training_settings": reg, "raw": results}, f, indent=2)
    print(f"\nSaved to {out}/{fname}")


if __name__ == "__main__":
    main()

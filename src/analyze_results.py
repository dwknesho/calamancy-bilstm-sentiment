"""
Aggregates the 30 saved per-seed test-set runs of each model into the
confusion matrix, per-class true/false positives and negatives, overall
metrics, and the paired/one-sample t-tests used in Chapter IV. Reads the
already-saved metrics.json files -- trains nothing.
"""
import argparse
import glob
import itertools
import json
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as spstats

from run_experiment import CHOSEN_REG, results_dir
from tokenization import DEFAULT_TAGGER

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
DEFAULT_MODELS = ["baseline", "proposed-projection"]
OVERALL_KEYS = ["accuracy", "macro_averaged_accuracy", "macro_precision", "macro_recall",
                "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]
BENCHMARK_F1 = 0.84  # Cosme & De Leon (2024), weighted F1, test-set only


def load_runs(tagger: str, label: str, split: str):
    filename = "metrics.json" if split == "test" else "metrics_val.json"
    pattern = os.path.join(results_dir(tagger, CHOSEN_REG, split), label, "seed_*", filename)
    files = sorted(glob.glob(pattern))
    runs = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def mean_confusion_matrix(runs) -> np.ndarray:
    """(3, 3) counts, averaged over runs. Rows = true label, columns = predicted."""
    return np.array([r["confusion_matrix"] for r in runs], dtype=float).mean(axis=0)


def per_class_breakdown(cm: np.ndarray) -> list[dict]:
    """One-vs-rest TP/FP/FN/TN and precision/recall/F1 for each class, from the matrix."""
    total = cm.sum()
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"class": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": precision, "recall": recall, "f1": f1})
    return rows


def overall_stats(runs) -> dict:
    return {k: (statistics.mean(r[k] for r in runs),
                statistics.stdev(r[k] for r in runs) if len(runs) > 1 else 0.0)
            for k in OVERALL_KEYS}


def by_seed(runs, key: str) -> dict:
    return {r["config"]["seed"]: r[key] for r in runs}


def paired_comparison(label_a: str, runs_a, label_b: str, runs_b, key: str = "macro_f1") -> dict:
    """Paired t-test between two models, matched by seed (not list order)."""
    a_by_seed, b_by_seed = by_seed(runs_a, key), by_seed(runs_b, key)
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    a = [a_by_seed[s] for s in seeds]
    b = [b_by_seed[s] for s in seeds]
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = statistics.mean(diffs)
    wins = sum(1 for d in diffs if d > 0)
    t, p_two = spstats.ttest_rel(a, b)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    half = (spstats.t.ppf(0.975, n - 1) * statistics.stdev(diffs) / math.sqrt(n)) if n > 1 else 0.0
    return {"label_a": label_a, "label_b": label_b, "key": key, "seeds": seeds, "n": n,
            "mean_diff": mean_d, "ci": (mean_d - half, mean_d + half), "wins": wins,
            "t": t, "p_two": p_two, "p_one": p_one}


def one_sample_vs_benchmark(label: str, runs, key: str = "weighted_f1") -> dict:
    """Ch3-F.4: one-sample t-test against the 0.84 benchmark. One-tailed direction
    tests mean > 0.84 ("outperforms"), per SOP3/Objective 3 -- see the note on
    hypothesis 2's wording in docs section 10."""
    values = [r[key] for r in runs]
    n = len(values)
    mean_v = statistics.mean(values)
    t, p_two = spstats.ttest_1samp(values, BENCHMARK_F1)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return {"label": label, "key": key, "n": n, "mean": mean_v,
            "diff": mean_v - BENCHMARK_F1, "t": t, "p_two": p_two, "p_one": p_one}


def plot_matrix(cm: np.ndarray, title: str, out_path: str, as_percent: bool) -> None:
    data = cm / cm.sum(axis=1, keepdims=True) * 100 if as_percent else cm
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    vmax = data.max()
    for i in range(3):
        for j in range(3):
            text = f"{data[i, j]:.1f}%" if as_percent else f"{data[i, j]:.1f}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if data[i, j] > vmax / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_side_by_side(matrices: dict, out_path: str, as_percent: bool) -> None:
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6))
    axes = [axes] if n == 1 else list(axes)
    for ax, (label, cm) in zip(axes, matrices.items()):
        data = cm / cm.sum(axis=1, keepdims=True) * 100 if as_percent else cm
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(label)
        vmax = data.max()
        for i in range(3):
            for j in range(3):
                text = f"{data[i, j]:.1f}%" if as_percent else f"{data[i, j]:.1f}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if data[i, j] > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tagger", default=DEFAULT_TAGGER)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="results-folder names, e.g. baseline proposed-projection proposed-projection-morph")
    p.add_argument("--out", default=None, help="output folder (default: results/.../analysis/)")
    args = p.parse_args()

    out_dir = args.out or os.path.join(results_dir(args.tagger, CHOSEN_REG, args.split), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    split_label = "VALIDATION" if args.split == "val" else "TEST"
    md = [f"# Results analysis -- {split_label} set, {args.tagger}\n"]
    matrices = {}
    all_runs = {}

    for label in args.models:
        runs = load_runs(args.tagger, label, args.split)
        if not runs:
            print(f"!!! no saved runs for '{label}' under {args.tagger}/{args.split} -- skipping")
            continue
        all_runs[label] = runs

        n = len(runs)
        cm = mean_confusion_matrix(runs)
        rows = per_class_breakdown(cm)
        stats = overall_stats(runs)
        matrices[label] = cm

        print(f"\n{'=' * 72}\n{label}  ({n} runs)\n{'=' * 72}")
        print(f"mean confusion matrix (avg reviews per run; rows=true, cols=pred; {CLASS_NAMES}):")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:9s} " + "  ".join(f"{v:7.1f}" for v in cm[i]))

        print(f"\n{'class':10s}{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>8}{'precision':>11}{'recall':>9}{'F1':>8}")
        for r in rows:
            print(f"{r['class']:10s}{r['tp']:8.1f}{r['fp']:8.1f}{r['fn']:8.1f}{r['tn']:8.1f}"
                  f"{r['precision']:11.4f}{r['recall']:9.4f}{r['f1']:8.4f}")

        print(f"\noverall (mean +/- sd over {n} runs):")
        for k, (mean, sd) in stats.items():
            print(f"  {k:24s} {mean:.4f} +/- {sd:.4f}")

        plot_matrix(cm, f"{label} -- mean confusion matrix ({n} runs)",
                   os.path.join(out_dir, f"{label}_confusion_counts.png"), as_percent=False)
        plot_matrix(cm, f"{label} -- mean confusion matrix (row %)",
                   os.path.join(out_dir, f"{label}_confusion_pct.png"), as_percent=True)

        md.append(f"## {label} ({n} runs)\n")
        md.append("**Mean confusion matrix** (rows = true label, columns = predicted)\n")
        md.append("| True \\\\ Predicted | " + " | ".join(CLASS_NAMES) + " |")
        md.append("|---|" + "---|" * len(CLASS_NAMES))
        for i, name in enumerate(CLASS_NAMES):
            md.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in cm[i]) + " |")
        md.append("\n**Per-class breakdown**\n")
        md.append("| Class | TP | FP | FN | TN | Precision | Recall | F1 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['class']} | {r['tp']:.1f} | {r['fp']:.1f} | {r['fn']:.1f} | {r['tn']:.1f} "
                      f"| {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
        md.append("\n**Overall metrics** (mean ± sd)\n")
        md.append("| Metric | Mean | SD |")
        md.append("|---|---|---|")
        for k, (mean, sd) in stats.items():
            md.append(f"| {k} | {mean:.4f} | {sd:.4f} |")
        md.append("")

    if len(matrices) > 1:
        plot_side_by_side(matrices, os.path.join(out_dir, "confusion_comparison_pct.png"), as_percent=True)
        plot_side_by_side(matrices, os.path.join(out_dir, "confusion_comparison_counts.png"), as_percent=False)

    # -- statistical comparisons (paired t-tests between every pair of models loaded) --
    # When "baseline" is one of the two, it is always the SECOND argument (the
    # subtrahend), so mean_diff = proposed - baseline and the one-tailed test
    # matches Ch3-F.3's Ha ("calamanCy-enhanced model outperforms the baseline").
    # Pairs not involving baseline have no hypothesis-mandated direction; the
    # one-tailed p shown for those is purely descriptive (label order as given).
    if len(all_runs) > 1:
        print(f"\n{'=' * 72}\nSTATISTICAL COMPARISONS\n{'=' * 72}")
        md.append("## Statistical comparisons\n")
        for x, y in itertools.combinations(all_runs.items(), 2):
            (la, ra), (lb, rb) = (y, x) if x[0] == "baseline" else (x, y)
            primary = {"baseline", "proposed-projection"} == {la, lb}
            tag = "PRIMARY, Ch3-F.3, Ha: calamanCy model outperforms baseline" if primary else \
                  "secondary -- no hypothesis-mandated direction"
            c = paired_comparison(la, ra, lb, rb, "macro_f1")
            cw = paired_comparison(la, ra, lb, rb, "weighted_f1")
            print(f"\nPaired comparison ({la} - {lb}), {split_label} macro F1 [{tag}]:")
            print(f"  n = {c['n']} matched seeds | mean diff = {c['mean_diff']:+.4f} "
                  f"| {la} wins {c['wins']}/{c['n']}")
            print(f"  95% CI of the difference: [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]")
            print(f"  paired t-test: t={c['t']:.3f}, two-sided p={c['p_two']:.4f}, "
                  f"one-tailed p ({la} > {lb})={c['p_one']:.4f}")
            print(f"  (secondary) weighted F1 mean diff = {cw['mean_diff']:+.4f}, "
                  f"two-sided p={cw['p_two']:.4f}")

            md.append(f"**{la} vs {lb}** ({tag}, n = {c['n']} matched seeds)\n")
            md.append("| Metric | Mean diff | 95% CI | Wins | t | two-sided p | one-tailed p |")
            md.append("|---|---|---|---|---|---|---|")
            md.append(f"| macro F1 | {c['mean_diff']:+.4f} | [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] "
                      f"| {c['wins']}/{c['n']} | {c['t']:.3f} | {c['p_two']:.4f} | {c['p_one']:.4f} |")
            md.append(f"| weighted F1 | {cw['mean_diff']:+.4f} | [{cw['ci'][0]:+.4f}, {cw['ci'][1]:+.4f}] "
                      f"| {cw['wins']}/{cw['n']} | {cw['t']:.3f} | {cw['p_two']:.4f} | {cw['p_one']:.4f} |")
            md.append("")

    # -- one-sample t-test vs the Cosme & De Leon (2024) benchmark (test set only) --
    if args.split == "test":
        print(f"\nOne-sample t-test vs Cosme & De Leon (2024) benchmark, weighted F1 = {BENCHMARK_F1} (Ch3-F.4):")
        print("  one-tailed direction tested: mean > 0.84 (SOP3/Objective 3 'outperforms';")
        print("  hypothesis 2's own wording says 'comparable to', which a one-sample t-test")
        print("  cannot test directly -- see docs section 10.)")
        md.append(f"## One-sample t-test vs benchmark (weighted F1 = {BENCHMARK_F1})\n")
        md.append("One-tailed direction tested: mean > 0.84, per SOP3/Objective 3 (\"outperforms\"). "
                  "Hypothesis 2 as worded (\"comparable to\") is not a one-sample-t-test claim; "
                  "see docs section 10.\n")
        md.append("| Model | n | Mean weighted F1 | Diff vs benchmark | t | two-sided p | one-tailed p (>0.84) |")
        md.append("|---|---|---|---|---|---|---|")
        for label, runs in all_runs.items():
            o = one_sample_vs_benchmark(label, runs)
            print(f"  {label:<28} n={o['n']:>2}  mean={o['mean']:.4f}  "
                  f"diff={o['diff']:+.4f}  t={o['t']:.3f}  two-sided p={o['p_two']:.4g}  "
                  f"one-tailed p={o['p_one']:.4g}")
            md.append(f"| {label} | {o['n']} | {o['mean']:.4f} | {o['diff']:+.4f} "
                      f"| {o['t']:.3f} | {o['p_two']:.4g} | {o['p_one']:.4g} |")
        md.append("")
    else:
        print("\n(Benchmark test skipped on validation -- it is a test-set number.)")

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nSaved tables to {md_path}")
    print(f"Saved confusion matrix images to {out_dir}/")


if __name__ == "__main__":
    main()

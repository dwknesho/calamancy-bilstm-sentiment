import quiet  # noqa: F401  (must be imported before torch)
import argparse
import json
import os
import statistics

from run_experiment import DEFAULT_REG, metrics_filename, model_label, reg_name, results_dir
from tokenization import DEFAULT_TAGGER

DEV_SEEDS = [42, 7, 123, 2024, 31337]
MIN_GAIN = 0.003

GRID = {
    "C0": {},
    "C1": {"dropout": 0.5},
    "C2": {"input_dropout": 0.25},
    "C3": {"weight_decay": 0.01},
    "C4": {"lr": 0.0005},
    "C5": {"input_dropout": 0.25, "weight_decay": 0.01},
}


def load_runs(tagger, reg, label):
    """Per-seed (macro_f1, best_epoch, train_acc_at_best, val_loss_at_best), or None if incomplete."""
    runs = []
    for s in DEV_SEEDS:
        path = os.path.join(results_dir(tagger, reg), label, f"seed_{s}", metrics_filename("val"))
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        best = min(m["history"], key=lambda e: e["val_loss"])
        runs.append((m["macro_f1"], best["epoch"], best["train_acc"], best["val_loss"]))
    return runs


def mean_of(runs, i):
    return statistics.mean(r[i] for r in runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tagger", default=DEFAULT_TAGGER)
    p.add_argument("--fusion", default="projection")
    args = p.parse_args()
    prop = model_label("proposed", args.fusion)

    rows, incomplete = {}, []
    for name, overrides in GRID.items():
        reg = {**DEFAULT_REG, **overrides}
        base, props = load_runs(args.tagger, reg, "baseline"), load_runs(args.tagger, reg, prop)
        if base is None or props is None:
            incomplete.append(f"{name} ({reg_name(reg) or 'defaults'})")
            continue
        rows[name] = {"reg": reg, "base": base, "prop": props}

    print("=" * 96)
    print(f"PHASE 3 -- VALIDATION macro F1 | tagger {args.tagger} | proposed = {prop} | seeds {DEV_SEEDS}")
    print("=" * 96)
    if "C0" not in rows:
        print("C0 (default settings) results are missing -- run `python src/run_sweep.py` first.")
        return

    c0_score = (mean_of(rows["C0"]["base"], 0) + mean_of(rows["C0"]["prop"], 0)) / 2
    print(f"{'config':<4} {'settings':<36} {'baseline':>8} {'proposed':>8} {'BOTH':>7} {'vs C0':>7} "
          f"{'best ep':>7} {'train acc':>9} {'prop-base':>9} {'wins':>5}")
    for name, r in rows.items():
        b, pr = mean_of(r["base"], 0), mean_of(r["prop"], 0)
        score = (b + pr) / 2
        r["score"] = score
        both = r["base"] + r["prop"]
        wins = sum(1 for x, y in zip(r["prop"], r["base"]) if x[0] > y[0])
        print(f"{name:<4} {reg_name(r['reg']) or 'defaults':<36} {b:>8.4f} {pr:>8.4f} {score:>7.4f} "
              f"{score - c0_score:>+7.4f} {mean_of(both, 1):>7.1f} {mean_of(both, 2):>9.3f} "
              f"{pr - b:>+9.4f} {wins:>3}/5")

    print("\nColumns: BOTH = mean of baseline and proposed (the selection score); best ep / train acc =")
    print("averages at the saved checkpoint (a later best epoch and lower train acc = less memorization);")
    print("prop-base and wins = proposed vs baseline under that config, shown for information only.")

    if incomplete:
        print(f"\nNot run yet (skipped): {', '.join(incomplete)}")

    winner = max(rows, key=lambda n: rows[n]["score"])
    gain = rows[winner]["score"] - c0_score
    print("\nSelection rule: highest BOTH score, and it must beat C0 by more than "
          f"{MIN_GAIN}.")
    if winner == "C0" or gain <= MIN_GAIN:
        verdict = f"KEEP C0 (best other config gains {gain:+.4f}, not more than {MIN_GAIN})" \
            if winner != "C0" else "KEEP C0 (no config scored higher)"
    else:
        c0_runs = rows["C0"]["base"] + rows["C0"]["prop"]
        w_runs = rows[winner]["base"] + rows[winner]["prop"]
        beats = sum(1 for x, y in zip(w_runs, c0_runs) if x[0] > y[0])
        verdict = (f"ADOPT {winner} ({reg_name(rows[winner]['reg'])}): {gain:+.4f} over C0, "
                   f"better on {beats}/{len(c0_runs)} individual runs")
    print(f"  RESULT: {verdict}")
    if incomplete:
        print("  (provisional -- not every config has been run)")


if __name__ == "__main__":
    main()

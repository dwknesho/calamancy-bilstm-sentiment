import quiet  # noqa: F401  (must be imported before calamancy)
import argparse
import collections
import os

import pandas as pd

from data_prep import load_all_splits
from tokenization import tag_series

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data", "processed")

DEFAULT_TAGGERS = ["tl_calamancy_md-0.1.0", "tl_calamancy_md-0.2.0", "tl_calamancy_lg-0.2.0"]
NEGATIONS = {"di", "hindi", "huwag", "wala", "not", "no"}
SPLITS = ("train", "val", "test")


def load(tagger, limit):
    """Full run -> cached tags. --limit -> tag a small head of each split, no caching."""
    if not limit:
        return load_all_splits(DATA_DIR, tagger)
    out = {}
    for split in SPLITS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv")).head(limit)
        tagged = tag_series(df["clean_review"].astype(str).tolist(), tagger=tagger, progress_every=0)
        for key, vals in tagged.items():
            df[key] = vals
        out[split] = df
    return out


def flat(splits, key, which=("train", "val")):
    return [x for s in which for seq in splits[s][key] for x in seq]


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def rule(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--taggers", nargs="+", default=DEFAULT_TAGGERS)
    p.add_argument("--limit", type=int, default=0, help="smoke test on N reviews per split (no caching)")
    args = p.parse_args()

    data = {}
    for tagger in args.taggers:
        print(f"\n>>> {tagger}")
        data[tagger] = load(tagger, args.limit)

    ref = args.taggers[0]

    # ------------------------------------------------------------------ 1. validity
    rule(f"1. TOKENIZATION -- identical to reference ({ref})?   [all splits]")
    token_identical = {}
    for tagger in args.taggers[1:]:
        same = total = 0
        n_ref = n_new = 0
        for s in SPLITS:
            for a, b in zip(data[ref][s]["tokens"], data[tagger][s]["tokens"]):
                total += 1
                same += a == b
                n_ref += len(a)
                n_new += len(b)
        token_identical[tagger] = same == total
        verdict = "IDENTICAL -- baseline results carry over" if same == total else \
                  "DIFFERENT -- baseline must be re-run under this tagger"
        print(f"  {tagger:<24} {same}/{total} reviews identical ({pct(same, total):.2f}%)"
              f" | tokens {n_ref:,} -> {n_new:,}  => {verdict}")

    # ------------------------------------------------------------------ 2. quality
    rule("2. TAG QUALITY   [train + val only -- test set not used for this decision]")
    summary = {}
    for tagger in args.taggers:
        sp = data[tagger]
        pos, dep, morph = flat(sp, "pos"), flat(sp, "dep"), flat(sp, "morph")
        toks = flat(sp, "tokens")
        pc, dc = collections.Counter(pos), collections.Counter(dep)
        n = len(dep)
        neg = [(t.lower(), p_, d) for t, p_, d in zip(toks, pos, dep) if t.lower() in NEGATIONS]
        neg_c = collections.Counter((p_, d) for _, p_, d in neg)
        summary[tagger] = {
            "generic_dep": pct(dc.get("dep", 0), n),
            "dep_labels": len(dc),
            "pos_tags": len(pc),
            "part": pct(pc.get("PART", 0), n),
            "intj": pc.get("INTJ", 0),
            "morph_cov": pct(sum(1 for m in morph if m), n),
            "neg_top": neg_c.most_common(1)[0] if neg_c else (("-", "-"), 0),
            "neg_n": len(neg),
        }
        print(f"\n  {tagger}   ({n:,} tokens)")
        print(f"    generic `dep` fallback : {summary[tagger]['generic_dep']:5.1f}%   "
              f"({len(dc)} distinct dependency labels)")
        print("    top dependency labels  : " +
              ", ".join(f"{lbl} {pct(c, n):.1f}%" for lbl, c in dc.most_common(8)))
        print("    top POS tags           : " +
              ", ".join(f"{lbl} {pct(c, n):.1f}%" for lbl, c in pc.most_common(8)))
        print(f"    PART {summary[tagger]['part']:.2f}% | INTJ count {summary[tagger]['intj']} "
              f"| tokens with morphology {summary[tagger]['morph_cov']:.1f}%")
        if neg:
            print(f"    negation words ({len(neg):,}) most often tagged as: " +
                  ", ".join(f"{p_}/{d} {pct(c, len(neg)):.0f}%" for (p_, d), c in neg_c.most_common(3)))

    # ------------------------------------------------------------------ 3. agreement
    rule(f"3. TOKEN-LEVEL AGREEMENT with {ref}   [train + val, identical-token reviews only]")
    for tagger in args.taggers[1:]:
        same_pos = same_dep = n = 0
        for s in ("train", "val"):
            r, t = data[ref][s], data[tagger][s]
            for tk_a, tk_b, pa, pb, da, db in zip(r["tokens"], t["tokens"], r["pos"], t["pos"], r["dep"], t["dep"]):
                if tk_a != tk_b:
                    continue
                for x1, x2, y1, y2 in zip(pa, pb, da, db):
                    n += 1
                    same_pos += x1 == x2
                    same_dep += y1 == y2
        print(f"  {tagger:<24} POS agree {pct(same_pos, n):5.1f}% | dependency label agree {pct(same_dep, n):5.1f}%")

    # ------------------------------------------------------------------ summary
    rule("SUMMARY")
    print(f"  {'tagger':<24}{'generic dep':>12}{'dep labels':>12}{'POS tags':>10}{'PART %':>8}{'INTJ':>6}"
          f"{'tokens =ref':>13}")
    for tagger in args.taggers:
        s = summary[tagger]
        same = "(ref)" if tagger == ref else ("yes" if token_identical[tagger] else "NO")
        print(f"  {tagger:<24}{s['generic_dep']:>11.1f}%{s['dep_labels']:>12}{s['pos_tags']:>10}"
              f"{s['part']:>8.2f}{s['intj']:>6}{same:>13}")

    print("\n  How to read this (decision rule set BEFORE running):")
    print("   - Upgrade is VERIFIED if generic `dep` drops substantially from the reference")
    print("     (target: well under ~30%) and the label set gets richer.")
    print("   - If tokens are NOT identical, the baseline must be re-run under the new tagger")
    print("     before any baseline-vs-proposed comparison uses it.")
    print("   - If md-0.2.0 is close to lg-0.2.0, the VERSION upgrade is doing the work, not size.")
    if args.limit:
        print(f"\n  [smoke test: only {args.limit} reviews per split -- numbers are NOT representative]")


if __name__ == "__main__":
    main()

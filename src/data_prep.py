"""Loads the processed splits and attaches calamanCy tags, cached to disk per
tagger and validated against a hash of the source text."""
import hashlib
import os
import pickle

import numpy as np
import pandas as pd

from tokenization import DEFAULT_TAGGER, tag_series

CACHE_VERSION = "v2"  # cache format version (v2 added morph + lemma)

ANNOTATIONS = ("tokens", "pos", "dep", "morph", "lemma")

MAX_LEN = 128  # longest FiReCS review is 57 tokens, so this never truncates


def cache_path(data_dir: str, split: str, tagger: str) -> str:
    return os.path.join(data_dir, f"{split}_tagged__{tagger}.pkl")


def _text_hash(texts, tagger: str) -> str:
    h = hashlib.sha256()
    h.update(f"{CACHE_VERSION}|{tagger}".encode("utf-8"))
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_split(data_dir: str, split: str, tagger: str = DEFAULT_TAGGER) -> pd.DataFrame:
    """Loads split.csv + tags (from cache, or re-tags if missing/stale)."""
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    df["label"] = df["label"].astype(int)
    texts = df["clean_review"].astype(str).tolist()
    digest = _text_hash(texts, tagger)
    path = cache_path(data_dir, split, tagger)

    if os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("hash") == digest and len(cached.get("tokens", [])) == len(df):
            for key in ANNOTATIONS:
                df[key] = cached[key]
            return df
        print(f"  cache for {split} [{tagger}] is stale -- re-tagging")

    print(f"  tagging {split} with {tagger} ({len(df)} reviews)...")
    tagged = tag_series(texts, tagger=tagger)
    with open(path, "wb") as f:
        pickle.dump({"hash": digest, **tagged}, f)
    for key in ANNOTATIONS:
        df[key] = tagged[key]
    return df


def load_all_splits(data_dir: str, tagger: str = DEFAULT_TAGGER) -> dict:
    return {s: load_split(data_dir, s, tagger) for s in ("train", "val", "test")}


def truncate(seq, max_len: int = MAX_LEN):
    """Truncates any per-token sequence (ids, tags, or feature rows)."""
    return seq[:max_len] if len(seq) > max_len else seq


def build_feature_arrays(df, pos_set, dep_set, max_len: int = MAX_LEN):
    """Per-review (seq_len, P+D) one-hot arrays for the proposed model."""
    from features import encode_token_features

    out = []
    for pos, dep in zip(df["pos"], df["dep"]):
        feats = encode_token_features(truncate(pos, max_len), truncate(dep, max_len), pos_set, dep_set)
        out.append(np.asarray(feats, dtype=np.float32))
    return out

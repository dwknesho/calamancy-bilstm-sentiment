"""Shared calamanCy tokenizer for both models: one pass produces tokens +
POS + dependency + morph + lemma together."""
import sys

import calamancy

DEFAULT_TAGGER = "tl_calamancy_md-0.2.0"
_NLPS = {}


def _get_nlp(tagger: str = DEFAULT_TAGGER):
    """Loads a calamanCy pipeline once per name and reuses it."""
    if tagger not in _NLPS:
        _NLPS[tagger] = calamancy.load(tagger)
    return _NLPS[tagger]


def tokenize(text: str, tagger: str = DEFAULT_TAGGER) -> list[str]:
    """Tokenizes a single (already-cleaned) review string into token strings."""
    return [tok.text for tok in _get_nlp(tagger)(text)]


def tag_series(texts, tagger: str = DEFAULT_TAGGER, progress_every: int = 500) -> dict:
    """Tags all texts, returns {"tokens","pos","dep","morph","lemma"} per review.
    Prints progress since tagging thousands of reviews takes a few minutes."""
    nlp = _get_nlp(tagger)
    texts = list(texts)
    out = {"tokens": [], "pos": [], "dep": [], "morph": [], "lemma": []}
    for i, doc in enumerate(nlp.pipe(texts), start=1):
        out["tokens"].append([t.text for t in doc])
        out["pos"].append([t.pos_ for t in doc])
        out["dep"].append([t.dep_ for t in doc])
        out["morph"].append([str(t.morph) for t in doc])
        out["lemma"].append([t.lemma_ for t in doc])
        if progress_every and (i % progress_every == 0 or i == len(texts)):
            print(f"    tagged {i:>5}/{len(texts)}", flush=True)
    return out

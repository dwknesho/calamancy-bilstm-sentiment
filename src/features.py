"""Builds one-hot POS + dependency features for the proposed model. Tagsets
are built from train+val+test combined."""
import json

import numpy as np

UNK_TAG = "<UNK_TAG>"


def build_tagset(tag_lists) -> dict[str, int]:
    """{tag: index} from per-review tag lists; UNK_TAG=0, rest sorted for
    reproducible ordering."""
    unique = set()
    for tags in tag_lists:
        unique.update(tags)
    ordered = [UNK_TAG] + sorted(unique)
    return {tag: i for i, tag in enumerate(ordered)}


def encode_tags(tags: list[str], tagset: dict[str, int]) -> np.ndarray:
    """One-hot encodes a review's tag sequence -> (seq_len, len(tagset)) float32."""
    out = np.zeros((len(tags), len(tagset)), dtype=np.float32)
    unk = tagset[UNK_TAG]
    for i, tag in enumerate(tags):
        out[i, tagset.get(tag, unk)] = 1.0
    return out


def encode_token_features(
    pos_tags: list[str], dep_tags: list[str], pos_set: dict[str, int], dep_set: dict[str, int]
) -> np.ndarray:
    """POS one-hot + dep one-hot per token -> (seq_len, P+D). FastText gets
    concatenated onto this later, inside the model's forward()."""
    assert len(pos_tags) == len(dep_tags), "POS/dep tag sequences must align"
    return np.concatenate(
        [encode_tags(pos_tags, pos_set), encode_tags(dep_tags, dep_set)], axis=1
    )


def save_tagsets(pos_set: dict, dep_set: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pos": pos_set, "dep": dep_set}, f, ensure_ascii=False, indent=2)


def load_tagsets(path: str) -> tuple[dict, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pos"], data["dep"]

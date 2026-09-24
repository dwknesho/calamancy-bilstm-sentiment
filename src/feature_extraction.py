import os

import fasttext
import fasttext.util
import numpy as np

_MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "fasttext")


def load_fasttext_model(lang: str = "tl"):
    """Loads FastText for `lang`, downloading it once if needed. Default: Tagalog."""
    bin_path = os.path.join(_MODEL_CACHE_DIR, f"cc.{lang}.300.bin")
    if not os.path.exists(bin_path):
        cwd = os.getcwd()
        os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
        os.chdir(_MODEL_CACHE_DIR)
        try:
            fasttext.util.download_model(lang, if_exists="ignore")
        finally:
            os.chdir(cwd)
    return fasttext.load_model(bin_path)


def get_word_vector(model, word: str) -> np.ndarray:
    """FastText vector for one word."""
    return model.get_word_vector(word)


def get_review_vectors(model, tokens: list) -> np.ndarray:
    """FastText vector per token -> (num_tokens, 300) matrix."""
    return np.array([get_word_vector(model, tok) for tok in tokens])
import fasttext
import fasttext.util
import numpy as np


def load_fasttext_model(lang: str = "tl"):
    """
    Downloads (if not already present) and loads the pre-trained
    Facebook FastText model for the given language code.
    Default 'tl' = Tagalog (cc.tl.300.bin).
    """
    fasttext.util.download_model(lang, if_exists='ignore')  # downloads cc.tl.300.bin
    model = fasttext.load_model(f'cc.{lang}.300.bin')
    return model


def get_word_vector(model, word: str) -> np.ndarray:
    """
    Returns the 300-dim FastText vector for a single word.
    FastText handles OOV words automatically via subword n-grams.
    """
    return model.get_word_vector(word)


def get_review_vectors(model, tokens: list) -> np.ndarray:
    """
    Given a list of tokens (a tokenized review), returns a
    (num_tokens, 300) matrix of FastText vectors — one row per token.
    """
    return np.array([get_word_vector(model, tok) for tok in tokens])
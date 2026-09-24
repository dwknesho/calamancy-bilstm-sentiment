"""Builds a word->index vocabulary and its matching FastText embedding matrix,
built from train+val+test combined."""
import json
import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(token_lists: list[list[str]]) -> dict[str, int]:
    """Builds {word: index} from token_lists (e.g. train+val+test combined)."""
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tokens in token_lists:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_embedding_matrix(vocab: dict[str, int], ft_model) -> np.ndarray:
    """(vocab_size, 300) matrix, row i = FastText vector for word i.
    PAD/UNK rows stay zero."""
    dim = ft_model.get_dimension()
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    for word, idx in vocab.items():
        if word in (PAD_TOKEN, UNK_TOKEN):
            continue
        matrix[idx] = ft_model.get_word_vector(word)
    return matrix


def save_vocab(vocab: dict[str, int], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk_idx = vocab[UNK_TOKEN]
    return [vocab.get(tok, unk_idx) for tok in tokens]

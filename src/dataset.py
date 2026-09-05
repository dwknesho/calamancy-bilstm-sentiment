import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TaglishReviewDataset(Dataset):
    """
    Wraps cleaned review text + labels for the baseline (FastText-only) model.
    Converts each review into a variable-length sequence of 300-dim FastText
    vectors at __getitem__ time (keeps memory usage low vs. precomputing
    every vector for 6.6k+ reviews up front).
    """

    def __init__(self, reviews, labels, ft_model):
        # Accept either a pandas Series or a plain list
        self.reviews = list(reviews)
        self.labels = list(labels)
        self.ft_model = ft_model

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        text = self.reviews[idx]
        tokens = text.split()  # simple whitespace tokenization on clean_review
        if len(tokens) == 0:
            tokens = ["<pad>"]  # guard against reviews that became empty after cleaning

        vectors = np.array([self.ft_model.get_word_vector(tok) for tok in tokens])
        x = torch.tensor(vectors, dtype=torch.float32)          # (seq_len, 300)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)  # 0, 1, or 2
        return x, y


def collate_fn(batch):
    """
    Pads a batch of variable-length (seq_len, 300) sequences to the same
    length so they can be stacked into (batch, max_len, 300) for the LSTM.
    Also returns the true (unpadded) lengths so the model can pack the
    sequence and ignore the padding internally.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)  # (batch, max_len, 300)
    labels = torch.stack(labels)
    return padded, lengths, labels

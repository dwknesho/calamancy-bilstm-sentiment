"""PyTorch Dataset + collate_fn: token ids (+ optional per-token feature
matrices) -> sentiment labels. features=None for the baseline."""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    """
    encoded_sequences: list[list[int]] -- vocab token indices per review.
    labels: list[int] -- 0/1/2 sentiment class per review.
    features: optional list of (seq_len_i, F) float arrays, aligned token-wise
        with encoded_sequences. None for the baseline.
    """

    def __init__(self, encoded_sequences, labels, features=None):
        assert len(encoded_sequences) == len(labels)
        if features is not None:
            assert len(features) == len(labels)
            for seq, feat in zip(encoded_sequences, features):
                assert len(seq) == len(feat), "features must align 1:1 with tokens"
        self.sequences = encoded_sequences
        self.labels = labels
        self.features = features

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def feature_dim(self) -> int:
        """Width of the per-token feature vector (0 when there are none)."""
        return 0 if self.features is None else int(self.features[0].shape[1])

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.features is None:
            return seq, None, label
        feat = torch.tensor(np.asarray(self.features[idx]), dtype=torch.float32)
        return seq, feat, label


def collate_fn(batch):
    """Pads a batch to its own max length. Returns (ids, lengths, features_or_None,
    labels), ready for pack_padded_sequence. Padding is zero and excluded from
    the LSTM via `lengths`."""
    sequences, features, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded_ids = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    if features[0] is None:
        return padded_ids, lengths, None, labels

    padded_feats = pad_sequence(features, batch_first=True, padding_value=0.0)
    return padded_ids, lengths, padded_feats, labels

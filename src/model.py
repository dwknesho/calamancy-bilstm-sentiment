import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    """
    Baseline Bi-LSTM sentiment classifier per Ch3-C.4.2.1:
    FastText embeddings (300-dim) -> Bi-LSTM (hidden=128) -> dropout(0.3)
    -> dense -> softmax over 3 classes (0=Negative, 1=Neutral, 2=Positive).

    NOTE: this same class is reused for the proposed (calamanCy-fused) model —
    just pass a larger input_dim (300 + POS one-hot dim + dep one-hot dim)
    once feature fusion is implemented. No architecture changes needed.
    """

    def __init__(self, input_dim=300, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # hidden_dim * 2 because bidirectional concatenates forward + backward states
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        """
        x: (batch, max_len, input_dim) padded FastText vectors
        lengths: (batch,) true sequence length of each review before padding
        Returns raw logits (batch, num_classes) — softmax is applied inside
        CrossEntropyLoss during training, not here.
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        # h_n: (num_layers * 2, batch, hidden_dim) -> take last layer's fwd/bwd states
        h_forward = h_n[-2]   # forward direction, last layer
        h_backward = h_n[-1]  # backward direction, last layer
        final_hidden = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim*2)

        out = self.dropout(final_hidden)
        logits = self.fc(out)
        return logits

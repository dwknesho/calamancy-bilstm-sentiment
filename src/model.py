"""Bi-LSTM sentiment classifier (Ch3-C.4.2 architecture), shared by both models."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

FUSIONS = ("concat", "separate_norm", "projection")


class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        embedding_matrix,
        hidden_size: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
        freeze_embeddings: bool = True,
        extra_feature_dim: int = 0,
        use_layernorm: bool = True,
        fusion: str = "concat",
        tag_proj_dim: int = 32,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        if fusion != "concat" and not use_layernorm:
            raise ValueError("use_layernorm=False is only defined for fusion='concat'")

        weight = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        embed_dim = weight.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=freeze_embeddings, padding_idx=0
        )
        self.extra_feature_dim = extra_feature_dim
        self.input_dropout = input_dropout
        # no tags -> every fusion strategy reduces to the same model
        self.fusion = fusion if extra_feature_dim > 0 else "concat"

        if self.fusion == "projection":
            lstm_input = embed_dim + tag_proj_dim
        else:
            lstm_input = embed_dim + extra_feature_dim

        if self.fusion == "concat":
            # Identity() when LayerNorm is off (ablation)
            self.input_norm = nn.LayerNorm(lstm_input) if use_layernorm else nn.Identity()
        else:
            self.sem_norm = nn.LayerNorm(embed_dim)
            tag_width = tag_proj_dim if self.fusion == "projection" else extra_feature_dim
            self.tag_norm = nn.LayerNorm(tag_width)

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # created last, so earlier seeds still reproduce
        if self.fusion == "projection":
            self.tag_proj = nn.Linear(extra_feature_dim, tag_proj_dim)

    def forward(self, token_ids, lengths, extra_features=None):
        """Returns raw logits -- CrossEntropyLoss applies softmax, don't do it here."""
        embedded = self.embedding(token_ids)

        if self.extra_feature_dim > 0 and extra_features is None:
            raise ValueError(
                f"Model expects extra_features of width {self.extra_feature_dim}, got None."
            )

        if self.fusion == "concat":
            if self.extra_feature_dim > 0:
                embedded = torch.cat([embedded, extra_features], dim=-1)
            fused = self.input_norm(embedded)
        elif self.fusion == "separate_norm":
            fused = torch.cat([self.sem_norm(embedded), self.tag_norm(extra_features)], dim=-1)
        else:  # projection
            tags = self.tag_norm(self.tag_proj(extra_features))
            fused = torch.cat([self.sem_norm(embedded), tags], dim=-1)

        if self.training and self.input_dropout > 0:
            # one mask per review, shared across its tokens
            keep = 1.0 - self.input_dropout
            mask = fused.new_empty(fused.size(0), 1, fused.size(2)).bernoulli_(keep) / keep
            fused = fused * mask

        packed = pack_padded_sequence(
            fused, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_directions=2, batch, hidden_size)
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.fc(self.dropout(final_hidden))

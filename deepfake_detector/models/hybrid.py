from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class HybridCNNLSTM(nn.Module):
    """Hybrid model that applies a CNN per-frame, followed by an LSTM for temporal modeling.

    Input: (B, T, C, H, W)
    Output: logits (B, num_classes)
    """

    def __init__(
        self,
        cnn_backbone: nn.Module,
        cnn_feature_dim: int,
        temporal_hidden_dim: int = 512,
        temporal_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
        num_classes: int = 2,
        temporal_pool: Literal["last", "mean"] = "last",
    ) -> None:
        super().__init__()
        self.cnn = cnn_backbone
        self.cnn_feature_dim = cnn_feature_dim
        self.temporal_pool = temporal_pool

        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout if temporal_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = temporal_hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        batch_size, num_frames = x.shape[0], x.shape[1]
        x = x.view(batch_size * num_frames, *x.shape[2:])  # (B*T, C, H, W)

        # CNN per-frame
        feats = self.cnn(x)  # (B*T, D)
        feats = feats.view(batch_size, num_frames, -1)  # (B, T, D)

        # LSTM
        outputs, (h_n, c_n) = self.lstm(feats)  # outputs: (B, T, H)

        if self.temporal_pool == "mean":
            pooled = outputs.mean(dim=1)
        else:
            pooled = outputs[:, -1, :]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def extract_cnn_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Helper to expose per-frame CNN features for Grad-CAM or analysis.
        frames: (B, T, C, H, W) -> returns (B, T, D)
        """
        b, t = frames.shape[:2]
        feats = self.cnn(frames.view(b * t, *frames.shape[2:]))
        return feats.view(b, t, -1)

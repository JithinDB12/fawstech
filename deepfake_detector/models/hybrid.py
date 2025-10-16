from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResNeXtBackbone(nn.Module):
    def __init__(self, model_name: str = "resnext50_32x4d", pretrained: bool = True, out_dim: int = 2048):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=[-1])
        feat_info = self.backbone.feature_info[-1]
        self.out_channels = feat_info['num_chs'] if isinstance(feat_info, dict) else out_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[0]  # B x C x H x W
        pooled = self.pool(feats).flatten(1)  # B x C
        return pooled


class TemporalLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x T x D
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        return last


class HybridDeepfakeDetector(nn.Module):
    def __init__(self, backbone_name: str = "resnext50_32x4d", pretrained: bool = True, lstm_hidden: int = 512):
        super().__init__()
        self.backbone = ResNeXtBackbone(backbone_name, pretrained=pretrained)
        self.temporal = TemporalLSTM(self.backbone.out_channels, hidden_dim=lstm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal.out_dim + self.backbone.out_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: B x T x C x H x W
        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)
        spatial_feats = self.backbone(frames)  # (B*T) x D
        d = spatial_feats.size(1)
        spatial_feats_time = spatial_feats.view(b, t, d)
        temporal_feat = self.temporal(spatial_feats_time)  # B x D_t
        # aggregate spatial features across time (mean)
        spatial_agg = spatial_feats_time.mean(dim=1)
        fused = torch.cat([spatial_agg, temporal_feat], dim=1)
        logits = self.classifier(fused)
        return logits

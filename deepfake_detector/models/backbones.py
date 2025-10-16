from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


class CNNFeatureExtractor(nn.Module):
    """Wrap a torchvision model to output a flattened feature vector.

    Optionally projects to a requested output dimension.
    """

    def __init__(self, feature_extractor: nn.Module, in_feature_dim: int, out_feature_dim: Optional[int] = None) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim or in_feature_dim
        if out_feature_dim is not None and out_feature_dim != in_feature_dim:
            self.proj = nn.Linear(in_feature_dim, out_feature_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: (B, C, H, W)
        feats = self.feature_extractor(x)  # (B, D, 1, 1)
        feats = torch.flatten(feats, start_dim=1)  # (B, D)
        return self.proj(feats)


def _resnext50_32x4d(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = tvm.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
    m = tvm.resnext50_32x4d(weights=weights)
    modules = list(m.children())[:-1]  # remove FC
    feat = nn.Sequential(*modules)
    return feat, 2048


def _efficientnet_b0(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = tvm.efficientnet_b0(weights=weights)
    # For efficientnet, features + avgpool
    feat = nn.Sequential(m.features, m.avgpool)
    return feat, 1280


def create_backbone(
    name: str = "resnext50_32x4d",
    pretrained: bool = True,
    out_feature_dim: Optional[int] = None,
) -> CNNFeatureExtractor:
    name = name.lower()
    if name in {"resnext", "resnext50", "resnext50_32x4d", "resnext-50"}:
        feat, in_dim = _resnext50_32x4d(pretrained)
    elif name in {"efficientnet", "efficientnet-b0", "efficientnet_b0", "b0"}:
        feat, in_dim = _efficientnet_b0(pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return CNNFeatureExtractor(feature_extractor=feat, in_feature_dim=in_dim, out_feature_dim=out_feature_dim)

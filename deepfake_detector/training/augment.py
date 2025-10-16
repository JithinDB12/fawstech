from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import random
from typing import Tuple


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets are soft labels (B x num_classes)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss


@dataclass
class MixupCutmixConfig:
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    prob: float = 0.5
    num_classes: int = 2


class MixupCutmix:
    def __init__(self, cfg: MixupCutmixConfig):
        self.cfg = cfg

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(labels, num_classes=self.cfg.num_classes).float()

    def _mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = torch.distributions.Beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha).sample().item()
        indices = torch.randperm(x.size(0), device=x.device)
        x_mixed = lam * x + (1 - lam) * x[indices]
        y_onehot = self._one_hot(y)
        y_mixed = lam * y_onehot + (1 - lam) * y_onehot[indices]
        return x_mixed, y_mixed

    def _cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: B x T x C x H x W
        b, t, c, h, w = x.shape
        lam = torch.distributions.Beta(self.cfg.cutmix_alpha, self.cfg.cutmix_alpha).sample().item()
        indices = torch.randperm(b, device=x.device)
        # Random bounding box
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)
        x_mixed = x.clone()
        x_mixed[:, :, :, y1:y2, x1:x2] = x[indices, :, :, y1:y2, x1:x2]
        lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
        y_onehot = self._one_hot(y)
        y_mixed = lam_adjusted * y_onehot + (1 - lam_adjusted) * y_onehot[indices]
        return x_mixed, y_mixed

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.cfg.prob:
            if random.random() < 0.5 and self.cfg.mixup_alpha > 0:
                return self._mixup(x, y)
            elif self.cfg.cutmix_alpha > 0:
                return self._cutmix(x, y)
        # default: no mix
        return x, self._one_hot(y)

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .metrics import compute_metrics


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    best_score: float = -float("inf")
    counter: int = 0
    mode: str = "max"  # "max" for AUC, "min" for loss

    def step(self, current: float) -> bool:
        improved = current > (self.best_score + self.min_delta) if self.mode == "max" else current < (self.best_score - self.min_delta)
        if improved:
            self.best_score = current
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_dir: str = "runs/exp",
        ckpt_dir: str = "artifacts",
        amp: bool = True,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir=log_dir)
        self.ckpt_dir = ckpt_dir
        self.amp = amp
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _mixup(self, x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if alpha <= 0.0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, float(lam)

    def _rand_bbox(self, size, lam):
        W = size[3]
        H = size[2]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def _cutmix(self, x: torch.Tensor, y: torch.Tensor, alpha: float):
        if alpha <= 0.0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, :, bby1:bby2, bbx1:bbx2] = x[index, :, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        return x, y, y[index], float(lam)

    def _criterion_with_targets(self, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        return lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)

    def train(self, num_epochs: int = 10) -> str:
        best_auc = -float("inf")
        best_ckpt_path = os.path.join(self.ckpt_dir, "best_model.pt")

        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch(epoch)
            val_loss, metrics = self._validate(epoch)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    self.scheduler.step()

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("AUC/val", metrics.get("roc_auc", float("nan")), epoch)

            auc = metrics.get("roc_auc", -float("inf"))
            if auc > best_auc:
                best_auc = auc
                torch.save({"model_state": self.model.state_dict()}, best_ckpt_path)

        self.writer.flush()
        return best_ckpt_path

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        num_samples = 0

        for step, (frames, labels) in enumerate(self.train_loader):
            # frames: (B, T, C, H, W)
            frames = frames.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Convert labels to class indices for CrossEntropy
            labels_ce = labels.long()

            # Optional mixup/cutmix; do not apply both simultaneously
            lam = 1.0
            y_a = labels_ce
            y_b = labels_ce
            if self.cutmix_alpha > 0.0:
                frames, y_a, y_b, lam = self._cutmix(frames, labels_ce, self.cutmix_alpha)
            elif self.mixup_alpha > 0.0:
                # Mixup across the entire sequence tensor
                b, t, c, h, w = frames.shape
                frames_2d = frames.view(b, -1, h, w)  # stack time into channels for mixup consistency
                frames_2d, y_a, y_b, lam = self._mixup(frames_2d, labels_ce, self.mixup_alpha)
                frames = frames_2d.view(b, t, c, h, w)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(frames)
                loss = self._criterion_with_targets(logits, y_a, y_b, lam)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = frames.size(0)
            running_loss += float(loss.item()) * batch_size
            num_samples += batch_size

        return running_loss / max(num_samples, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        num_samples = 0
        all_probs: list = []
        all_labels: list = []

        softmax = nn.Softmax(dim=1)

        for frames, labels in self.val_loader:
            frames = frames.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(frames)
                loss = self.criterion(logits, labels.long())
                probs = softmax(logits)  # (B, 2)

            batch_size = frames.size(0)
            running_loss += float(loss.item()) * batch_size
            num_samples += batch_size

            all_probs.append(probs[:, 1].detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics(y_true=y_true, y_prob=y_prob)
        return running_loss / max(num_samples, 1), metrics

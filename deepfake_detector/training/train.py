import os
import math
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from ..models.hybrid import HybridDeepfakeDetector
from .augment import MixupCutmix, MixupCutmixConfig, SoftTargetCrossEntropy


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, writer, criterion, mixup: MixupCutmix | None = None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for step, (frames, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # Apply MixUp/CutMix on-the-fly if configured
        targets_soft = None
        if mixup is not None:
            frames, targets_soft = mixup(frames, labels)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(frames)
            if targets_soft is not None:
                loss = criterion(logits, targets_soft)
            else:
                loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * frames.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step % 10 == 0:
            writer.add_scalar('train/step_loss', loss.item(), epoch * len(loader) + step)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/acc', epoch_acc, epoch)
    return epoch_loss, epoch_acc


def evaluate(model, loader, device, epoch, writer, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    running_loss = 0.0
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(frames)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            running_loss += loss.item() * frames.size(0)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    preds = (all_probs >= 0.5).astype('int64')
    acc = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', zero_division=0)
    try:
        roc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc = float('nan')
    val_loss = running_loss / len(all_labels)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/acc', acc, epoch)
    writer.add_scalar('val/precision', precision, epoch)
    writer.add_scalar('val/recall', recall, epoch)
    writer.add_scalar('val/f1', f1, epoch)
    if not math.isnan(roc):
        writer.add_scalar('val/roc_auc', roc, epoch)
    return {
        'loss': val_loss,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc,
    }


def fit(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'runs',
    backbone: str = 'resnext50_32x4d',
    lstm_hidden: int = 512,
    resume: str | None = None,
    use_mixup_cutmix: bool = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridDeepfakeDetector(backbone_name=backbone, pretrained=True, lstm_hidden=lstm_hidden).to(device)
    # Configure loss/augment
    mixup = MixupCutmix(MixupCutmixConfig()) if use_mixup_cutmix else None
    criterion = SoftTargetCrossEntropy() if mixup is not None else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    best_f1 = -1.0

    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        best_f1 = ckpt.get('best_f1', -1.0)

    os.makedirs(checkpoint_dir, exist_ok=True)

    patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, writer, criterion, mixup=mixup)
        metrics = evaluate(model, val_loader, device, epoch, writer, criterion)
        f1 = metrics['f1']

        # checkpoint
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_f1': max(best_f1, f1),
        }
        torch.save(state, os.path.join(checkpoint_dir, 'last.pt'))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(state, os.path.join(checkpoint_dir, 'best.pt'))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    writer.close()
    return os.path.join(checkpoint_dir, 'best.pt')

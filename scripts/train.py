import os
import json
import argparse
from torch.utils.data import DataLoader, random_split
import torch

from deepfake_detector.data.dataset import VideoFaceDataset
from deepfake_detector.training.train import fit
from deepfake_detector.training.augment import MixupCutmix, MixupCutmixConfig, SoftTargetCrossEntropy


def build_dataloaders(dataset_json: str, sequence_length: int, stride: int, batch_size: int, num_workers: int, val_ratio: float):
    with open(dataset_json, 'r') as f:
        items = json.load(f)
    dataset = VideoFaceDataset(items=items, sequence_length=sequence_length, stride=stride)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json', type=str, required=True)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--backbone', type=str, default='resnext50_32x4d')
    parser.add_argument('--lstm_hidden', type=int, default=512)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--mixup', action='store_true')
    args = parser.parse_args()

    train_loader, val_loader = build_dataloaders(
        args.dataset_json, args.sequence_length, args.stride, args.batch_size, args.num_workers, args.val_ratio
    )

    # Hand off to trainer; mixup handled internally via flags
    best_path = fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        backbone=args.backbone,
        lstm_hidden=args.lstm_hidden,
        resume=args.resume,
        use_mixup_cutmix=args.mixup,
    )
    print(f"Best checkpoint saved at: {best_path}")


if __name__ == '__main__':
    main()

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepfake_detector.data.dataset import FrameSequenceDataset
from deepfake_detector.models.backbones import create_backbone
from deepfake_detector.models.hybrid import HybridCNNLSTM
from deepfake_detector.training.trainer import Trainer


def build_dataloaders(train_manifest: str, val_manifest: str, image_size: int, sequence_len: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = FrameSequenceDataset(manifest_path=train_manifest, sequence_length=sequence_len, image_size=image_size, is_train=True)
    val_ds = FrameSequenceDataset(manifest_path=val_manifest, sequence_length=sequence_len, image_size=image_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--backbone", default="resnext50_32x4d")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-dir", default="runs/exp")
    parser.add_argument("--ckpt-dir", default="artifacts")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        image_size=args.image_size,
        sequence_len=args.sequence_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    backbone = create_backbone(name=args.backbone, pretrained=args.pretrained)
    cnn_feature_dim = backbone.out_feature_dim

    model = HybridCNNLSTM(
        cnn_backbone=backbone,
        cnn_feature_dim=cnn_feature_dim,
        temporal_hidden_dim=args.hidden_dim,
        temporal_layers=args.layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        num_classes=2,
        temporal_pool="last",
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=None,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        amp=not args.no_amp,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
    )

    best_ckpt = trainer.train(num_epochs=args.epochs)
    print(f"Best checkpoint saved to: {best_ckpt}")


if __name__ == "__main__":
    main()

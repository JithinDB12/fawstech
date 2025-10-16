import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepfake_detector.data.dataset import FrameSequenceDataset
from deepfake_detector.models.backbones import create_backbone
from deepfake_detector.models.hybrid import HybridCNNLSTM
from deepfake_detector.training.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="resnext50_32x4d")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FrameSequenceDataset(manifest_path=args.manifest, sequence_length=args.sequence_len, image_size=args.image_size, is_train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    backbone = create_backbone(name=args.backbone, pretrained=False)
    cnn_feature_dim = backbone.out_feature_dim

    model = HybridCNNLSTM(
        cnn_backbone=backbone,
        cnn_feature_dim=cnn_feature_dim,
        temporal_hidden_dim=args.hidden_dim,
        temporal_layers=args.layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        num_classes=2,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device)
    model.eval()

    y_true = []
    y_probs = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            logits = model(frames)
            probs = softmax(logits)[:, 1].detach().cpu().numpy()
            y_probs.append(probs)
            y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)
    y_probs = np.concatenate(y_probs)
    metrics = compute_metrics(y_true, y_probs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

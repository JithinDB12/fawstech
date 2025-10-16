import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from deepfake_detector.data.dataset import VideoFaceDataset
from deepfake_detector.inference.predict import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='resnext50_32x4d')
    parser.add_argument('--lstm_hidden', type=int, default=512)
    args = parser.parse_args()

    with open(args.dataset_json, 'r') as f:
        items = json.load(f)
    dataset = VideoFaceDataset(items=items, sequence_length=args.sequence_length, stride=args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model, device = load_model(args.weights, backbone=args.backbone, lstm_hidden=args.lstm_hidden)
    model.eval()

    all_labels = []
    all_probs = []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            logits = model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.append(labels)
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_pred = (y_prob >= 0.5).astype('int64')

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float('nan')

    print(json.dumps({
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc,
    }, indent=2))


if __name__ == '__main__':
    main()

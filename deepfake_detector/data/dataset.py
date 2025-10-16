import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transforms(image_size: int, is_train: bool) -> T.Compose:
    if is_train:
        return T.Compose(
            [
                T.Resize((image_size + 32, image_size + 32)),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                T.RandomErasing(p=0.25, value="random"),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


class FrameSequenceDataset(Dataset):
    """Load sequences of frames from a JSONL manifest.

    Each line: {"frames": ["/path/a.jpg", ...], "label": 0 or 1}
    """

    def __init__(
        self,
        manifest_path: str,
        sequence_length: int = 16,
        image_size: int = 224,
        is_train: bool = True,
        shuffle_within_video: bool = False,
    ) -> None:
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.is_train = is_train
        self.shuffle_within_video = shuffle_within_video

        self.samples: List[Dict[str, Any]] = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        self.transform = _build_transforms(image_size=image_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, num_available: int) -> List[int]:
        if num_available <= 0:
            return []
        if num_available <= self.sequence_length:
            indices = list(range(num_available))
            # Repeat last index to meet length
            indices += [indices[-1]] * (self.sequence_length - len(indices))
            return indices
        if self.is_train:
            # random subsequence
            start = random.randint(0, num_available - self.sequence_length)
            return list(range(start, start + self.sequence_length))
        else:
            # uniform sampling
            step = num_available / float(self.sequence_length)
            return [int(i * step) for i in range(self.sequence_length)]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        frame_paths: List[str] = item["frames"]
        label: int = int(item["label"])  # 0 real, 1 fake

        if self.shuffle_within_video and self.is_train:
            random.shuffle(frame_paths)

        indices = self._sample_indices(len(frame_paths))

        frames: List[torch.Tensor] = []
        for i in indices:
            path = frame_paths[min(i, len(frame_paths) - 1)]
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))

        # (T, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        return frames_tensor, label


def build_manifest_from_dir(
    root_dir: str,
    output_manifest_path: str,
    image_exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> int:
    """Build a JSONL manifest by scanning a directory tree.

    Expects structure like:
    root_dir/
      real/
        video_0001/
          frame_0001.jpg
          ...
      fake/
        video_0001/
          frame_0001.jpg
          ...

    Returns number of entries written.
    """
    categories = {
        "real": 0,
        "fake": 1,
    }

    entries: List[Dict[str, Any]] = []
    for split_label, label in categories.items():
        split_dir = os.path.join(root_dir, split_label)
        if not os.path.isdir(split_dir):
            continue
        for video_folder in sorted(os.listdir(split_dir)):
            video_dir = os.path.join(split_dir, video_folder)
            if not os.path.isdir(video_dir):
                continue
            frame_files = [
                os.path.join(video_dir, f)
                for f in sorted(os.listdir(video_dir))
                if os.path.splitext(f)[1].lower() in image_exts
            ]
            if len(frame_files) == 0:
                continue
            entries.append({"frames": frame_files, "label": label})

    os.makedirs(os.path.dirname(output_manifest_path) or ".", exist_ok=True)
    with open(output_manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return len(entries)

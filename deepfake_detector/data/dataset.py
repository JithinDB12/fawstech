import os
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, GaussNoise, CoarseDropout
from albumentations.pytorch import ToTensorV2
from ..utils.video import sample_frame_indices
from ..utils.face import FaceDetector


class VideoFaceDataset(Dataset):
    def __init__(
        self,
        items: List[Dict],
        sequence_length: int = 16,
        stride: int = 2,
        transform: Optional[Compose] = None,
        face_detector: Optional[FaceDetector] = None,
        cache_faces: bool = True,
    ):
        self.items = items
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform or self.default_transforms()
        self.face_detector = face_detector or FaceDetector()
        self.cache_faces = cache_faces
        self._cache: Dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample = self.items[idx]
        frame_paths = sample["frames"]  # list of frame file paths
        label = int(sample["label"])  # 0 real, 1 fake
        indices = sample_frame_indices(len(frame_paths), self.sequence_length, self.stride)
        # choose a random sequence window
        seq = random.choice(indices)
        images: List[Image.Image] = []
        for i in seq:
            fp = frame_paths[i]
            img = self._load_face(fp)
            if img is None:
                # fallback to center crop when no face
                img = Image.open(fp).convert("RGB")
            images.append(img)

        # apply transforms per frame
        frames_tensor = []
        for img in images:
            arr = np.array(img)
            augmented = self.transform(image=arr)["image"]
            frames_tensor.append(augmented)
        # shape: T x C x H x W
        frames = torch.stack(frames_tensor, dim=0)
        return frames, torch.tensor(label, dtype=torch.long)

    def _load_face(self, frame_path: str) -> Optional[Image.Image]:
        if self.cache_faces and frame_path in self._cache:
            return self._cache[frame_path]
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception:
            return None
        face = self.face_detector.detect_and_crop(img)
        if face is not None and self.cache_faces:
            self._cache[frame_path] = face
        return face

    @staticmethod
    def default_transforms(image_size: int = 224) -> Compose:
        return Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            GaussNoise(p=0.15),
            CoarseDropout(max_holes=8, max_height=image_size//12, max_width=image_size//12, p=0.3),
            ToTensorV2(),
        ])

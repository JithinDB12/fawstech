from typing import List, Tuple, Optional
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import numpy as np


class FaceDetector:
    def __init__(self, device: Optional[str] = None, image_size: int = 224, margin: int = 32):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.image_size = image_size
        self.margin = margin
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, device=device, keep_all=False, post_process=True)

    @torch.inference_mode()
    def detect_and_crop(self, image: Image.Image) -> Optional[Image.Image]:
        # Returns cropped aligned face or None if no face
        face = self.mtcnn(image)
        if face is None:
            return None
        # mtcnn returns tensor CxHxW normalized to [0,1]
        face = (face.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(face)

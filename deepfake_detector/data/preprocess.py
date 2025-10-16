import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN


class VideoFrameExtractor:
    """Extract frames from a video file with configurable sampling strategies.

    - Either specify target_fps to sample by time or num_frames for uniform sampling.
    - Returns frames as PIL RGB images.
    """

    def __init__(
        self,
        target_fps: Optional[float] = None,
        num_frames: Optional[int] = 16,
        image_size: int = 224,
        sampling: str = "uniform",  # "uniform" or "stride"
    ) -> None:
        if target_fps is None and num_frames is None:
            raise ValueError("Specify either target_fps or num_frames")
        self.target_fps = target_fps
        self.num_frames = num_frames
        self.image_size = image_size
        self.sampling = sampling

    def _uniform_indices(self, total_frames: int, num_frames: int) -> List[int]:
        if total_frames <= 0 or num_frames <= 0:
            return []
        if num_frames >= total_frames:
            return list(range(total_frames))
        step = total_frames / float(num_frames)
        return [int(i * step) for i in range(num_frames)]

    def extract(self, video_path: str) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        target_indices: List[int]
        if self.target_fps is not None:
            stride = max(int(round(original_fps / max(self.target_fps, 1e-6))), 1)
            target_indices = list(range(0, total_frames, stride))
        else:
            target_indices = self._uniform_indices(total_frames, int(self.num_frames or 1))

        frames: List[Image.Image] = []
        index_set = set(target_indices)
        current_index = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if current_index in index_set:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb).resize((self.image_size, self.image_size), Image.BILINEAR)
                frames.append(pil_img)
            current_index += 1

        cap.release()

        # If video is shorter, repeat last frame to reach desired length
        if self.num_frames is not None and len(frames) > 0 and len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))

        return frames


class FaceCropper:
    """Detect and crop faces using MTCNN. Returns square aligned crops.

    Falls back to center-crop resize when face detection fails.
    """

    def __init__(
        self,
        image_size: int = 224,
        margin: int = 20,
        device: Optional[str] = None,
        keep_all: bool = False,
    ) -> None:
        self.image_size = image_size
        self.keep_all = keep_all
        device_resolved = device
        if device_resolved is None:
            device_resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            post_process=True,
            device=device_resolved,
            keep_all=keep_all,
            selection_method="probability",
        )

    def _to_pil(self, tensor_img: torch.Tensor) -> Image.Image:
        # facenet-pytorch returns torch tensor CxHxW in [0,1]
        array = (tensor_img.clamp(0, 1).mul(255).byte().cpu().numpy())
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError("Unexpected tensor shape from MTCNN")
        array = np.transpose(array, (1, 2, 0))
        return Image.fromarray(array)

    def crop(self, image: Image.Image) -> Image.Image:
        result = self.mtcnn(image)
        if result is None:
            # Fallback: resize original
            return image.resize((self.image_size, self.image_size), Image.BILINEAR)
        if isinstance(result, torch.Tensor):
            return self._to_pil(result)
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], torch.Tensor):
            # keep_all=True returns list; select the first (highest prob)
            return self._to_pil(result[0])
        # Unknown return type, fallback
        return image.resize((self.image_size, self.image_size), Image.BILINEAR)

    def crop_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        if self.keep_all:
            # For keep_all=True, per-image handling is safer
            return [self.crop(img) for img in images]
        results = self.mtcnn(images)
        crops: List[Image.Image] = []
        if results is None:
            return [img.resize((self.image_size, self.image_size), Image.BILINEAR) for img in images]
        for idx, res in enumerate(results):
            if res is None:
                crops.append(images[idx].resize((self.image_size, self.image_size), Image.BILINEAR))
            else:
                crops.append(self._to_pil(res))
        return crops


def process_image(
    image_path: str,
    face_cropper: FaceCropper,
    output_path: Optional[str] = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    crop = face_cropper.crop(image)
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        crop.save(output_path)
    return crop


def process_video(
    video_path: str,
    frame_extractor: VideoFrameExtractor,
    face_cropper: FaceCropper,
    output_dir: Optional[str] = None,
) -> List[Image.Image]:
    frames = frame_extractor.extract(video_path)
    crops = face_cropper.crop_batch(frames)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        written: List[Image.Image] = []
        for i, img in enumerate(crops):
            out_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            img.save(out_path)
            written.append(img)
        return written

    return crops

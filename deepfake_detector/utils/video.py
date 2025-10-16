import os
import math
import cv2
import ffmpeg
from typing import List, Tuple


def extract_frames_ffmpeg(video_path: str, output_dir: str, fps: float = 5.0, overwrite: bool = False) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    # Use ffmpeg to extract frames at a fixed fps
    output_pattern = os.path.join(output_dir, "%06d.jpg")
    if not overwrite and any(f.endswith(".jpg") for f in os.listdir(output_dir)):
        return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")])
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=fps)
        .output(output_pattern, start_number=0, qscale=2, vsync='vfr')
        .overwrite_output()
        .run(quiet=True)
    )
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")])


def sample_frame_indices(num_frames: int, sequence_length: int, stride: int = 1) -> List[List[int]]:
    if num_frames <= 0 or sequence_length <= 0:
        return []
    indices = []
    max_start = max(0, num_frames - sequence_length * stride)
    for start in range(0, max_start + 1, sequence_length * stride):
        seq = [min(start + i * stride, num_frames - 1) for i in range(sequence_length)]
        indices.append(seq)
    if not indices:
        # Pad last frame if too short
        seq = [min(i, num_frames - 1) for i in range(sequence_length)]
        indices.append(seq)
    return indices

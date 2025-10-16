import os
from typing import List, Tuple
import torch
from PIL import Image
import numpy as np
from ..models.hybrid import HybridDeepfakeDetector
from ..utils.face import FaceDetector
from ..utils.video import extract_frames_ffmpeg, sample_frame_indices


def load_model(weights_path: str, backbone: str = 'resnext50_32x4d', lstm_hidden: int = 512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridDeepfakeDetector(backbone_name=backbone, pretrained=False, lstm_hidden=lstm_hidden)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def predict_image(model, device, image: Image.Image, face_first: bool = True) -> Tuple[float, int]:
    if face_first:
        face = FaceDetector(device=str(device)).detect_and_crop(image)
        if face is not None:
            image = face
    image = image.resize((224, 224))
    x = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    x = x.unsqueeze(0)  # 1 x C x H x W
    x = x.to(device)

    # replicate to a fake temporal dimension of length 8 for image
    x_seq = x.unsqueeze(1).repeat(1, 8, 1, 1, 1)
    logits = model(x_seq)
    prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    pred = int(prob_fake >= 0.5)
    return prob_fake, pred


@torch.no_grad()
def predict_video(model, device, video_path: str, tmp_dir: str, sequence_length: int = 16, stride: int = 2) -> Tuple[float, int]:
    os.makedirs(tmp_dir, exist_ok=True)
    frames_dir = os.path.join(tmp_dir, os.path.splitext(os.path.basename(video_path))[0])
    frame_paths = extract_frames_ffmpeg(video_path, frames_dir, fps=5.0, overwrite=False)
    if len(frame_paths) == 0:
        raise RuntimeError('No frames extracted from video')

    indices_list = sample_frame_indices(len(frame_paths), sequence_length, stride)
    # choose middle sequence if multiple
    seq = indices_list[len(indices_list) // 2]

    face_detector = FaceDetector(device=str(device))
    frames_tensor: List[torch.Tensor] = []
    for i in seq:
        img = Image.open(frame_paths[i]).convert('RGB')
        face = face_detector.detect_and_crop(img)
        if face is None:
            face = img
        face = face.resize((224, 224))
        x = torch.from_numpy(np.array(face)).float().permute(2, 0, 1) / 255.0
        frames_tensor.append(x)

    x_seq = torch.stack(frames_tensor, dim=0).unsqueeze(0).to(device)  # 1 x T x C x H x W
    logits = model(x_seq)
    prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    pred = int(prob_fake >= 0.5)
    return prob_fake, pred
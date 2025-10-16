from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def _find_last_conv_layer(module: nn.Module) -> nn.Module:
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM target")
    return last_conv


class GradCAMWrapper:
    """Helper for producing Grad-CAM heatmaps for the CNN component.

    Assumes the model exposes either a .cnn with conv layers or is itself a CNN.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.eval()

        # Try to get CNN submodule if available
        cnn_module = getattr(model, "cnn", model)
        self.target_layer = _find_last_conv_layer(cnn_module)
        self.cam = GradCAM(model=cnn_module, target_layers=[self.target_layer], use_cuda=self.device.type == "cuda")

    @torch.no_grad()
    def generate(self, image_tensor: torch.Tensor, rgb_image: np.ndarray) -> np.ndarray:
        """Generate heatmap for a single image.

        image_tensor: (1, 3, H, W) on device
        rgb_image: (H, W, 3) float32 in [0,1]
        """
        grayscale_cam = self.cam(input_tensor=image_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        return visualization

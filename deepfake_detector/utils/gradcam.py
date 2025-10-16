from typing import Optional, Tuple
import torch
import numpy as np
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:  # library may be missing
    GradCAM = None


def _get_target_layer(model) -> Optional[torch.nn.Module]:
    # Try to locate the last conv block of the ResNeXt backbone
    try:
        feature_net = model.backbone.backbone  # FeatureListNet
        base = getattr(feature_net, 'model', None)
        if base is None:
            base = feature_net
        # Prefer layer4 if exists
        layer4 = getattr(base, 'layer4', None)
        if layer4 is not None:
            return layer4[-1] if hasattr(layer4, '__getitem__') else layer4
        # fallback: last child module
        children = [m for m in base.modules() if isinstance(m, torch.nn.Conv2d)]
        return children[-1] if children else None
    except Exception:
        return None


@torch.no_grad()
def gradcam_on_image(model, device, image: Image.Image, alpha: float = 0.5) -> Optional[Image.Image]:
    model.eval()
    # Preprocess
    rgb = image.resize((224, 224)).convert('RGB')
    img_np = np.array(rgb).astype(np.float32) / 255.0
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Create a pseudo sequence of length 8
    x_seq = x.unsqueeze(1).repeat(1, 8, 1, 1, 1)

    if GradCAM is None:
        return None
    target_layer = _get_target_layer(model)
    if target_layer is None:
        return None

    cam = GradCAM(model=model.backbone.backbone.model, target_layers=[target_layer], use_cuda=(device.type=='cuda'))
    # Run backbone on a single frame for CAM
    frame = x  # 1 x C x H x W
    outputs = cam(input_tensor=frame, targets=None)
    heatmap = outputs[0]
    vis = show_cam_on_image(img_np, heatmap, use_rgb=True)
    return Image.fromarray(vis)

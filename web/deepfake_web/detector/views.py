import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import torch

from deepfake_detector.inference.predict import load_model, predict_image
from deepfake_detector.utils.gradcam import gradcam_on_image

# Lazy global model
_MODEL = None
_DEVICE = None


def _get_model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        weights = os.environ.get('DEEPFAKE_WEIGHTS', str(settings.BASE_DIR.parent / 'checkpoints' / 'best.pt'))
        try:
            _MODEL, _DEVICE = load_model(weights)
        except Exception:
            _MODEL, _DEVICE = None, torch.device('cpu')
    return _MODEL, _DEVICE


def index(request):
    return render(request, 'detector/index.html')


@csrf_exempt
def predict_view(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        f = request.FILES['file']
        path = default_storage.save(os.path.join('uploads', f.name), f)
        full_path = os.path.join(settings.MEDIA_ROOT, path)
        model, device = _get_model()
        if model is None:
            context['error'] = 'Model weights not found. Please train the model.'
        else:
            try:
                img = Image.open(full_path).convert('RGB')
                prob_fake, pred = predict_image(model, device, img)
                context['prob_fake'] = prob_fake
                context['prediction'] = 'Fake' if pred == 1 else 'Real'
                # Try Grad-CAM visualization (optional)
                heatmap = gradcam_on_image(model, device, img)
                if heatmap is not None:
                    # save next to upload
                    heatmap_rel = os.path.join('uploads', f"heatmap_{os.path.basename(full_path)}.jpg")
                    heatmap_path = os.path.join(settings.MEDIA_ROOT, heatmap_rel)
                    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                    heatmap.save(heatmap_path)
                    context['heatmap_url'] = settings.MEDIA_URL + heatmap_rel
            except Exception as e:
                context['error'] = str(e)
    return render(request, 'detector/index.html', context)

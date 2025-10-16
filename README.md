## Deepfake Detector (Hybrid ResNeXt + LSTM)

End-to-end deepfake detection using spatial (CNN) and temporal (LSTM) features.

### Features
- ResNeXt backbone for spatial features via timm
- LSTM for temporal dynamics
- Data utilities: video frame extraction (ffmpeg), face detection (MTCNN)
- Training with AMP, early stopping, checkpoints, TensorBoard
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Django web app for uploads and predictions
- Grad-CAM hooks prepared via pytorch-grad-cam (TODO in later step)

### Structure
- `deepfake_detector/`: core library
  - `data/`: dataset and transforms
  - `models/`: model architectures
  - `training/`: training loop and evaluation
  - `utils/`: helpers for video and face
  - `inference/`: prediction utilities
- `scripts/`: scripts for data prep, training, evaluation
- `web/`: Django app `deepfake_web`

### Quickstart
1. Install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Extract frames from videos:
```bash
python scripts/prepare_data.py --videos_dir data/raw/videos --out_dir data/processed/frames --fps 5
```

3. Prepare dataset JSON (example schema):
```json
[
  {"frames": ["data/processed/frames/video1/000001.jpg", "..."], "label": 0},
  {"frames": ["data/processed/frames/video2/000001.jpg", "..."], "label": 1}
]
```

4. Train: implement your DataLoader to feed `VideoFaceDataset` and call `fit` in `deepfake_detector/training/train.py`.

5. Run web app:
```bash
cd web/deepfake_web
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Set env var `DEEPFAKE_WEIGHTS` to your checkpoint path for inference.

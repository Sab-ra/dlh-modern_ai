#!/usr/bin/env python3
"""
Tune YOLOv8 hyps, then
train a final detector
"""
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML


DATA = 'datasets/detection/data.yaml'
TUNE_DIR = Path('runs/detect/tune')
HYP_PATH = TUNE_DIR / 'best_hyperparameters.yaml'
BEST_TUNE = TUNE_DIR / 'weights' / 'best.pt'
FINAL_NAME = 'best_model.pt'


def tune_hyperparameters():
    """
    Phase 1: short genetic search with
    'model.tune()'
    Phase 2: continue from 'best.pt'
    toward ~150 epochs
    """
    model = YOLO('yolov8n.pt')
    space = {
        'lr0': (1e-5, 1e-2),
        'lrf': (0.01, 1.0),
        'momentum': (0.7, 0.98),
        'weight_decay': (0.0, 0.001),
        'mosaic': (0.0, 1.0),
        'mixup': (0.0, 0.3),
        'hsv_h': (0.0, 0.1),
        'hsv_s': (0.0, 0.9),
        'hsv_v': (0.0, 0.9),
        'degrees': (0.0, 30.0),
        'translate': (0.0, 0.3),
        'scale': (0.0, 0.7),
        'box': (5.0, 10.0),
        'cls': (0.2, 1.5),
        'dfl': (0.8, 2.0)
    }
    model.tune(
        data=DATA,
        space=space,
        iterations=15,
        epochs=10,
        imgsz=640,
        batch=16,
        optimizer='AdamW',
        plots=False,
        save=True,
        val=True,
        name='tune'
    )
    hyp = YAML.load(HYP_PATH)
    model = YOLO(str(BEST_TUNE))
    results = model.train(
        data=DATA,
        epochs=140,
        imgsz=640,
        batch=16,
        patience=20,
        plots=True,
        save=True,
        **hyp
    )
    src = Path(results.save_dir) / 'weights' / 'best.pt'
    shutil.copy2(src, FINAL_NAME)


if __name__ == '__main__':
    tune_hyperparameters()

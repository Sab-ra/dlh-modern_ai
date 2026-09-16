#!/usr/bin/env python3
"""
Train YOLOv8 with optional custom Albumentations
"""
from ultralytics import YOLO


def train_with_augmentation(
        data,
        model_path='yolov8n.pt',
        epochs=50,
        imgsz=640,
        batch=16,
        augmentation=False,
        yolo_aug_params=None,
        albumentations_transforms=None,
        save=False,
        plots=False,
        verbose=False
):
    """
    Train a YOLO model with native and
    custom augmentation
    """
    model = YOLO(model_path)
    train_kw = {
        'data': data,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'save': save,
        'plots': plots,
        'verbose': verbose
    }
    if yolo_aug_params:
        train_kw.update(yolo_aug_params)
    if albumentations_transforms:
        train_kw['augmentations'] = (
            albumentations_transforms
        )
    results = model.train(**train_kw)

    return model, results

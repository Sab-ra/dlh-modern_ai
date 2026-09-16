#!/usr/bin/env python3
"""
YOLO-compatible basic augumentation with Albumentations
"""
import albumentations as A
import numpy as np


def basic_aug(image, bboxes, labels):
    """
    Apply flip, brightness/contrast, and
    affine to an image and 
    its Pascal VOC boxes
    """
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(
                translate_percent=0.1,
                scale=0.1,
                rotate=(-30, 0),
                p=0.5
            )
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ),
        seed=42
    )
    out = transform(
        image=image,
        bboxes=bboxes,
        labels=labels
    )
    boxes = np.asarray(out['bboxes'])
    labs = [int(x) for x in out['labels']]

    return out['image'], boxes, labs
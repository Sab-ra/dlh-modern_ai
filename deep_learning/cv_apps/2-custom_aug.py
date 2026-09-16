#!/usr/bin/env python3
"""
YOLO-compatible custom augmentation
with Albumentations
"""
import albumentations as A
import numpy as np


def custom_aug(image, bboxes, labels):
    """
    Apply motion blur and one geomentric warp
    to an image and its Pascal VOC boxes
    """
    transform = A.Compose(
        [
            A.MotionBlur(blur_limit=5, p=0.9),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        p=0.2
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.05,
                        p=0.2
                    )
                ],
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    label_fields=['labels']
                ),
                seed=42
            )
        ]
    )
    output = transform(
        image = image,
        bboxes=bboxes,
        labels=labels
    )
    boxes = np.asarray(output['bboxes'])
    labs = list(output['labels'])

    return output['image'], boxes, labs

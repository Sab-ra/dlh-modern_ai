#!/usr/bin/env python3
"""
Grid-search YOLO conf and NMS IoU on validation
"""
import pandas as pd
from ultralytics import YOLO


DATA = 'datasets/detection/data.yaml'


def _as_yolo(model):
    """
    Accept a weight path or a YOLO object
    """
    if isinstance(model, YOLO):
        return model
    return YOLO(model)


def _f1(precision, recall):
    """
    Harmonic mean of precision and recall
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (
        precision + recall
    )


def tune_inference(
        model,
        val_images_path,
        conf_thresholds=[
            0.25, 0.3, 0.35, 0.4, 0.45, 0.5
        ],
        iou_thresholds=[
            0.4, 0.45, 0.5, 0.55, 0.6, 0.65
        ],
        imgsz=640
):
    """
    Try every conf/IoU pair with model.val()
    """
    yolo = _as_yolo(model)
    data = val_images_path
    if not str(data).endswith(('.yaml', '.yml')):
        data = DATA
    rows = []
    best = None
    for conf in conf_thresholds:
        for iou in iou_thresholds:
            metrics = yolo.val(
                data=data,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                plots=False,
                verbose=False,
                save=False
            )
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            map50 = float(metrics.box.map50)
            map5095 = float(metrics.box.map)
            f1 = _f1(precision, recall)
            row = {
                'conf': conf,
                'iou': iou,
                'mAP50': map50,
                'mAP5095': map5095,
                'precision': precision,
                'recall': recall,
                'F1': f1
            }
            rows.append(row)
            score = (f1, map50, map5095)
            if best is None or score > best[0]:
                best = (score, row)
    all_results = pd.DataFrame(rows)
    winner = best[1]
    return {
        'best_conf': winner['conf'],
        'best_iou': winner['iou'],
        'best_metrics': {
            'mAP50': winner['mAP50'],
            'mAP50-95': winner['mAP5095'],
            'precision': winner['precision'],
            'recall': winner['recall'],
            'F1': winner['F1']
        },
        'all_results': all_results
    }

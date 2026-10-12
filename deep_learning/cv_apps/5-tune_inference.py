#!/usr/bin/env python3
"""
Grid-search YOLO conf and NMS IoU on validation
"""
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


def inference_tuning(
        data_yaml,
        model,
        conf_list=None,
        iou_list=None,
        imgsz=640
):
    """
    Try every conf/IoU pair with model.val()
    """
    if conf_list is None:
        conf_list = [
            0.25, 0.3, 0.35, 0.4, 0.45, 0.5
        ]
    if iou_list is None:
        iou_list = [
            0.4, 0.45, 0.5, 0.55, 0.6, 0.65
        ]
    yolo = _as_yolo(model)
    results = []
    for conf in conf_list:
        for iou in iou_list:
            metrics = yolo.val(
                data=data_yaml,
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
            results.append({
                'conf': conf,
                'iou': iou,
                'mAP50': map50,
                'mAP5095': map5095,
                'precision': precision,
                'recall': recall,
                'F1': _f1(precision, recall)
            })

    return results

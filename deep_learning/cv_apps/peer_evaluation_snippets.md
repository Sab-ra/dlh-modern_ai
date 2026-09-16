# Peer Evaluation

## 0-prep_data.py

```python3
#!/usr/bin/env python3
"""
Convert Pascal VOC 2012 into YOLOv8 layout.
Keep person, car, and bicycle only.
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASSES = ['person', 'car', 'bicycle']
CLASS_TO_ID = {
    name: idx for idx, name in enumerate(CLASSES)
}

YAML_TEXT = (
    'path: datasets/detection/\n'
    'train: images/train\n'
    'val: images/val\n'
    '\n'
    'nc: 3\n'
    'names: ["person", "car", "bicycle"]\n'
)


def read_ids(list_path):
    """
    Read image IDs from a sample list file.
    """
    ids = []
    with open(list_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            stem = line.strip()
            if stem:
                ids.append(Path(stem).stem)
    return ids


def voc_box_to_yolo(xmin, ymin, xmax, ymax, width, height):
    """
    Convert a VOC pixel box to YOLO xywh in 0-1
    """
    xmin = max(0.0, min(float(xmin), width))
    xmax = max(0.0, min(float(xmax), width))
    ymin = max(0.0, min(float(ymin), height))
    ymax = max(0.0, min(float(ymax), height))
    xc = ((xmin + xmax) / 2.0) / width
    yc = ((ymin + ymax) / 2.0) / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height

    return xc, yc, bw, bh


def parse_voc_xml(xml_path):
    """
    Parse VOC XML and return YOLO lines for kept classes
    """
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        if name not in CLASS_TO_ID:
            continue
        box = obj.find('bndbox')
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
        xc, yc, bw, bh = voc_box_to_yolo(
            xmin, ymin, xmax, ymax, width, height
        )
        if bw <= 0 or bh <= 0:
            continue
        cid = CLASS_TO_ID[name]
        line = (
            f'{cid} {xc:.6f} {yc:.6f} '
            f'{bw:.6f} {bh:.6f}'
        )
        lines.append(line)

    return lines


def export_split(ids, voc_root, img_out, lbl_out):
    """
    Copy images and write YOLO labels for one split
    """
    jpeg_dir = voc_root / 'JPEGImages'
    ann_dir = voc_root / 'Annotations'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for stem in ids:
        src_img = jpeg_dir / f'{stem}.jpg'
        src_xml = ann_dir / f'{stem}.xml'
        if not src_img.is_file():
            raise FileNotFoundError(src_img)
        if not src_xml.is_file():
            raise FileNotFoundError(src_xml)
        shutil.copy2(src_img, img_out / f'{stem}.jpg')
        lines = parse_voc_xml(src_xml)
        lbl_path = lbl_out / f'{stem}.txt'
        with open(lbl_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
            if lines:
                handle.write('\n')


def write_data_yaml(yaml_path):
    """
    Write the required YOLOv8 data.yaml file
    """
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as handle:
        handle.write(YAML_TEXT)


def prep_data(
        voc_root,
        out_root=None,
        train_list='train_samples.txt',
        val_list='val_samples.txt'
):
    """
    Build datasets/detection from VOC 2012 and sample lists
    """
    here = Path(__file__).resolve().parent
    voc_root = Path(voc_root)
    if out_root is None:
        out_root = here / 'datasets' / 'detection'
    else:
        out_root = Path(out_root)
    train_ids = read_ids(here / train_list)
    val_ids = read_ids(here / val_list)
    export_split(
        train_ids,
        voc_root,
        out_root / 'images' / 'train',
        out_root / 'labels' / 'train'
    )
    export_split(
        val_ids,
        voc_root,
        out_root / 'images' / 'val',
        out_root / 'labels' / 'val'
    )
    write_data_yaml(out_root / 'data.yaml')


if __name__ == '__main__':
    voc = Path('VOCdevkit') / 'VOC2012'
    prep_data(voc)

```

## 4-tune_train.py

```python3
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

```

## 5-tune_inference.py

```python3
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

```

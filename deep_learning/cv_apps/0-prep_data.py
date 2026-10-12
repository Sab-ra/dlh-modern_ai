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

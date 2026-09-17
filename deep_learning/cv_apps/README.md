# Computer Vision Apps

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

- What is object detection?
- What is a single-shot detector?
- What is the YOLO algorithm?
- What is IoU and how do you calculate it?
- What is non-max suppression?
- What are anchor boxes?
- What is mAP and how do you calculate it?
- What is the difference between object detection and image segmentation?
- What are semantic and instance segmentation?
- What is a segmentation mask?
- How are objects represented in instance segmentation?
- What is polygon-based annotation?
- What is mask IoU and how is it calculated?
- Bounding box mAP vs Segmentation mAP?
- How is non-max suppression applied when masks overlap?
- When to use object detection vs segmentation?

## 0-prep_data.py

### Download & Organize

Download the Pascal VOC 2012 dataset and organize it into the YOLOv8 format, keeping only the classes `["person", "car", "bicycle"]`.

Organize the dataset in the project forlder as follows:

```text
cv_apps/
└── datasets/
    └── detection/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml
```

- Only Keep the images with the names given in the files `train_samples.txt` and `val_samples.txt`
- Every image in `images/train/` must have a corresponding `.txt` label file in `labels/train/` (same for `val/`).
- Keep only the classes `person`, `car` and `bicycle`
- Create `data.yaml` with the following content:

```yaml
path: datasets/detection/
train: images/train
val: images/val

nc: 3
names: ["person", "car", "bicycle"]
```

## 1-basic_aug.py

### Basic Transformations

Write a function `def basic_aug(image, bboxes, labels):` that applies YOLO-compatible data augmentation using Albumentations.

Arguments:
- `image` (`np.ndarray`): Input image
- `bboxes` (`List[List[int]]`): Bounding boxes in Pascal VOC format
- `labels` (`List[int]`): Class labels corresponding to each bounding box

The function must apply the following transformations:
- random horizontal flipping (`p = 0.5`)
- brightness/contrast augmentation (`p = 0.2`)
- Affine (`translate_percent: 0.1`, `scale 0.1`, `rotate [-30, 0]` with `p = 0.5`)

Returns the augmented image `np.ndarray`, augmented bounding boxes `np.ndarray` and labels `List[int]`

Note: For reproducibility, set `seed=42` (`A.Compose([...], seed=42)`)

## 2-custom_aug.py

### Albumentations custom Transformations

Write a function `def custom_aug(image, bboxes, labels):` that applies YOLO-compatible Albumentations-exclusive data augmentation.

Arguments:
- `image` (`np.ndarray`): Input image
- `bboxes` (`List[List[int]]`): Bounding boxes in Pascal VOC format
- `labels` (`List[int]`): Class labels corresponding to each bounding box

The function must apply the following transformations:
- motion blur (`blur_limit=5, p=0.9`)

With `p= 0.9`, One of these will be applied:
- elastic (`alpha=1, sigma=50, p=0.2`)
- or optical distortions (`distort_limit=0.05, p=0.2`)

Returns the augmented image `np.ndarray`, augmented bounding boxes `np.ndarray` and labels `List[int]`

Note: For reproducibility, set `seed=42` (`A.Compose([…], seed=42)`)

## 3-train_aug.py

### Train & Augment

write a function `def train_with_augmentation(data_yaml, model="yolov8n.pt", aug=None, custom_albu=None, epochs=50, imgsz=640, batch=16):` that trains a YOLO model using data augmentation, and allows custom Albumentations transforms to be applied during training.

Arguments:
- `data` (`str`): Path to the dataset YAML file containing train/val/test paths and class names for Ultralytics YOLO.
- `model_path` (`str`, optional): Path to pre-trained weights file (e.g., `yolov8n.pt`) or model configuration file.
- `epochs` (`int`, optional): Number of training epochs.
- `imgsz` (`int or tuple`, optional): Input image size for training. Can be a single integer (square) or tuple (height, width).
- `batch` (`int`, optional): Batch size for training.
- `augmentation` (`bool`, optional): Global flag to enable/disable augmentation.
- `yolo_aug_params` (`dict`, optional): Dictionary of YOLO's native augmentation parameters to customize built-in transforms. When provided, these parameters override the default augmentation settings.
- `albumentations_transforms` (`list`, optional): List of Albumentations transform objects (e.g., `[A.Blur(...), A.HorizontalFlip(...)]`) for custom augmentation pipeline. When provided, this takes precedence over YOLO's built-in augmentations.
- `save` (`bool`, optional): Whether to save training checkpoints and final model.
- `plots` (`bool`, optional): Whether to generate and save training plots (loss curves, metrics, etc.).
- `verbose` (`bool`, optional): Whether to display detailed training progress and output. Retuns the trained yolo model and the full training output.

## 4-tune_train.py

### Hyperparameter Tuning

Write a function `def tune_hyperparameters():` that performs hyperparameter tuning to discover optimal training hyperparameters, then trains and saves a final object detection model.

Your pipeline should automatically find the best hyperparameters through lightweight exploration, then continue training from the best checkpoint.

```python
def tune_hyperparameters():
    """
    Performs two-phase hyperparameter tuning and training:
    Phase 1: Lightweight hyperparameter search
    Phase 2: Continue training from best checkpoint

    Returns:
        None
    """
```

Perform lightweight tuning to discover optimal hyperparameters:

- Use YOLOv8's built-in `model.tune()` method
- Test multiple configurations (e.g., 15-20 iterations)
- Keep trials short (e.g., 10 epochs per trial) for time efficiency
- Track validation metrics to identify the best configuration

Key hyperparameters to explore:

- Learning rates (`lr0`, `lrf`)
- Augmentation parameters (`mosaic`, `mixup`, `hsv_h`, `hsv_s`, `hsv_v`)
- Geometric transforms (`degrees`, `translate`, `scale`)
- Loss weights (`box`, `cls`, `dfl`)
- Optimizer settings (`momentum`, `weight_decay`)
- etc.

Save the best hyperparameters from tuning

YOLO automatically saves to: `runs/detect/tune/best_hyperparameters.yaml`

The best model checkpoint is saved to: `runs/detect/tune/weights/best.pt`

Continue training to reach full convergence:

- Load the best model from `runs/detect/tune/weights/best.pt`
- Train for additional epochs to reach a total of ~150 epochs
- Use the optimal hyperparameters discovered in Phase 1
- Apply early stopping to prevent overfitting
- Save the final trained model to your github as `best_model.pt`

Your final model `best_model.pt` should achieve: `mAP50 ≥ 65%` and `mAP50-95 ≥ 46%`

Tips:

- Use short trials for tuning
- Continue from `best.pt` to save initial training epochs
- You can use GoogleColab or Kaggle for better GPU performances
- Use Early stopping during final training to avoid overfitting
- Check training plots in `runs/detect/` directories
- Use `4-main.py` to validate your model meets performance requirements

## 5-tune_inference.py

### Inference Tuning

Write a function `def tune_inference(model, val_images_path, conf_thresholds=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5], iou_thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65], imgsz=640):` that performs inference parameter tuning to find optimal confidence and IoU thresholds for best performance on the validation set.

Arguments:
- `model` (`str or YOLO`): Path to trained model weights or YOLO model object
- `val_images_path` (`str`): Path to validation images directory
- `conf_thresholds` (`List[float]`): List of confidence thresholds to test (default: `[0.25, 0.3, 0.35, 0.4, 0.45, 0.5]`)
- `iou_thresholds` (`List[float]`): List of IoU thresholds for NMS to test (default: `[0.4, 0.45, 0.5, 0.55, 0.6, 0.65]`)
- `imgsz` (`int`): Image size for inference (default: `640`)

The function must:
- Perform grid search over all combinations of `conf_thresholds` and `iou_thresholds`
- For each combination, run validation using `model.val()` and collect metrics:
- `mAP50` (mean Average Precision at `IoU=0.50`)
- `mAP50-95` (mean Average Precision at `IoU=0.50:0.95`)
- `Precision`
- `Recall`
- `F1-score` Returns a dictionary containing:
- `best_conf`: Optimal confidence threshold
- `best_iou`: Optimal IoU threshold
- `best_metrics`: Dictionary of metrics at optimal settings (`mAP50`, `mAP50-95`, `precision`, `recall`, `F1`)
- `all_results`: DataFrame or list of dictionaries with all tested combinations and their metrics

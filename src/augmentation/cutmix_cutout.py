import numpy as np
import torch
import albumentations as A
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        logger.warning(f"Could not load {config_path}. Using empty dict.")
        return {}

class CutMixAugmentation:
    def __init__(self, alpha: float = 1.0, probability: float = 0.3):
        self.alpha = alpha
        self.probability = probability

    def __call__(self, images_batch: torch.Tensor, labels_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if np.random.rand() > self.probability:
            return images_batch, labels_batch, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, h, w = images_batch.size()
        index = torch.randperm(batch_size)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        cut_w = int(w * np.sqrt(1. - lam))
        cut_h = int(h * np.sqrt(1. - lam))
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        images_batch[:, :, y1:y2, x1:x2] = images_batch[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        if labels_batch.dim() == 1:
            labels_batch = torch.nn.functional.one_hot(labels_batch, num_classes=5).float()
            
        mixed_labels = lam * labels_batch + (1 - lam) * labels_batch[index]
        return images_batch, mixed_labels, lam

class CutoutAugmentation:
    def __init__(self, num_holes: int = 1, max_h_size: int = 64, max_w_size: int = 64, probability: float = 0.3):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.probability = probability

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.probability:
            return image
        
        h, w = image.shape[:2]
        mask = np.ones((h, w), np.float32)
        for _ in range(self.num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.max_h_size // 2, 0, h)
            y2 = np.clip(y + self.max_h_size // 2, 0, h)
            x1 = np.clip(x - self.max_w_size // 2, 0, w)
            x2 = np.clip(x + self.max_w_size // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
            
        if image.ndim == 3:
            mask = np.expand_dims(mask, -1)
        return image * mask

class FundusAugmentationPipeline:
    def __init__(self, config_name: str = "pipeline_config.yaml"):
        self.config = load_config(config_name)
        aug_cfg = self.config.get("augmentation", {})
        self.img_size = aug_cfg.get("image_size", 224)
        
    def get_train_transforms(self) -> A.Compose:
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.ElasticTransform(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def get_val_transforms(self) -> A.Compose:
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
    def get_test_transforms(self) -> A.Compose:
        return self.get_val_transforms()

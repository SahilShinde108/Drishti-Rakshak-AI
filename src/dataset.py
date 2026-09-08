import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import random
import logging

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'optic_disc_area', 'optic_cup_area', 'cup_to_disc_ratio',
    'exudates_count', 'hemorrhages_count', 'microaneurysms_count',
    'vessel_tortuosity', 'bifurcation_angle', 'texture_glcm_contrast',
    'texture_gabor_response', 'deep_feature_1', 'deep_feature_2',
    'deep_feature_3', 'image_quality_score', 'diabetes_duration',
    'hba1c', 'fasting_glucose', 'systolic_bp', 'diastolic_bp',
    'age', 'bmi', 'medications'
]

def find_image(base_dir: Path, img_id: str):
    search_paths = [
        base_dir / 'data' / 'Augmented_Images' / f'{img_id}.jpg',
        base_dir / 'data' / 'Augmented_Images' / f'{img_id}.png',
        base_dir / 'data' / 'Augmented_Images' / f'{img_id}_orig.jpg',
        base_dir / 'data' / 'Augmented_Images' / f'{img_id}_aug_1.jpg',
        base_dir / 'data' / 'Preprocessed_Images' / f'{img_id}_preprocessed.jpg',
        base_dir / 'data' / 'Preprocessed_Images' / f'{img_id}.png',
        base_dir / 'data' / 'Preprocessed_Images' / f'{img_id}.jpg',
        base_dir / 'data' / 'Raw' / 'B. Disease Grading' / 'B. Disease Grading' / '1. Original Images' / 'a. Training Set' / f'{img_id}.jpg',
        base_dir / 'data' / 'Raw' / 'B. Disease Grading' / 'B. Disease Grading' / '1. Original Images' / 'b. Testing Set' / f'{img_id}.jpg',
        base_dir / 'data' / 'Raw' / 'A. Segmentation' / 'A. Segmentation' / '1. Original Images' / 'a. Training Set' / f'{img_id}.jpg',
        base_dir / 'data' / 'Raw' / 'A. Segmentation' / 'A. Segmentation' / '1. Original Images' / 'b. Testing Set' / f'{img_id}.jpg',
        base_dir / 'data' / 'Raw' / 'messidor-2' / 'preprocess' / f'{img_id}_PP.png',
        base_dir / 'data' / 'Raw' / 'messidor-2' / 'preprocess' / f'{img_id}.png',
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None

def load_augmented_training_data(base_dir: Path) -> pd.DataFrame:
    """
    Loads the balanced augmented images (5,067 samples) merged with 22 multimodal tabular features.
    """
    aug_man_path = base_dir / 'data' / 'augmented_images_manifest.csv'
    tab_path = base_dir / 'data' / 'multimodal_dr_dataset_22_features.csv'
    
    if aug_man_path.exists() and tab_path.exists():
        df_aug_man = pd.read_csv(aug_man_path)
        df_tab = pd.read_csv(tab_path)
        
        # Filter for train split
        train_man = df_aug_man[df_aug_man['split_type'] != 'test']
        merged = pd.merge(
            train_man,
            df_tab.drop(columns=['split_type', 'dr_stage_label', 'dataset_source'], errors='ignore'),
            left_on='original_image_id',
            right_on='image_id',
            suffixes=('', '_tab')
        )
        logger.info(f"Loaded {len(merged)} balanced augmented training samples with 22 tabular features.")
        return merged
    else:
        train_path = base_dir / 'data' / 'splits' / 'train_manifest.csv'
        return pd.read_csv(train_path)

class RetinalMultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: Path, tabular_features: np.ndarray, 
                 img_size: tuple = (224, 224), is_training: bool = False):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.tabular_features = tabular_features
        self.img_size = img_size
        self.is_training = is_training
        
        self.image_paths = []
        for idx in range(len(self.df)):
            img_id = str(self.df.iloc[idx]['image_id'])
            p = find_image(self.base_dir, img_id)
            self.image_paths.append(p)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_paths[idx]
        
        if img_path and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
            else:
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        else:
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            
        if self.is_training and random.random() > 0.5:
            img = np.fliplr(img).copy()
            
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        clin_tensor = torch.from_numpy(self.tabular_features[idx]).float()
        dr_stage = int(row.get('dr_stage_label', 0))
        
        exudates = float(row.get('exudates_count', 0))
        if exudates > 40:
            dme_grade = 2
        elif exudates > 5:
            dme_grade = 1
        else:
            dme_grade = 0
            
        severity = float(dr_stage)
        referable = 1.0 if (dr_stage >= 2 or dme_grade >= 1) else 0.0
        
        return {
            'image': img_tensor,
            'clinical': clin_tensor,
            'dr_label': torch.tensor(dr_stage, dtype=torch.long),
            'dme_label': torch.tensor(dme_grade, dtype=torch.long),
            'severity_label': torch.tensor(severity, dtype=torch.float),
            'referable_label': torch.tensor(referable, dtype=torch.float)
        }

import os
import sys
import argparse
import logging
import yaml
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from sklearn.metrics import cohen_kappa_score, accuracy_score

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.multi_task_head import DrishtiRakshakModel
from src.models.loss_functions import MultiTaskLoss
from src.backbones.tabular_mlp import ClinicalFeatureProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_COLUMNS = [
    'optic_disc_area', 'optic_cup_area', 'cup_to_disc_ratio',
    'exudates_count', 'hemorrhages_count', 'microaneurysms_count',
    'vessel_tortuosity', 'bifurcation_angle', 'texture_glcm_contrast',
    'texture_gabor_response', 'deep_feature_1', 'deep_feature_2',
    'deep_feature_3', 'image_quality_score', 'diabetes_duration',
    'hba1c', 'fasting_glucose', 'systolic_bp', 'diastolic_bp',
    'age', 'bmi', 'medications'
]

def load_config(config_name: str = 'pipeline_config.yaml') -> dict:
    config_path = Path(__file__).resolve().parents[1] / 'configs' / config_name
    if not config_path.exists():
        logger.warning(f'Config {config_name} not found.')
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_image(base_dir: Path, img_id: str):
    search_paths = [
        base_dir / 'data' / 'Preprocessed_Images' / f'{img_id}_preprocessed.jpg',
        base_dir / 'data' / 'Preprocessed_Images' / f'{img_id}.png',
        base_dir / 'data' / 'Preprocessed_Images' / f'{img_id}.jpg',
        base_dir / 'data' / 'Augmented_Images' / f'{img_id}.jpg',
        base_dir / 'data' / 'Augmented_Images' / f'{img_id}.png',
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

def train_one_epoch(model, dataloader, optimizer, criterion, scaler=None):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc='Training'):
        images = batch['image'].to(DEVICE)
        clinical = batch['clinical'].to(DEVICE)
        
        targets = {
            'dr_label': batch['dr_label'].to(DEVICE),
            'dme_label': batch['dme_label'].to(DEVICE),
            'severity_label': batch['severity_label'].to(DEVICE),
            'referable_label': batch['referable_label'].to(DEVICE)
        }
        
        optimizer.zero_grad()
        
        if scaler and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = model(image=images, clinical=clinical)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(image=images, clinical=clinical)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        total_loss += loss.item()
        
    return total_loss / max(1, len(dataloader))

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_dr_preds = []
    all_dr_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].to(DEVICE)
            clinical = batch['clinical'].to(DEVICE)
            
            targets = {
                'dr_label': batch['dr_label'].to(DEVICE),
                'dme_label': batch['dme_label'].to(DEVICE),
                'severity_label': batch['severity_label'].to(DEVICE),
                'referable_label': batch['referable_label'].to(DEVICE)
            }
            
            outputs = model(image=images, clinical=clinical)
            loss_dict = criterion(outputs, targets)
            total_loss += loss_dict['total_loss'].item()
            
            preds = torch.argmax(outputs['dr_logits'], dim=1)
            all_dr_preds.extend(preds.cpu().numpy())
            all_dr_labels.extend(targets['dr_label'].cpu().numpy())
            
    all_dr_preds = np.array(all_dr_preds)
    all_dr_labels = np.array(all_dr_labels)
    
    acc = accuracy_score(all_dr_labels, all_dr_preds)
    qwk = cohen_kappa_score(all_dr_labels, all_dr_preds, weights='quadratic')
    
    return total_loss / max(1, len(dataloader)), acc, qwk

def main():
    parser = argparse.ArgumentParser(description='Train DrishtiRakshak Real Models')
    parser.add_argument('--demo', action='store_true', help='Run on small subset for verification')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()
    
    config = load_config()
    train_config = config.get('training', {})
    
    epochs = args.epochs or train_config.get('epochs', 25)
    batch_size = args.batch_size or train_config.get('batch_size', 16)
    lr = args.lr or train_config.get('learning_rate', 1e-4)
    
    base_dir = Path(__file__).resolve().parents[1]
    splits_dir = base_dir / 'data' / 'splits'
    
    val_manifest = splits_dir / 'val_manifest.csv'
    if not val_manifest.exists():
        logger.info('Splits not found. Running split preparation...')
        import importlib
        prep_module = importlib.import_module('scripts.01_prepare_splits')
        prep_module.prepare_splits(str(base_dir / 'data' / 'multimodal_dr_dataset_22_features.csv'))
        
    from src.dataset import load_augmented_training_data, RetinalMultimodalDataset
    train_df = load_augmented_training_data(base_dir)
    val_df = pd.read_csv(val_manifest)
    
    logger.info(f'Loaded balanced multimodal training dataset: {len(train_df)} augmented train samples | {len(val_df)} val samples')
    
    if args.demo:
        logger.info('DEMO flag passed: Using subset (50 train, 20 val) for fast pipeline execution.')
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        epochs = min(epochs, 2)
        
    set_seed(train_config.get('seed', 42))
    
    logger.info('Fitting ClinicalFeatureProcessor on 22 multimodal tabular features...')
    processor = ClinicalFeatureProcessor(feature_columns=FEATURE_COLUMNS)
    train_clin_features = processor.fit_transform(train_df)
    val_clin_features = processor.transform(val_df)
    
    models_out_dir = base_dir / 'outputs' / 'models'
    models_out_dir.mkdir(parents=True, exist_ok=True)
    processor.save_statistics(str(models_out_dir / 'clinical_scaler_stats.json'))
    
    train_dataset = RetinalMultimodalDataset(
        df=train_df,
        base_dir=base_dir,
        tabular_features=train_clin_features,
        img_size=(224, 224),
        is_training=True
    )
    val_dataset = RetinalMultimodalDataset(
        df=val_df,
        base_dir=base_dir,
        tabular_features=val_clin_features,
        img_size=(224, 224),
        is_training=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info('Instantiating DrishtiRakshakModel (EfficientNet-CBAM + Tabular MLP + AGF + MultiTaskHead)...')
    model = DrishtiRakshakModel(clinical_in_dim=len(FEATURE_COLUMNS), pretrained=True).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MultiTaskLoss().to(DEVICE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_qwk = -1.0
    logger.info(f'Starting actual training on {DEVICE} for {epochs} epochs...')
    
    for epoch in range(epochs):
        logger.info(f'--- Epoch {epoch+1}/{epochs} ---')
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_qwk = validate(model, val_loader, criterion)
        scheduler.step()
        
        logger.info(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val QWK: {val_qwk:.4f}')
        
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_qwk': best_qwk,
                'val_acc': val_acc,
                'config': config
            }
            torch.save(checkpoint, models_out_dir / 'best_model.pth')
            logger.info(f'==> Saved new best model checkpoint with Val QWK: {best_qwk:.4f}')
            
    logger.info(f'Training completed. Best Validation QWK: {best_qwk:.4f}')

    # --- Train Stacking Ensemble on 512-d Fused Multimodal Embeddings ---
    logger.info("Extracting 512-d fused multimodal embeddings for Stacking Ensemble...")
    # Load best model weights
    best_ckpt = torch.load(models_out_dir / 'best_model.pth', map_location=DEVICE)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    def extract_features(loader):
        embeddings, labels = [], []
        with torch.no_grad():
            for batch in loader:
                imgs = batch['image'].to(DEVICE)
                clins = batch['clinical'].to(DEVICE)
                out = model(image=imgs, clinical=clins)
                embeddings.append(out['fused'].cpu().numpy())
                labels.append(batch['dr_label'].cpu().numpy())
        return np.vstack(embeddings), np.concatenate(labels)

    train_emb, train_lbl = extract_features(train_loader)
    val_emb, val_lbl = extract_features(val_loader)

    from src.models.stacking_ensemble import StackingEnsemble
    logger.info(f"Training Stacking Ensemble (CatBoost, XGBoost, Random Forest) on {len(train_emb)} embeddings...")
    ensemble = StackingEnsemble()
    ensemble.train(train_emb, train_lbl, val_emb, val_lbl)

    ensemble_dir = models_out_dir / "ensemble"
    ensemble.save(str(ensemble_dir))
    logger.info(f"==> Successfully saved trained Stacking Ensemble to {ensemble_dir}")

if __name__ == '__main__':
    main()

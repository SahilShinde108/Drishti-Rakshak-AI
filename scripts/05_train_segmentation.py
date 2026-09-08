import os
import sys
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.segmentation.swin_unet_vessel import SwinUNetVessel
from src.segmentation.concat_unet_lesion import ConcatUNetLesion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss specifically designed for extreme class imbalance in medical segmentation.
    Penalizes False Positives (alpha=0.7) more than False Negatives (beta=0.3) to prevent over-segmentation.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten tensors per channel
        pred = pred.contiguous()
        target = target.contiguous()

        tp = (pred * target).sum(dim=[0, 2, 3])
        fp = (pred * (1 - target)).sum(dim=[0, 2, 3])
        fn = ((1 - pred) * target).sum(dim=[0, 2, 3])

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)
        return focal_tversky.mean()

class IDRiDSegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, gt_base_dir: Path, img_size: tuple = (224, 224)):
        self.image_dir = Path(image_dir)
        self.gt_base_dir = Path(gt_base_dir)
        self.img_size = img_size
        
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        logger.info(f"Loaded {len(self.image_files)} IDRiD scans from {self.image_dir.name}")

    def __len__(self):
        return len(self.image_files)

    def _load_mask(self, folder_name: str, suffix: str, img_id: str) -> np.ndarray:
        p = self.gt_base_dir / folder_name / f"{img_id}{suffix}"
        if p.exists():
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                return (mask > 0).astype(np.float32)
        return np.zeros(self.img_size, dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_id = img_path.stem
        
        # 1. Load Fundus Image
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
        else:
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # 2. Load 4 Lesion Ground Truth Masks (MA, HE, EX, SE)
        ma_mask = self._load_mask("1. Microaneurysms", "_MA.tif", img_id)
        he_mask = self._load_mask("2. Haemorrhages", "_HE.tif", img_id)
        ex_mask = self._load_mask("3. Hard Exudates", "_EX.tif", img_id)
        se_mask = self._load_mask("4. Soft Exudates", "_SE.tif", img_id)
        
        lesion_tensor = torch.from_numpy(np.stack([ma_mask, he_mask, ex_mask, se_mask], axis=0)).float()

        # 3. Load Vessel & Optic Disc / Cup Masks
        od_mask = self._load_mask("5. Optic Disc", "_OD.tif", img_id)
        
        # Cup ground truth: interior central region of optic disc (35% disc area)
        cup_mask = np.zeros_like(od_mask)
        if od_mask.sum() > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cup_mask = cv2.erode(od_mask, kernel, iterations=2)
            
        # Retinal Vessel mask: green-channel adaptive threshold
        green = img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_clahe = clahe.apply(green)
        vessel_mask = cv2.adaptiveThreshold(g_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        vessel_mask = cv2.medianBlur(vessel_mask, 3)
        vessel_mask = (vessel_mask > 0).astype(np.float32)

        vessel_tensor = torch.from_numpy(np.stack([vessel_mask, od_mask, cup_mask], axis=0)).float()

        return {
            'image': img_tensor,
            'lesion_masks': lesion_tensor,
            'vessel_masks': vessel_tensor,
            'image_id': img_id
        }

def train_lesion_segmentor(train_loader, val_loader, epochs: int = 15, lr: float = 3e-4):
    logger.info("\n" + "=" * 65)
    logger.info("  TRAINING CONCAT-UNET LESION MODEL (4-CH: MA, HE, EX, SE)")
    logger.info("=" * 65)
    
    model = ConcatUNetLesion(in_channels=3, out_channels=4).to(DEVICE)
    criterion = FocalTverskyLoss(alpha=0.75, beta=0.25, gamma=1.33)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    output_dir = BASE_DIR / "outputs" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    chk_path = output_dir / "concat_unet_lesion.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Lesion Epoch {epoch+1}/{epochs}"):
            imgs = batch['image'].to(DEVICE)
            targets = batch['lesion_masks'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            pred_tensor = torch.cat([outputs['ma'], outputs['he'], outputs['ex'], outputs['se']], dim=1)
            
            loss = criterion(pred_tensor, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(DEVICE)
                targets = batch['lesion_masks'].to(DEVICE)
                outputs = model(imgs)
                pred_tensor = torch.cat([outputs['ma'], outputs['he'], outputs['ex'], outputs['se']], dim=1)
                val_loss += criterion(pred_tensor, targets).item()
        val_loss /= max(1, len(val_loader))

        logger.info(f"Epoch {epoch+1}/{epochs} | Lesion Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), chk_path)
            logger.info(f"==> Saved improved ConcatUNetLesion weights (Val Loss: {best_loss:.4f})")

    logger.info(f"ConcatUNetLesion Training Complete. Optimal Val Loss: {best_loss:.4f}")
    return chk_path

def train_vessel_segmentor(train_loader, val_loader, epochs: int = 15, lr: float = 3e-4):
    logger.info("\n" + "=" * 65)
    logger.info("  TRAINING SWIN-UNET PROGRESSIVE VESSEL & DISC SEGMENTATION")
    logger.info("=" * 65)

    model = SwinUNetVessel(in_channels=3, out_channels=3).to(DEVICE)
    criterion = FocalTverskyLoss(alpha=0.65, beta=0.35, gamma=1.2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    output_dir = BASE_DIR / "outputs" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    chk_path = output_dir / "swin_unet_vessel.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Vessel Epoch {epoch+1}/{epochs}"):
            imgs = batch['image'].to(DEVICE)
            targets = batch['vessel_masks'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            pred_tensor = torch.cat([outputs['vessel'], outputs['disc'], outputs['cup']], dim=1)

            loss = criterion(pred_tensor, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(DEVICE)
                targets = batch['vessel_masks'].to(DEVICE)
                outputs = model(imgs)
                pred_tensor = torch.cat([outputs['vessel'], outputs['disc'], outputs['cup']], dim=1)
                val_loss += criterion(pred_tensor, targets).item()
        val_loss /= max(1, len(val_loader))

        logger.info(f"Epoch {epoch+1}/{epochs} | Swin-UNet Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), chk_path)
            logger.info(f"==> Saved improved SwinUNetVessel weights (Val Loss: {best_loss:.4f})")

    logger.info(f"SwinUNetVessel Training Complete. Optimal Val Loss: {best_loss:.4f}")
    return chk_path

def main():
    parser = argparse.ArgumentParser(description="Train Drishti-Rakshak Segmentation Networks")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    def find_dir(pattern):
        matches = list((BASE_DIR / "data" / "Raw").glob(pattern))
        return matches[0] if matches else None

    train_img_dir = find_dir("**/A*Segmentation/**/1. Original Images/a. Training Set")
    train_gt_dir = find_dir("**/A*Segmentation/**/2. All Segmentation Groundtruths/a. Training Set")
    test_img_dir = find_dir("**/A*Segmentation/**/1. Original Images/b. Testing Set")
    test_gt_dir = find_dir("**/A*Segmentation/**/2. All Segmentation Groundtruths/b. Testing Set")

    if not train_img_dir or not train_gt_dir:
        logger.error("IDRiD Segmentation directories not found.")
        sys.exit(1)

    logger.info(f"Loading IDRiD Segmentation Datasets on {DEVICE}...")
    train_dataset = IDRiDSegmentationDataset(train_img_dir, train_gt_dir, img_size=(224, 224))
    val_dataset = IDRiDSegmentationDataset(test_img_dir, test_gt_dir, img_size=(224, 224))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 1. Train ConcatUNetLesion (4-channel lesion model with Focal Tversky)
    lesion_path = train_lesion_segmentor(train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    # 2. Train SwinUNetVessel (Progressive multi-stage decoder)
    vessel_path = train_vessel_segmentor(train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    logger.info("\n" + "=" * 65)
    logger.info("  UPGRADED SEGMENTATION NETWORKS TRAINED & SAVED SUCCESSFULLY")
    logger.info("=" * 65)
    logger.info(f"1. ConcatUNetLesion Checkpoint: {lesion_path}")
    logger.info(f"2. SwinUNetVessel Checkpoint   : {vessel_path}")
    logger.info("=" * 65 + "\n")

if __name__ == "__main__":
    main()

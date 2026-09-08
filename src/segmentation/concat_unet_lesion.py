import numpy as np
import cv2
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "models_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        red = max(1, in_channels // reduction)
        self.ca_mlp = nn.Sequential(
            nn.Linear(in_channels, red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(red, in_channels, bias=False)
        )
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pool = torch.mean(x, dim=[2, 3])
        max_pool = torch.amax(x, dim=[2, 3])
        ca_out = self.ca_mlp(avg_pool) + self.ca_mlp(max_pool)
        ca_out = self.sigmoid(ca_out).view(b, c, 1, 1)
        x = x * ca_out
        
        avg_pool_s = torch.mean(x, dim=1, keepdim=True)
        max_pool_s, _ = torch.max(x, dim=1, keepdim=True)
        sa_in = torch.cat([avg_pool_s, max_pool_s], dim=1)
        sa_out = self.sigmoid(self.sa_conv(sa_in))
        return x * sa_out

class DoubleConv(nn.Module):
    """(Conv -> BN -> GELU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Up(nn.Module):
    """Upscaling then double conv with skip connection"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if shapes differ slightly
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_x > 0 or diff_y > 0:
            x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                       diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConcatUNetLesion(nn.Module):
    """
    4-Stage Multi-Scale Encoder-Decoder U-Net with CBAM Attention
    Predicts 4 distinct pathological lesion channels: MA, HE, EX, SE.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # Bottleneck Attention
        self.cbam = CBAMBlock(256)

        # Decoder
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up3 = Up(64 + 32, 32)

        # Output head
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder
        x1 = self.inc(x)       # (B, 32, H, W)
        x2 = self.down1(x1)    # (B, 64, H/2, W/2)
        x3 = self.down2(x2)    # (B, 128, H/4, W/4)
        x4 = self.down3(x3)    # (B, 256, H/8, W/8)

        # Bottleneck CBAM
        x4 = self.cbam(x4)

        # Decoder with skip connections
        d1 = self.up1(x4, x3)  # (B, 128, H/4, W/4)
        d2 = self.up2(d1, x2)  # (B, 64, H/2, W/2)
        d3 = self.up3(d2, x1)  # (B, 32, H, W)

        logits = self.outc(d3)
        out = torch.sigmoid(logits)

        return {
            'ma': out[:, 0:1, :, :],
            'he': out[:, 1:2, :, :],
            'ex': out[:, 2:3, :, :],
            'se': out[:, 3:4, :, :]
        }

class LesionSegmentor:
    def __init__(self, weights_path: str = None, device: str = 'auto'):
        self.device = DEVICE if device == 'auto' else torch.device(device)
        self.model = ConcatUNetLesion().to(self.device)
        self.has_weights = False
        default_path = Path(__file__).resolve().parents[2] / "outputs" / "models" / "concat_unet_lesion.pth"
        actual_path = weights_path or (default_path if default_path.exists() else None)
        
        if actual_path and Path(actual_path).exists():
            try:
                ckpt = torch.load(actual_path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.model.load_state_dict(ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(ckpt)
                self.has_weights = True
                logger.info(f"Loaded trained ConcatUNetLesion weights from {actual_path}")
            except Exception as e:
                logger.warning(f"Error loading weights from {actual_path}: {e}. Fallback active.")
        else:
            logger.warning("[DEMO MODE] No weights found for LesionSegmentor. Using color thresholding fallbacks.")

    def segment(self, image: np.ndarray, threshold: float = None) -> Dict[str, np.ndarray]:
        h, w = image.shape[:2]
        if not self.has_weights:
            # Color thresholding fallback [DEMO]
            red_channel = image[:,:,0] > 200
            yellow_channel = (image[:,:,0] > 200) & (image[:,:,1] > 200)
            return {
                'ma': red_channel,
                'he': red_channel,
                'ex': yellow_channel,
                'se': np.zeros((h, w), dtype=bool)
            }
            
        self.model.eval()
        with torch.no_grad():
            img_resized = cv2.resize(image, (224, 224))
            img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            img_tensor = ((img_tensor / 255.0) - mean) / std
            out = self.model(img_tensor)
            results = {}
            calibrated_thresholds = {'ma': 0.65, 'he': 0.80, 'ex': 0.75, 'se': 0.75}
            for k, v in out.items():
                thresh = threshold if threshold is not None else calibrated_thresholds.get(k, 0.5)
                mask_224 = (v.squeeze().cpu().numpy() > thresh).astype(np.uint8)
                mask_hw = cv2.resize(mask_224, (w, h), interpolation=cv2.INTER_NEAREST)
                results[k] = mask_hw > 0
            return results

    def compute_lesion_counts(self, masks: Dict[str, np.ndarray]) -> Dict[str, int]:
        from scipy.ndimage import label
        counts = {}
        for k, v in masks.items():
            labeled_array, num_features = label(v)
            counts[f"{k}_count"] = int(num_features)
        return counts

    def compute_lesion_density(self, masks: Dict[str, np.ndarray], retina_area: float) -> Dict[str, float]:
        densities = {}
        if retina_area <= 0:
            return {f"{k}_density": 0.0 for k in masks.keys()}
        for k, v in masks.items():
            densities[f"{k}_density"] = float(np.sum(v) / retina_area)
        return densities

    def compute_exudate_fovea_distance(self, ex_mask: np.ndarray, fovea_center: tuple = None) -> float:
        if ex_mask is None or np.sum(ex_mask) == 0:
            return 999.0
        h, w = ex_mask.shape[:2]
        if fovea_center is None:
            fovea_center = (w // 2, h // 2)
        y_indices, x_indices = np.where(ex_mask > 0)
        if len(y_indices) == 0:
            return 999.0
        distances = np.sqrt((x_indices - fovea_center[0])**2 + (y_indices - fovea_center[1])**2)
        return float(np.min(distances))

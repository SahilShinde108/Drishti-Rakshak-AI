import torch
import torch.nn as nn
import cv2
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class EnlightenGANGenerator(nn.Module):
    """U-Net style generator for EnlightenGAN."""
    def __init__(self, in_channels=3, out_channels=3, num_residuals=4):
        super(EnlightenGANGenerator, self).__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
            
        # Residual blocks
        for _ in range(num_residuals):
            model += [ResidualBlock(in_features)]
            
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
            
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class EnlightenGANEnhancer:
    def __init__(self, weights_path: Optional[str] = None):
        self.config = load_config()
        self.enabled = self.config.get("preprocessing", {}).get("enlightengan_enabled", False)
        
        self.is_demo_mode = True
        self.model = EnlightenGANGenerator().to(DEVICE)
        
        if self.enabled and weights_path and Path(weights_path).exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            self.model.eval()
            self.is_demo_mode = False
            logger.info("EnlightenGAN weights loaded. Running in production mode.")
        else:
            logger.warning("EnlightenGAN weights not found or disabled. Running in DEMO/FALLBACK mode.")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self.is_demo_mode:
            logger.info("[DEMO/FALLBACK] Using CLAHE + Gamma instead of EnlightenGAN.")
            # Convert to LAB for luminance enhancement
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # CLAHE on L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                
                # Gamma correction to brighten
                gamma = 0.8
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                cl_gamma = cv2.LUT(cl, table)
                
                merged = cv2.merge((cl_gamma, a, b))
                return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(image)
                gamma = 0.8
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(cl, table)
                
        # Production Mode
        self.model.eval()
        with torch.no_grad():
            img_tensor = torch.from_numpy(image).float() / 255.0
            if len(image.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            else:
                img_tensor = img_tensor.unsqueeze(0)
                
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            
            enhanced = self.model(img_tensor)
            
            enhanced = enhanced.squeeze(0).cpu().numpy()
            if len(image.shape) == 3:
                enhanced = np.transpose(enhanced, (1, 2, 0))
            else:
                enhanced = enhanced.squeeze(0)
                
            enhanced = (enhanced * 255.0).clip(0, 255).astype(np.uint8)
            return enhanced

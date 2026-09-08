import torch
import torch.nn as nn
import yaml
from pathlib import Path
import logging
from typing import Optional, List
try:
    import timm
except ImportError:
    timm = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_name: str = "models_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EfficientNetCBAM(nn.Module):
    def __init__(self, model_name: str = 'efficientnet_b0', pretrained: bool = True, freeze_layers: bool = False, dropout: float = 0.2):
        super().__init__()
        if timm is not None:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        else:
            logger.warning("timm not installed. Using dummy sequential model.")
            self.backbone = nn.Sequential(nn.Conv2d(3, 1280, kernel_size=3, stride=2, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.feature_dim = 1280

        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.cbam = CBAM(self.feature_dim) if timm is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def get_feature_dim(self) -> int:
        return self.feature_dim
        
    def get_last_conv_layer(self) -> nn.Module:
        # Simplification to get the last conv block for GradCAM
        return self.backbone.conv_head if hasattr(self.backbone, 'conv_head') else self.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming timm model with features extracted right before the pooling layer
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
            features = self.cbam(features)
            out = self.backbone.global_pool(features)
        else:
            out = self.backbone(x)
        return self.dropout(out)

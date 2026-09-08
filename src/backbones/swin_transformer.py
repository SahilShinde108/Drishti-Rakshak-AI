import torch
import torch.nn as nn
import yaml
from pathlib import Path
import logging
from typing import List, Optional
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

class SwinTransformerBackbone(nn.Module):
    def __init__(self, variant: str = 'tiny', pretrained: bool = True, img_size: int = 224, dropout: float = 0.2):
        super().__init__()
        self.variant = variant
        model_name = f'swin_{variant}_patch4_window7_224'
        
        if timm is not None:
            # Load swin without classification head
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        else:
            logger.warning("timm not installed. Using dummy sequential model for Swin.")
            self.backbone = nn.Sequential(nn.Conv2d(3, 768, kernel_size=3, stride=2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.feature_dim = 768

        self.dropout = nn.Dropout(dropout)
        
    def get_feature_dim(self) -> int:
        return self.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.dropout(out)
        
    def get_intermediate_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        if not hasattr(self.backbone, 'forward_intermediates'):
            logger.warning("This model does not support get_intermediate_features via timm.")
            return []
        # Return features from different stages
        return self.backbone.forward_intermediates(x)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder for extracting attention maps which typically requires model hooks
        logger.warning("get_attention_maps requires custom hooks on Swin Transformer blocks.")
        return torch.tensor([])

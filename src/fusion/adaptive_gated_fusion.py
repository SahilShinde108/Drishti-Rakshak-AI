import torch
import torch.nn as nn
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "models_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using default config.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Gated Fusion (AGF) for combining image and clinical features.
    """
    def __init__(self, image_dim: int, clinical_dim: int, fused_dim: int = 512, dropout: float = 0.2):
        super(AdaptiveGatedFusion, self).__init__()
        
        config = load_config()
        agf_cfg = config.get('adaptive_gated_fusion', {})
        self.fused_dim = agf_cfg.get('fused_dim', fused_dim)
        self.dropout_rate = agf_cfg.get('dropout', dropout)
        
        self.image_projection = nn.Linear(image_dim, self.fused_dim)
        self.clinical_projection = nn.Linear(clinical_dim, self.fused_dim)
        
        self.gate_network = nn.Sequential(
            nn.Linear(self.fused_dim * 2, self.fused_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Buffer for statistics
        self.last_gate_values = None
        
    def forward(self, image_features: torch.Tensor, clinical_features: torch.Tensor) -> dict:
        """
        Args:
            image_features: Tensor of shape (B, image_dim)
            clinical_features: Tensor of shape (B, clinical_dim)
        Returns:
            Dict containing fused embeddings and interpretability metrics.
        """
        if image_features.size(0) != clinical_features.size(0):
            raise ValueError(f"Batch size mismatch: image {image_features.size(0)} != clinical {clinical_features.size(0)}")
            
        f_img_proj = self.image_projection(image_features)
        f_clin_proj = self.clinical_projection(clinical_features)
        
        # Concatenate to compute gate
        cat_features = torch.cat([f_img_proj, f_clin_proj], dim=-1)
        g = self.gate_network(cat_features)
        
        self.last_gate_values = g.detach()
        
        # Gated fusion
        image_contribution = g * f_img_proj
        clinical_contribution = (1 - g) * f_clin_proj
        
        fused = image_contribution + clinical_contribution
        fused = self.dropout(fused)
        
        return {
            'fused': fused,
            'gate_values': g,
            'image_contribution': image_contribution,
            'clinical_contribution': clinical_contribution
        }
        
    def get_output_dim(self) -> int:
        return self.fused_dim
        
    def get_gate_statistics(self) -> dict:
        """
        Returns mean/std of gate values (image vs clinical weighting) from the last forward pass.
        """
        if self.last_gate_values is None:
            return {'mean': 0.0, 'std': 0.0}
            
        return {
            'image_weight_mean': self.last_gate_values.mean().item(),
            'image_weight_std': self.last_gate_values.std().item(),
            'clinical_weight_mean': (1 - self.last_gate_values).mean().item(),
            'clinical_weight_std': (1 - self.last_gate_values).std().item()
        }

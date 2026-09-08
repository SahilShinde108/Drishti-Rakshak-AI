import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using default config.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, label_smoothing: float = 0.1, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 1 and logits.size(1) > 1: # Multi-class
            ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none', label_smoothing=self.label_smoothing)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        else: # Binary
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.alpha, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * bce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class OrdinalRegressionLoss(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLoss, self).__init__()
        self.criterion = nn.SmoothL1Loss()
        
    def forward(self, predicted_severity: torch.Tensor, true_severity: torch.Tensor) -> torch.Tensor:
        return self.criterion(predicted_severity, true_severity.float())

class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        super(MultiTaskLoss, self).__init__()
        config = load_config()
        loss_cfg = config.get('loss', {})
        
        self.weights = task_weights or loss_cfg.get('task_weights', {
            'dr': 1.0,
            'dme': 1.0,
            'severity': 0.5,
            'triage': 1.0
        })
        
        self.dr_loss_fn = FocalLoss(gamma=2.0)
        self.dme_loss_fn = FocalLoss(gamma=2.0)
        self.severity_loss_fn = OrdinalRegressionLoss()
        self.triage_loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dr_loss = self.dr_loss_fn(predictions['dr_logits'], targets['dr_label'])
        dme_loss = self.dme_loss_fn(predictions['dme_logits'], targets['dme_label'])
        severity_loss = self.severity_loss_fn(predictions['severity_score'], targets['severity_label'])
        triage_loss = self.triage_loss_fn(predictions['referable_logit'], targets['referable_label'].float())
        
        total_loss = (
            self.weights.get('dr', 1.0) * dr_loss +
            self.weights.get('dme', 1.0) * dme_loss +
            self.weights.get('severity', 0.5) * severity_loss +
            self.weights.get('triage', 1.0) * triage_loss
        )
        
        return {
            'total_loss': total_loss,
            'dr_loss': dr_loss,
            'dme_loss': dme_loss,
            'severity_loss': severity_loss,
            'triage_loss': triage_loss
        }

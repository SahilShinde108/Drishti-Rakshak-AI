import torch
import torch.nn as nn
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "models_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using default config.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class MultiTaskHead(nn.Module):
    """
    Multi-task prediction heads for DR, DME, Severity, and Referable classification.
    """
    def __init__(self, input_dim: int = 512, dr_classes: int = 5, dme_classes: int = 3, hidden_dim: int = 256, dropout: float = 0.3):
        super(MultiTaskHead, self).__init__()
        
        self.shared_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Task 1: DR classification
        self.dr_head = nn.Linear(hidden_dim, dr_classes)
        
        # Task 2: DME classification
        self.dme_head = nn.Linear(hidden_dim, dme_classes)
        
        # Task 3: Severity regression
        self.severity_head = nn.Linear(hidden_dim, 1)
        
        # Task 4: Binary referable triage
        self.referable_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared_hidden(fused_embedding)
        
        dr_logits = self.dr_head(h)
        dme_logits = self.dme_head(h)
        severity_raw = self.severity_head(h)
        referable_logit = self.referable_head(h)
        
        dr_probs = torch.softmax(dr_logits, dim=-1)
        dr_pred = torch.argmax(dr_probs, dim=-1)
        
        dme_probs = torch.softmax(dme_logits, dim=-1)
        dme_pred = torch.argmax(dme_probs, dim=-1)
        
        # Severity mapped to [0.0, 4.0]
        severity_score = torch.sigmoid(severity_raw) * 4.0
        
        referable_prob = torch.sigmoid(referable_logit)
        referable_pred = (referable_prob >= 0.5).long()
        
        return {
            'dr_logits': dr_logits,
            'dr_probs': dr_probs,
            'dr_pred': dr_pred,
            'dme_logits': dme_logits,
            'dme_probs': dme_probs,
            'dme_pred': dme_pred,
            'severity_score': severity_score.squeeze(-1),
            'referable_logit': referable_logit.squeeze(-1),
            'referable_prob': referable_prob.squeeze(-1),
            'referable_pred': referable_pred.squeeze(-1)
        }
        
    @staticmethod
    def compute_referable_from_predictions(dr_pred: torch.Tensor, dme_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute deterministic referable triage: DR >= 2 OR DME >= 1
        """
        return ((dr_pred >= 2) | (dme_pred >= 1)).long()


class DrishtiRakshakModel(nn.Module):
    """
    End-to-End Multimodal model for Drishti-Rakshak AI.
    Combines EfficientNet-CBAM, Tabular MLP, Eye-Pair LSTM, Adaptive Gated Fusion, and Multi-Task heads.
    """
    def __init__(self, clinical_in_dim: int = 22, pretrained: bool = True, cnn_backbone: str = 'efficientnet_b0'):
        super(DrishtiRakshakModel, self).__init__()
        
        try:
            from src.backbones.efficientnet_cbam import EfficientNetCBAM
            self.visual_encoder = EfficientNetCBAM(model_name=cnn_backbone, pretrained=pretrained)
            self.image_dim = self.visual_encoder.get_feature_dim()
        except Exception as e:
            logger.warning(f"EfficientNetCBAM load error ({e}). Using adaptive fallback CNN.")
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 512),
                nn.GELU()
            )
            self.image_dim = 512
            
        try:
            from src.backbones.tabular_mlp import TabularMLP
            self.clinical_encoder = TabularMLP(in_features=clinical_in_dim, out_features=128)
            self.clinical_dim = 128
        except Exception:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(clinical_in_dim, 128),
                nn.LayerNorm(128),
                nn.GELU()
            )
            self.clinical_dim = 128
        
        try:
            from src.fusion.eye_pair_lstm import EyePairLSTM
            self.eye_pair_lstm = EyePairLSTM(input_dim=self.image_dim, hidden_dim=256, output_dim=256)
            self.lstm_out_dim = self.eye_pair_lstm.get_output_dim()
        except Exception:
            self.eye_pair_lstm = nn.Linear(self.image_dim, 256)
            self.lstm_out_dim = 256
        
        try:
            from src.fusion.adaptive_gated_fusion import AdaptiveGatedFusion
            self.fusion = AdaptiveGatedFusion(
                image_dim=self.lstm_out_dim,
                clinical_dim=self.clinical_dim,
                fused_dim=512
            )
            self.fused_dim = self.fusion.get_output_dim()
        except Exception:
            self.fusion = None
            self.fused_dim = 512
            self.fallback_fusion = nn.Linear(self.lstm_out_dim + self.clinical_dim, 512)
        
        self.head = MultiTaskHead(
            input_dim=self.fused_dim,
            dr_classes=5,
            dme_classes=3
        )
        
    def forward(self, image: torch.Tensor, clinical: Optional[torch.Tensor] = None, 
                clinical_features: Optional[torch.Tensor] = None, 
                bilateral_image: Optional[torch.Tensor] = None) -> dict:
        """
        Full multimodal forward pass.
        Args:
            image: (B, 3, H, W) raw image or (B, D) image feature vector
            clinical / clinical_features: (B, 22) raw clinical tabular tensor
            bilateral_image: Optional (B, 3, H, W) fellow eye image
        """
        clin = clinical if clinical is not None else clinical_features
        
        # 1. Visual Feature Extraction
        if image.dim() == 4:
            v_feat = self.visual_encoder(image)
        else:
            v_feat = image
            
        # 2. Clinical Feature Extraction
        if clin is not None:
            c_feat = self.clinical_encoder(clin)
        else:
            c_feat = torch.zeros(image.size(0), self.clinical_dim, device=image.device)
            
        # 3. Eye-Pair / Sequential Modeling
        if hasattr(self.eye_pair_lstm, 'forward_bilateral') and bilateral_image is not None:
            if bilateral_image.dim() == 4:
                b_feat = self.visual_encoder(bilateral_image)
            else:
                b_feat = bilateral_image
            img_emb = self.eye_pair_lstm.forward_bilateral(v_feat, b_feat)
        elif hasattr(self.eye_pair_lstm, 'forward_single_eye'):
            img_emb = self.eye_pair_lstm.forward_single_eye(v_feat)
        else:
            img_emb = self.eye_pair_lstm(v_feat)
            
        # 4. Adaptive Gated Fusion
        if self.fusion is not None and hasattr(self.fusion, 'forward'):
            fusion_out = self.fusion(img_emb, c_feat)
            fused = fusion_out['fused']
        else:
            fused = torch.relu(self.fallback_fusion(torch.cat([img_emb, c_feat], dim=-1)))
            fusion_out = {'fused': fused}
            
        # 5. Multi-Task Head
        head_out = self.head(fused)
        
        return {**fusion_out, **head_out}
        
    def get_fused_embedding(self, image: torch.Tensor, clinical: torch.Tensor, bilateral_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.forward(image=image, clinical=clinical, bilateral_image=bilateral_image)
        return out['fused']
        
    def get_all_feature_dims(self) -> dict:
        return {
            'image_dim': self.image_dim,
            'clinical_dim': self.clinical_dim,
            'lstm_out': self.lstm_out_dim,
            'fused_dim': self.fused_dim
        }

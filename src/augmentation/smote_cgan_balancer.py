import numpy as np
import torch
import torch.nn as nn
import logging
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
try:
    from imblearn.combine import SMOTEENN
except ImportError:
    SMOTEENN = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        logger.warning(f"Could not load {config_path}. Using empty dict.")
        return {}

class SMOTEENNBalancer:
    def __init__(self, k_neighbors: int = 5, sampling_strategy: str = 'auto'):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        if SMOTEENN is None:
            logger.warning("imblearn not installed. Please install imbalanced-learn.")

    def balance(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if SMOTEENN is None:
            return features, labels
            
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Original distribution: {dict(zip(unique, counts))}")
        
        smote_enn = SMOTEENN(sampling_strategy=self.sampling_strategy)
        balanced_features, balanced_labels = smote_enn.fit_resample(features, labels)
        
        unique_b, counts_b = np.unique(balanced_labels, return_counts=True)
        logger.info(f"Balanced distribution: {dict(zip(unique_b, counts_b))}")
        return balanced_features, balanced_labels

class CGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_emb(labels)
        x = torch.cat([noise, c], -1)
        return self.model(x)

class CGANDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_emb(labels)
        x = torch.cat([features, c], -1)
        return self.model(x)

class ConditionalGANBalancer:
    def __init__(self, latent_dim: int = 100, feature_dim: int = 128, num_classes: int = 5):
        self.latent_dim = latent_dim
        self.generator = CGANGenerator(latent_dim, feature_dim, num_classes).to(DEVICE)
        self.discriminator = CGANDiscriminator(feature_dim, num_classes).to(DEVICE)
        self.is_trained = False

    def train(self, features: np.ndarray, labels: np.ndarray, epochs: int = 100):
        # Implementation of GAN training loop would go here
        self.is_trained = True
        pass

    def generate(self, class_label: int, num_samples: int) -> np.ndarray:
        if not self.is_trained:
            logger.warning("[DEMO MODE] CGAN not trained, returning random noise.")
            return np.random.randn(num_samples, self.generator.model[-1].out_features)
            
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(DEVICE)
            labels = torch.full((num_samples,), class_label, dtype=torch.long).to(DEVICE)
            gen_features = self.generator(z, labels).cpu().numpy()
        return gen_features

class ClassBalancer:
    def __init__(self, config_name: str = "pipeline_config.yaml"):
        self.config = load_config(config_name)
        self.enable = self.config.get("balancing", {}).get("enable", True)
        
    def balance(self, features: np.ndarray, labels: np.ndarray, method: str = 'smote_enn', is_train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not is_train or not self.enable:
            return features, labels
            
        if method == 'smote_enn':
            balancer = SMOTEENNBalancer()
            return balancer.balance(features, labels)
        elif method == 'cgan':
            logger.warning("CGAN standalone balancing not fully implemented yet.")
            return features, labels
        else:
            return features, labels

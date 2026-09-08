import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_name: str = "models_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

class TabularMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int = 128, dropout: float = 0.3):
        super().__init__()
        self.out_features = out_features
        
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.LayerNorm(256),
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.LayerNorm(128),
            nn.Linear(128, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_feature_dim(self) -> int:
        return self.out_features

class ClinicalFeatureProcessor:
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.stats = {}
        self.is_fitted = False

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in self.feature_columns:
            if col not in df_clean.columns:
                logger.warning(f"Column {col} missing from DataFrame. Adding default.")
                df_clean[col] = 0.0
            
            if df_clean[col].dtype.kind in 'biufc':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        return df_clean

    def fit(self, df: pd.DataFrame) -> 'ClinicalFeatureProcessor':
        df = self.handle_missing(df)
        for col in self.feature_columns:
            if df[col].dtype.kind in 'biufc':
                self.stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()) if float(df[col].std()) > 0 else 1.0
                }
            else:
                # Basic encoding for categorical mapping if needed
                pass
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Processor not fitted.")
            
        df = self.handle_missing(df)
        out_features = []
        for col in self.feature_columns:
            if col in self.stats:
                val = (df[col] - self.stats[col]['mean']) / self.stats[col]['std']
                out_features.append(val.values)
            else:
                # Handle unknown/categorical mapping
                out_features.append(np.zeros(len(df)))
                
        return np.column_stack(out_features).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        return self.feature_columns

    def save_statistics(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.stats, f)

    def load_statistics(self, path: str):
        with open(path, 'r') as f:
            self.stats = json.load(f)
        self.is_fitted = True

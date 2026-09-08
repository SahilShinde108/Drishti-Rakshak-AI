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

class EyePairLSTM(nn.Module):
    """
    Bilateral eye-pair LSTM fusion module.
    Processes features from both eyes (or a single eye) using a Bidirectional LSTM.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_layers: int = 2, bidirectional: bool = True, dropout: float = 0.3):
        super(EyePairLSTM, self).__init__()
        
        # Explicit arguments take precedence; config provides fallbacks
        config = load_config()
        lstm_cfg = config.get('eye_pair_lstm', {})
        self.input_dim = input_dim if input_dim is not None else lstm_cfg.get('input_dim', 512)
        self.hidden_dim = hidden_dim if hidden_dim is not None else lstm_cfg.get('hidden_dim', 256)
        self.num_layers = num_layers if num_layers is not None else lstm_cfg.get('num_layers', 2)
        self.bidirectional = bidirectional if bidirectional is not None else lstm_cfg.get('bidirectional', True)
        self.dropout_rate = dropout if dropout is not None else lstm_cfg.get('dropout', 0.3)
        
        self.projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2 if self.bidirectional else self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0
        )
        
    def forward(self, features_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features_sequence: Tensor of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            Tensor of shape (Batch, hidden_dim)
        """
        projected = self.projection(features_sequence) # (B, S, H)
        lstm_out, (hn, cn) = self.lstm(projected)
        # Taking the last hidden state of the sequence
        # If bidirectional, hn shape is (num_layers * num_directions, B, hidden_size)
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states from the last layer
            hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            hidden = hn[-1, :, :]
            
        return hidden
        
    def forward_single_eye(self, features: torch.Tensor) -> torch.Tensor:
        """
        Handle single eye case by duplicating the features to form a sequence of length 2.
        Args:
            features: Tensor of shape (Batch, Input_Dim)
        """
        seq = torch.stack([features, features], dim=1) # (B, 2, Input_Dim)
        return self.forward(seq)
        
    def forward_bilateral(self, right_features: torch.Tensor, left_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            right_features: Tensor of shape (Batch, Input_Dim)
            left_features: Tensor of shape (Batch, Input_Dim)
        """
        seq = torch.stack([right_features, left_features], dim=1) # (B, 2, Input_Dim)
        return self.forward(seq)
        
    def get_output_dim(self) -> int:
        return self.hidden_dim

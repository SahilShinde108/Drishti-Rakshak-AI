import torch
import torch.nn as nn
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using empty config.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class MCDropoutEstimator:
    """
    Monte Carlo Dropout for uncertainty estimation.
    """
    def __init__(self, model: nn.Module, num_passes: int = 20, dropout_rate: float = 0.3):
        self.model = model.to(DEVICE)
        self.num_passes = num_passes
        self.dropout_rate = dropout_rate
        
        # Load config if available
        config = load_config().get('trustworthy_ai', {}).get('mc_dropout', {})
        if config:
            self.num_passes = config.get('num_passes', self.num_passes)
            self.dropout_rate = config.get('dropout_rate', self.dropout_rate)

    def enable_mc_dropout(self):
        """
        Set dropout layers to train mode while keeping rest in eval.
        """
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def estimate(self, input_data: Dict[str, torch.Tensor], num_passes: int = None) -> Dict[str, Any]:
        """
        Run model num_passes times with dropout active to estimate uncertainty.
        """
        if num_passes is None:
            num_passes = self.num_passes
            
        self.enable_mc_dropout()
        
        all_probs = []
        with torch.no_grad():
            for _ in range(num_passes):
                # Assumes model returns logits or dictionary of logits
                outputs = self.model(**input_data)
                
                # Handling if model outputs tuple, dict or just tensor (dr logits)
                if isinstance(outputs, dict):
                    logits = outputs.get('dr_logits', outputs.get('logits'))
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)
                
        all_probs = np.stack(all_probs, axis=0) # Shape: (num_passes, batch_size, num_classes)
        
        mean_prediction = np.mean(all_probs, axis=0)
        prediction_variance = np.var(all_probs, axis=0)
        
        # Entropy (predictive entropy) - measure of total uncertainty
        epsilon = 1e-12
        entropy = -np.sum(mean_prediction * np.log(mean_prediction + epsilon), axis=1)
        
        # Expected entropy (aleatoric uncertainty)
        expected_entropy = np.mean(-np.sum(all_probs * np.log(all_probs + epsilon), axis=2), axis=0)
        
        # Mutual information (epistemic uncertainty)
        mutual_information = entropy - expected_entropy
        
        predicted_class = np.argmax(mean_prediction, axis=1)
        confidence = np.max(mean_prediction, axis=1)
        
        return {
            'mean_prediction': mean_prediction.tolist(),
            'prediction_variance': np.mean(prediction_variance, axis=1).tolist(), # Mean variance across classes
            'entropy': entropy.tolist(),
            'mutual_information': mutual_information.tolist(),
            'predicted_class': predicted_class.tolist(),
            'confidence': confidence.tolist()
        }

    def is_high_uncertainty(self, variance: float, threshold: float = None) -> bool:
        """
        Determine if uncertainty is considered high based on threshold.
        """
        if threshold is None:
            threshold = 0.15
        return variance > threshold

    def get_uncertainty_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate human-readable uncertainty report.
        """
        variance = np.mean(results['prediction_variance']) # Taking mean over batch if applicable, or just the value
        
        if variance > 0.2:
            level = 'high'
            human_review = True
            explanation = "High model uncertainty detected. Predictions may be unreliable. Human review is strongly recommended."
        elif variance > 0.1:
            level = 'medium'
            human_review = True
            explanation = "Moderate uncertainty detected. Model is somewhat unsure. Human review advised."
        else:
            level = 'low'
            human_review = False
            explanation = "Low uncertainty. Model is confident in its predictions."
            
        return {
            'uncertainty_level': level,
            'human_review_required': human_review,
            'explanation': explanation,
            'mean_variance': float(variance)
        }

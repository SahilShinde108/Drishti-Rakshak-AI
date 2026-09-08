import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple

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

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error using equal-width binning.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

class TemperatureScaling(nn.Module):
    """
    Post-hoc logit calibration using Temperature Scaling.
    """
    def __init__(self, max_iter: int = 100, lr: float = 0.01):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.max_iter = max_iter
        self.lr = lr
        self.to(DEVICE)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> 'TemperatureScaling':
        """
        Optimize temperature on validation set using NLL loss.
        """
        self.to(DEVICE)
        logits = logits.to(DEVICE)
        labels = labels.to(DEVICE)
        
        nll_criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.calibrate(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        logger.info(f"Temperature fitted: {self.temperature.item():.4f}")
        return self

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Divide logits by temperature.
        """
        return logits / self.temperature

    def compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
        """
        Expected Calibration Error using equal-width binning.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = predictions == labels
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.astype(float).mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return float(ece)

    def compute_reliability_diagram(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict:
        """
        Compute components for reliability diagram.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = predictions == labels
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            count_in_bin = in_bin.sum()
            
            if count_in_bin > 0:
                bin_accuracies.append(float(accuracies[in_bin].mean()))
                bin_confidences.append(float(confidences[in_bin].mean()))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
            bin_counts.append(int(count_in_bin))
                
        return {
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'bin_boundaries': bin_boundaries.tolist()
        }

    def get_calibration_report(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict:
        """
        Generate calibration report with actual ECE.
        """
        # Pre-calibration
        pre_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        pre_ece = self.compute_ece(pre_probs, labels_np)
        
        # Post-calibration
        post_logits = self.calibrate(logits.to(DEVICE))
        post_probs = torch.softmax(post_logits, dim=1).detach().cpu().numpy()
        post_ece = self.compute_ece(post_probs, labels_np)
        
        return {
            'pre_calibration_ece': pre_ece,
            'post_calibration_ece': post_ece,
            'temperature_value': self.temperature.item(),
            'pre_confidences': np.max(pre_probs, axis=1).tolist(),
            'post_confidences': np.max(post_probs, axis=1).tolist()
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Temperature model saved to {path}")
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=DEVICE))
        logger.info(f"Temperature model loaded from {path}")

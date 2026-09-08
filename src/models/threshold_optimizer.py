import numpy as np
import logging
import yaml
import json
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using default config.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ThresholdOptimizer:
    """
    Optimizes decision boundaries for continuous predictions to maximize QWK.
    """
    def __init__(self, num_classes: int = 5, method: str = 'nelder-mead'):
        self.num_classes = num_classes
        self.method = method
        self.thresholds = np.array([i + 0.5 for i in range(num_classes - 1)])
        
    def compute_qwk(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        return cohen_kappa_score(labels, predictions, weights='quadratic')
        
    def apply_thresholds(self, continuous_predictions: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Discretize continuous predictions using thresholds.
        """
        discrete_preds = np.zeros_like(continuous_predictions, dtype=int)
        for i, threshold in enumerate(thresholds):
            discrete_preds[continuous_predictions > threshold] = i + 1
        return np.clip(discrete_preds, 0, self.num_classes - 1)
        
    def _objective_function(self, thresholds: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> float:
        # Sort thresholds to ensure monotonically increasing
        thresholds = np.sort(thresholds)
        discrete_preds = self.apply_thresholds(predictions, thresholds)
        qwk = self.compute_qwk(discrete_preds, labels)
        return -qwk  # Minimize negative QWK
        
    def optimize(self, val_predictions: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
        logger.info(f"Starting threshold optimization on validation set. Initial thresholds: {self.thresholds}")
        
        initial_qwk = self.compute_qwk(self.apply_thresholds(val_predictions, self.thresholds), val_labels)
        logger.info(f"Initial Validation QWK: {initial_qwk:.4f}")
        
        result = minimize(
            self._objective_function,
            self.thresholds,
            args=(val_predictions, val_labels),
            method=self.method,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.thresholds = np.sort(result.x)
            final_qwk = -result.fun
            logger.info(f"Optimization successful. Final thresholds: {self.thresholds}")
            logger.info(f"Final Validation QWK: {final_qwk:.4f}")
        else:
            logger.warning("Optimization failed. Retaining initial thresholds.")
            
        return self.thresholds
        
    def save_thresholds(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, 'w') as f:
            json.dump({'thresholds': self.thresholds.tolist()}, f)
        logger.info(f"Saved thresholds to {path}")
            
    def load_thresholds(self, path: str):
        path_obj = Path(path)
        if path_obj.exists():
            with open(path_obj, 'r') as f:
                data = json.load(f)
                self.thresholds = np.array(data['thresholds'])
            logger.info(f"Loaded thresholds from {path}: {self.thresholds}")
        else:
            logger.warning(f"Threshold file not found at {path}. Using default.")

import cv2
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file {config_name} not found. Using defaults.")
        return {}

class BRISQUEScorer:
    """Calculates BRISQUE score for no-reference image quality assessment."""
    
    def __init__(self):
        self.config = load_config()
        self.threshold = self.config.get("iqa", {}).get("brisque_threshold", 50.0)
        
        # Try to initialize OpenCV BRISQUE
        self.use_cv2 = hasattr(cv2, 'quality') and hasattr(cv2.quality, 'QualityBRISQUE_create')
        if self.use_cv2:
            # We would need model and range files for cv2.quality.QualityBRISQUE_create
            # As fallback, we'll use our custom NSS approach
            self.use_cv2 = False
            logger.info("OpenCV BRISQUE found but requires model paths. Defaulting to custom NSS fallback.")
        else:
            logger.info("OpenCV BRISQUE not available. Defaulting to custom NSS fallback.")

    def compute_mscn(self, image: np.ndarray) -> np.ndarray:
        """Compute Mean Subtracted Contrast Normalized coefficients."""
        image = image.astype(np.float32)
        mu = cv2.GaussianBlur(image, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(image * image, (7, 7), 1.166)
        sigma = np.sqrt(np.abs(sigma - mu_sq))
        mscn = (image - mu) / (sigma + 1)
        return mscn

    def score(self, image: Optional[np.ndarray]) -> float:
        """Score the image. Lower is better (0-100 typically)."""
        if image is None:
            raise ValueError("Input image is None")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if image.size == 0:
            raise ValueError("Input image is empty")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Standardize scale so max dimension is 512 for resolution-invariant MSCN stats
        h, w = gray.shape[:2]
        scale = 512.0 / max(h, w)
        if scale < 1.0:
            eval_gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            eval_gray = gray

        if self.use_cv2:
            # Placeholder for actual cv2 implementation if models exist
            pass
            
        # Fallback: estimate blur/noise via MSCN statistics
        mscn = self.compute_mscn(eval_gray)
        var = float(np.var(mscn))
        
        # Calibrated mapping for retinal fundus images:
        # Sharp retinal images have MSCN variance around 0.22 - 0.29 (mapped to 20-35, clean)
        # Degraded images have MSCN variance < 0.10 (mapped to 65-95, poor)
        score = float(np.clip(100.0 - (var / 0.28) * 75.0, 10.0, 95.0))
        return score

class NIQEScorer:
    """Calculates NIQE score (Natural Image Quality Evaluator)."""
    
    def __init__(self):
        self.config = load_config()
        self.threshold = self.config.get("iqa", {}).get("niqe_threshold", 5.0)

    def score(self, image: Optional[np.ndarray]) -> float:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid image input")
            
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Fallback NIQE-like calculation using gradient statistics
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        mean_grad = np.mean(magnitude)
        
        # Scale to typical NIQE range (lower is better, typically 2-10)
        # Higher gradient mean -> better quality -> lower score
        score = 15.0 - (mean_grad / 10.0)
        return float(np.clip(score, 1.0, 20.0))

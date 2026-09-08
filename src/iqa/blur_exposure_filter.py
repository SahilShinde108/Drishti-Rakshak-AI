import cv2
import numpy as np
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file {config_name} not found. Using defaults.")
        return {}

def compute_laplacian_variance(image: np.ndarray) -> float:
    """Calculate Var(∇²I) for blur detection.
    
    Standardizes scale so max dimension is at most 512 to ensure
    scale-invariance between 512x512 crops and 4000x3000 raw camera scans.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    h, w = gray.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        eval_gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        eval_gray = gray
        
    return float(cv2.Laplacian(eval_gray, cv2.CV_64F).var())

def compute_shannon_entropy(image: np.ndarray) -> float:
    """Calculate Shannon entropy for exposure assessment."""
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.log2(hist + 1e-7)
    entropy = -np.sum(hist * logs)
    return float(entropy)

def compute_mean_brightness(image: np.ndarray) -> float:
    """Calculate mean brightness."""
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))
    return float(np.mean(image))

def assess_blur_exposure(image: np.ndarray, blur_threshold: float = 70.0, entropy_threshold: float = 4.2) -> dict:
    """Assess image for blur and exposure issues."""
    if image is None or image.size == 0:
        return {
            "is_blurry": True,
            "is_severe_blur": True,
            "is_underexposed": True,
            "is_severe_dark": True,
            "is_overexposed": True,
            "blur_score": 0.0,
            "entropy": 0.0,
            "brightness": 0.0,
            "error": "Empty or invalid image"
        }
        
    config = load_config()
    cfg_blur = config.get("iqa", {}).get("blur_threshold", blur_threshold)
    cfg_entropy = config.get("iqa", {}).get("entropy_threshold", entropy_threshold)
    cfg_bright_low = config.get("iqa", {}).get("brightness_low_threshold", 40.0)
    cfg_bright_high = config.get("iqa", {}).get("brightness_high_threshold", 220.0)

    blur_score = compute_laplacian_variance(image)
    entropy = compute_shannon_entropy(image)
    brightness = compute_mean_brightness(image)

    return {
        "is_blurry": blur_score < cfg_blur,
        "is_severe_blur": blur_score < 20.0,
        "is_underexposed": brightness < cfg_bright_low or (entropy < cfg_entropy and brightness < 127),
        "is_severe_dark": brightness < 20.0 or entropy < 2.0,
        "is_overexposed": brightness > cfg_bright_high or (entropy < cfg_entropy and brightness >= 127),
        "blur_score": blur_score,
        "entropy": entropy,
        "brightness": brightness
    }

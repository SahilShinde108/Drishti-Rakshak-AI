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
        return {}

class RetinalBorderCropper:
    def __init__(self):
        self.config = load_config()
        preproc_cfg = self.config.get("preprocessing", {})
        self.min_radius_ratio = preproc_cfg.get("min_radius_ratio", 0.3)
        self.padding = preproc_cfg.get("padding", 10)
        
    def detect_retina_circle(self, image: np.ndarray) -> tuple:
        """Find the circular retinal region."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Thresholding to find the bright circular region
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Validate radius
        h, w = image.shape[:2]
        if radius < self.min_radius_ratio * min(h, w):
            return None
            
        return (int(x), int(y), int(radius))

    def crop_and_resize(self, image: np.ndarray, target_size: tuple = (1024, 1024)) -> np.ndarray:
        """Crop to retinal ROI, pad if needed, and resize."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image")
            
        h, w = image.shape[:2]
        circle = self.detect_retina_circle(image)
        
        if circle:
            cx, cy, r = circle
            r += self.padding
            
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            
            cropped = image[y1:y2, x1:x2]
            logger.info(f"Cropped via retinal border detection: {(x1, y1, x2, y2)}")
        else:
            # Fallback center crop
            crop_size = min(h, w)
            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2
            cropped = image[y1:y1+crop_size, x1:x1+crop_size]
            logger.warning("Retinal border not detected. Used fallback center crop.")
            
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
        return resized

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

class SequentialPreprocessor:
    """7-step sequential preprocessing chain for retinal images."""
    def __init__(self):
        self.config = load_config()
        self.cfg = self.config.get("preprocessing", {})
        
    def extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) != 3:
            return image # Already grayscale
        return image[:, :, 1] # BGR format -> Green is index 1

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_size: tuple = (8, 8)) -> np.ndarray:
        clip = self.cfg.get("clahe_clip", clip_limit)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_size)
        return clahe.apply(image)

    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        k = self.cfg.get("blur_kernel", kernel_size)
        return cv2.GaussianBlur(image, (k, k), 0)

    def apply_highpass_filter(self, image: np.ndarray, blurred: np.ndarray) -> np.ndarray:
        # Highpass = Original - Blurred (adding 127 for visibility/stability)
        return cv2.addWeighted(image, 1.0, blurred, -1.0, 127)

    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        g = self.cfg.get("gamma", gamma)
        inv_gamma = 1.0 / g
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_laplacian_sharpening(self, image: np.ndarray) -> np.ndarray:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        # Add back to original
        sharpened = cv2.subtract(image.astype(np.float64), laplacian)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0

    def process_with_intermediates(self, image: np.ndarray) -> dict:
        """Run all 7 steps, returning intermediates."""
        if image is None:
            raise ValueError("Image is None")
            
        intermediates = {}
        
        img = self.extract_green_channel(image)
        intermediates["step1_green"] = img.copy()
        
        img = self.apply_clahe(img)
        intermediates["step2_clahe"] = img.copy()
        
        blurred = self.apply_gaussian_blur(img)
        intermediates["step3_blur"] = blurred.copy()
        
        img = self.apply_highpass_filter(img, blurred)
        intermediates["step4_highpass"] = img.copy()
        
        img = self.apply_gamma_correction(img)
        intermediates["step5_gamma"] = img.copy()
        
        img = self.apply_laplacian_sharpening(img)
        intermediates["step6_sharpened"] = img.copy()
        
        final = self.normalize(img)
        intermediates["step7_normalized"] = final
        
        logger.info("Sequential preprocessing completed with intermediates.")
        return intermediates

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run all 7 steps in sequence and return only the final result."""
        intermediates = self.process_with_intermediates(image)
        return intermediates["step7_normalized"]

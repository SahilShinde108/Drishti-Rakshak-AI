import logging
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Adjust imports based on package structure when running
try:
    from .brisque_scorer import BRISQUEScorer
    from .blur_exposure_filter import assess_blur_exposure
except ImportError:
    from brisque_scorer import BRISQUEScorer
    from blur_exposure_filter import assess_blur_exposure

logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

class IQAFeedbackGenerator:
    """Generates actionable feedback based on IQA metrics."""
    
    def __init__(self):
        self.config = load_config()
        self.brisque_scorer = BRISQUEScorer()
        
    def generate_feedback(self, iqa_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable feedback from raw IQA results."""
        feedback_en = []
        feedback_hi = []
        warnings = []
        
        is_blurry = iqa_results.get("is_blurry", False)
        is_under = iqa_results.get("is_underexposed", False)
        is_over = iqa_results.get("is_overexposed", False)
        brisque = iqa_results.get("brisque_score", 50.0)
        
        brisque_thresh = self.config.get("iqa", {}).get("brisque_threshold", 50.0)
        
        if is_blurry:
            warnings.append("High blur detected.")
            feedback_en.append("Blurry: Refocus lens")
            feedback_hi.append("धुंधला: लेंस को फिर से फोकस करें")
            
        if is_under:
            warnings.append("Underexposure detected.")
            feedback_en.append("Underexposed: Increase illumination")
            feedback_hi.append("कम रोशनी: रोशनी बढ़ाएं")
            
        if is_over:
            warnings.append("Overexposure detected.")
            feedback_en.append("Overexposed: Decrease illumination or avoid glare")
            feedback_hi.append("अधिक रोशनी: रोशनी कम करें या चमक से बचें")
            
        if brisque > brisque_thresh:
            warnings.append(f"Poor overall image quality (BRISQUE: {brisque:.2f}).")
            feedback_en.append("Poor quality: Please recapture image carefully")
            feedback_hi.append("खराब गुणवत्ता: कृपया छवि को सावधानीपूर्वक दोबारा लें")

        num_issues = sum([is_blurry, is_under, is_over, brisque > brisque_thresh])
        is_severe = iqa_results.get("is_severe_blur", False) or iqa_results.get("is_severe_dark", False)
        
        if is_severe or num_issues >= 3:
            quality_tier = "ungradable"
            is_gradable = False
        elif num_issues == 2:
            quality_tier = "poor"
            is_gradable = True
        elif num_issues == 1:
            quality_tier = "acceptable"
            is_gradable = True
        else:
            quality_tier = "good"
            is_gradable = True
            
        recapture = not is_gradable

        return {
            "is_gradable": is_gradable,
            "quality_tier": quality_tier,
            "warnings": warnings,
            "feedback_en": feedback_en,
            "feedback_hi": feedback_hi,
            "recapture_required": recapture
        }

def run_full_iqa(image: np.ndarray) -> Dict[str, Any]:
    """Orchestrates BRISQUE + blur/exposure + feedback into single structured output."""
    try:
        be_results = assess_blur_exposure(image)
        scorer = BRISQUEScorer()
        brisque_score = scorer.score(image)
        
        combined = {**be_results, "brisque_score": brisque_score}
        
        fg = IQAFeedbackGenerator()
        feedback = fg.generate_feedback(combined)
        
        blur_score = float(combined.get("blur_score", 0.0))
        entropy = float(combined.get("entropy", 0.0))
        brisque = float(combined.get("brisque_score", 50.0))
        
        # Continuous composite quality score [0.05, 0.99]
        blur_norm = min(1.0, max(0.0, blur_score / 120.0))
        entropy_norm = min(1.0, max(0.0, (entropy - 2.0) / 4.5))
        brisque_norm = min(1.0, max(0.0, (100.0 - brisque) / 80.0))
        quality_score = float(np.clip(0.40 * blur_norm + 0.30 * entropy_norm + 0.30 * brisque_norm, 0.05, 0.99))
        
        # Combine all into one dictionary
        return {
            "is_gradable": feedback["is_gradable"],
            "quality_tier": feedback["quality_tier"],
            "quality_score": round(quality_score, 2),
            "blur_score": blur_score,
            "entropy": entropy,
            "brisque_score": brisque,
            "warnings": feedback["warnings"],
            "feedback_en": feedback["feedback_en"],
            "feedback_hi": feedback["feedback_hi"],
            "recapture_required": feedback["recapture_required"]
        }
    except Exception as e:
        logger.error(f"Error in IQA pipeline: {e}")
        return {
            "is_gradable": False,
            "quality_tier": "ungradable",
            "quality_score": 0.05,
            "blur_score": 0.0,
            "entropy": 0.0,
            "brisque_score": 95.0,
            "warnings": [f"Processing error: {str(e)}"],
            "feedback_en": ["Processing error: Please recapture scan"],
            "feedback_hi": ["प्रसंस्करण त्रुटि: कृपया पुनः स्कैन करें"],
            "recapture_required": True
        }

if __name__ == "__main__":
    # Test stub
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    print(run_full_iqa(test_img))

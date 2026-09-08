import numpy as np
import logging
from typing import Dict, List
import matplotlib.pyplot as plt
import cv2
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LesionAlignmentChecker:
    """
    Quantitative validation of Grad-CAM / model attention overlap with actual lesion masks.
    """
    def __init__(self, threshold: float = 0.5):
        self.default_threshold = threshold

    def compute_iou(self, heatmap: np.ndarray, mask: np.ndarray, threshold: float = None) -> float:
        """
        Compute Intersection over Union (IoU).
        """
        if threshold is None:
            threshold = self.default_threshold
            
        if heatmap.shape != mask.shape:
            mask = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        heatmap_binary = (heatmap >= threshold).astype(bool)
        mask_binary = (mask > 0).astype(bool)
        
        intersection = np.logical_and(heatmap_binary, mask_binary).sum()
        union = np.logical_or(heatmap_binary, mask_binary).sum()
        
        if union == 0:
            return 0.0
            
        return float(intersection / union)

    def compute_dice(self, heatmap: np.ndarray, mask: np.ndarray, threshold: float = None) -> float:
        """
        Compute Dice coefficient.
        """
        if threshold is None:
            threshold = self.default_threshold
            
        if heatmap.shape != mask.shape:
            mask = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        heatmap_binary = (heatmap >= threshold).astype(bool)
        mask_binary = (mask > 0).astype(bool)
        
        intersection = np.logical_and(heatmap_binary, mask_binary).sum()
        sum_areas = heatmap_binary.sum() + mask_binary.sum()
        
        if sum_areas == 0:
            return 0.0
            
        return float(2.0 * intersection / sum_areas)

    def compute_alignment(self, heatmap: np.ndarray, lesion_masks: Dict[str, np.ndarray], threshold: float = None) -> Dict:
        """
        Compute alignment metrics for multiple lesion types and overall.
        """
        if threshold is None:
            threshold = self.default_threshold
            
        per_lesion_iou = {}
        per_lesion_dice = {}
        
        union_mask = np.zeros_like(heatmap, dtype=bool)
        
        for lesion_type, mask in lesion_masks.items():
            if mask is not None and mask.sum() > 0:
                iou = self.compute_iou(heatmap, mask, threshold)
                dice = self.compute_dice(heatmap, mask, threshold)
                
                per_lesion_iou[lesion_type] = iou
                per_lesion_dice[lesion_type] = dice
                
                if mask.shape != union_mask.shape:
                    mask_resized = cv2.resize(mask, (union_mask.shape[1], union_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask
                    
                union_mask = np.logical_or(union_mask, mask_resized > 0)
            else:
                per_lesion_iou[lesion_type] = 0.0
                per_lesion_dice[lesion_type] = 0.0
                
        # Overall metrics
        overall_iou = self.compute_iou(heatmap, union_mask.astype(float), threshold)
        overall_dice = self.compute_dice(heatmap, union_mask.astype(float), threshold)
        
        heatmap_area = int((heatmap >= threshold).sum())
        lesion_area = int(union_mask.sum())
        
        logger.info(f"Lesion Alignment - Overall IoU: {overall_iou:.4f}, Dice: {overall_dice:.4f}")
        
        return {
            'iou': overall_iou,
            'dice': overall_dice,
            'per_lesion_iou': per_lesion_iou,
            'per_lesion_dice': per_lesion_dice,
            'threshold': threshold,
            'heatmap_area': heatmap_area,
            'lesion_area': lesion_area
        }

    def compute_at_multiple_thresholds(self, heatmap: np.ndarray, mask: np.ndarray, 
                                      thresholds: List[float] = None) -> Dict:
        """
        Find optimal threshold for alignment.
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            
        results = {}
        best_iou = -1
        best_thresh = thresholds[0]
        
        for thresh in thresholds:
            iou = self.compute_iou(heatmap, mask, thresh)
            dice = self.compute_dice(heatmap, mask, thresh)
            
            results[str(thresh)] = {'iou': iou, 'dice': dice}
            
            if iou > best_iou:
                best_iou = iou
                best_thresh = thresh
                
        results['optimal_threshold'] = best_thresh
        results['best_iou'] = best_iou
        return results

    def visualize_alignment(self, image: np.ndarray, heatmap: np.ndarray, mask: np.ndarray, 
                           save_path: str = None) -> np.ndarray:
        """
        Create overlay visualization of image, heatmap, and true mask.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(image)
        axes[1].imshow(heatmap, alpha=0.5, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(image)
        axes[2].imshow(mask, alpha=0.5, cmap='Reds')
        axes[2].set_title("True Lesion Mask")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return vis_image

def load_ground_truth_masks(img_id: str, base_dir: str = None) -> Dict[str, np.ndarray]:
    """
    Search and load ground truth pixel masks (MA, HE, EX, SE, OD) from IDRiD dataset.
    """
    from pathlib import Path
    base_path = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
    masks = {}
    lesion_keys = [('MA', '_MA.tif'), ('HE', '_HE.tif'), ('EX', '_EX.tif'), ('SE', '_SE.tif'), ('OD', '_OD.tif')]
    
    clean_id = img_id.replace('_preprocessed', '').replace('_orig', '')
    
    for key, suffix in lesion_keys:
        matches = list(base_path.glob(f"data/Raw/**/{clean_id}{suffix}"))
        if matches:
            img = cv2.imread(str(matches[0]), cv2.IMREAD_GRAYSCALE)
            masks[key] = img
        else:
            masks[key] = None
    return masks

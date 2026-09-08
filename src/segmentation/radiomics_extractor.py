import numpy as np
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Optional
import scipy.stats as stats
try:
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import regionprops
except ImportError:
    graycomatrix = graycoprops = regionprops = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadiomicsExtractor:
    def __init__(self, use_pyradiomics: bool = True):
        self.use_pyradiomics = use_pyradiomics
        
    def extract_intensity_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        features = {}
        if mask.sum() == 0:
            return {'mean': 0.0, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0, 'entropy': 0.0, 'energy': 0.0}
            
        roi_pixels = image[mask > 0]
        if len(roi_pixels.shape) > 1:
            roi_pixels = roi_pixels[:, 0] # Use only one channel for simplicity
            
        features['mean'] = float(np.mean(roi_pixels))
        features['std'] = float(np.std(roi_pixels))
        features['skewness'] = float(stats.skew(roi_pixels) if len(roi_pixels) > 0 else 0)
        features['kurtosis'] = float(stats.kurtosis(roi_pixels) if len(roi_pixels) > 0 else 0)
        
        hist, _ = np.histogram(roi_pixels, bins=256, density=True)
        hist = hist[hist > 0]
        features['entropy'] = float(-np.sum(hist * np.log2(hist)))
        features['energy'] = float(np.sum(hist ** 2))
        return features

    def extract_texture_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        features = {'contrast': 0.0, 'correlation': 0.0, 'energy': 0.0, 'homogeneity': 0.0}
        if mask.sum() == 0 or graycomatrix is None:
            return features
            
        img_gray = (image[:,:,0] if len(image.shape)==3 else image).astype(np.uint8)
        # Apply mask simply
        img_gray = img_gray * mask.astype(np.uint8)
        
        glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        features['contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
        features['correlation'] = float(graycoprops(glcm, 'correlation')[0, 0])
        features['energy'] = float(graycoprops(glcm, 'energy')[0, 0])
        features['homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
        return features

    def extract_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        features = {'area': 0.0, 'perimeter': 0.0, 'circularity': 0.0, 'eccentricity': 0.0, 'solidity': 0.0}
        if mask.sum() == 0 or regionprops is None:
            return features
            
        props = regionprops(mask.astype(int))
        if props:
            p = props[0]
            features['area'] = float(p.area)
            features['perimeter'] = float(p.perimeter)
            if p.perimeter > 0:
                features['circularity'] = float((4 * np.pi * p.area) / (p.perimeter ** 2))
            features['eccentricity'] = float(p.eccentricity)
            features['solidity'] = float(p.solidity)
            
        return features

    def extract_all(self, image: np.ndarray, vessel_mask: np.ndarray, disc_mask: np.ndarray, cup_mask: np.ndarray, lesion_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        all_features = {}
        all_features.update({f"vessel_intensity_{k}": v for k, v in self.extract_intensity_features(image, vessel_mask).items()})
        all_features.update({f"disc_shape_{k}": v for k, v in self.extract_shape_features(disc_mask).items()})
        all_features.update({f"disc_texture_{k}": v for k, v in self.extract_texture_features(image, disc_mask).items()})
        
        for name, mask in lesion_masks.items():
            all_features.update({f"lesion_{name}_shape_{k}": v for k, v in self.extract_shape_features(mask).items()})
            
        return all_features

    def get_feature_names(self) -> List[str]:
        return [
            'vessel_intensity_mean', 'vessel_intensity_std', 'vessel_intensity_skewness',
            'vessel_intensity_kurtosis', 'vessel_intensity_entropy', 'vessel_intensity_energy',
            'disc_shape_area', 'disc_shape_perimeter', 'disc_shape_circularity',
            'disc_shape_eccentricity', 'disc_shape_solidity',
            'disc_texture_contrast', 'disc_texture_correlation', 'disc_texture_energy', 'disc_texture_homogeneity'
        ]

    def to_feature_vector(self, features_dict: Dict[str, float]) -> np.ndarray:
        vec = []
        for name in self.get_feature_names():
            vec.append(features_dict.get(name, np.nan))
        return np.array(vec)

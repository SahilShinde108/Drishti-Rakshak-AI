import torch
import torch.nn as nn
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Callable
import matplotlib.pyplot as plt
import cv2
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: str = 'auto'):
        self.model = model.to(DEVICE)
        self.model.eval()
        self.target_layer_name = target_layer
        self.target_layer = self._find_target_layer()
        self.gradients = None
        self.activations = None
        
        if self.target_layer:
            self.target_layer.register_forward_hook(self.save_activation)
            self.target_layer.register_full_backward_hook(self.save_gradient)
        else:
            logger.warning("Could not find suitable target layer for Grad-CAM++ (e.g., pure transformer).")

    def _find_target_layer(self):
        if self.target_layer_name != 'auto':
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    return module
            return None
            
        # Priority 1: Model's visual_encoder last conv layer
        if hasattr(self.model, 'visual_encoder') and hasattr(self.model.visual_encoder, 'get_last_conv_layer'):
            return self.model.visual_encoder.get_last_conv_layer()
            
        # Priority 2: Last conv layer in entire module hierarchy
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image: torch.Tensor, target_class: int = None, clinical: torch.Tensor = None) -> np.ndarray:
        if self.target_layer is None:
            logger.warning("Grad-CAM++ not applicable (no target layer).")
            return np.zeros((image.shape[-2], image.shape[-1]))
            
        self.gradients = None
        self.activations = None
        self.model.eval()
        self.model.zero_grad()
        
        image = image.to(DEVICE)
        image.requires_grad = True
        
        if clinical is not None:
            clinical = clinical.to(DEVICE)
            output = self.model(image=image, clinical=clinical)
        else:
            output = self.model(image)
            
        if isinstance(output, dict):
            logits = output.get('dr_logits', output.get('logits'))
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
            
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        target = logits[0, target_class]
        target.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros((image.shape[-2], image.shape[-1]))
            
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Grad-CAM++ formulation (simplified approximation)
        weights = np.maximum(gradients, 0).mean(axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (image.shape[-1], image.shape[-2]))
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        return cam

    def overlay_on_image(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        heatmap = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Convert BGR (from OpenCV applyColorMap) to RGB to match RGB input image
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        overlay = cv2.addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)
        return overlay

class LIMEExplainer:
    def __init__(self, model: nn.Module, num_samples: int = 1000, num_features: int = 10):
        self.model = model
        self.num_samples = num_samples
        self.num_features = num_features
        try:
            from lime import lime_image
            self.explainer = lime_image.LimeImageExplainer()
            self.available = True
        except ImportError:
            logger.warning("lime package not found. LIMEExplainer will be disabled.")
            self.available = False

    def explain(self, image: np.ndarray, predict_fn: Callable) -> Dict[str, Any]:
        if not self.available:
            return {'error': 'LIME not available'}
            
        try:
            explanation = self.explainer.explain_instance(
                image, 
                predict_fn, 
                top_labels=1, 
                hide_color=0, 
                num_samples=self.num_samples
            )
            
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True, 
                num_features=self.num_features, 
                hide_rest=False
            )
            
            return {
                'segments': explanation.segments,
                'explanation_image': temp,
                'mask': mask
            }
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'error': str(e)}

class SHAPExplainer:
    def __init__(self, model_predict_fn: Callable, feature_names: List[str], max_samples: int = 100):
        self.predict_fn = model_predict_fn
        self.feature_names = feature_names
        self.max_samples = max_samples
        try:
            import shap
            self.shap = shap
            self.available = True
        except ImportError:
            logger.warning("shap package not found. SHAPExplainer disabled.")
            self.available = False

    def explain_tabular(self, features: np.ndarray) -> Dict[str, Any]:
        if not self.available:
            return {'error': 'SHAP not available'}
            
        try:
            # Using KernelExplainer as fallback for arbitrary models
            background = np.zeros((1, features.shape[1])) # Dummy background
            explainer = self.shap.KernelExplainer(self.predict_fn, background)
            shap_values = explainer.shap_values(features, nsamples=self.max_samples)
            
            return {
                'shap_values': shap_values,
                'feature_names': self.feature_names,
                'feature_values': features[0].tolist(),
                'base_value': explainer.expected_value
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}

    def generate_bar_chart(self, shap_values, feature_names, save_path: str) -> str:
        if not self.available:
            return ""
            
        try:
            plt.figure(figsize=(10, 6))
            # Handle list of shap values (multi-class)
            vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            if len(vals.shape) > 1:
                vals = vals[0]
                
            y_pos = np.arange(len(feature_names))
            plt.barh(y_pos, vals)
            plt.yticks(y_pos, feature_names)
            plt.xlabel('SHAP Value (Impact on model output)')
            plt.title('Clinical Feature Importance')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return save_path
        except Exception as e:
            logger.error(f"SHAP chart generation failed: {e}")
            return ""

class ExplainabilityStack:
    def __init__(self, model: nn.Module, config: dict = None):
        self.model = model
        self.config = config or {}
        
        self.gradcam = GradCAMPlusPlus(model)
        self.lime = LIMEExplainer(model)
        
    def explain_image(self, image: torch.Tensor, original_image: np.ndarray, predict_fn: Callable, target_class: int = None) -> Dict[str, Any]:
        results = {}
        
        try:
            heatmap = self.gradcam.generate(image, target_class)
            overlay = self.gradcam.overlay_on_image(original_image, heatmap)
            results['gradcam'] = {
                'heatmap': heatmap,
                'overlay': overlay
            }
        except Exception as e:
            logger.error(f"Grad-CAM failed: {e}")
            
        try:
            lime_res = self.lime.explain(original_image, predict_fn)
            results['lime'] = lime_res
        except Exception as e:
            logger.error(f"LIME failed: {e}")
            
        return results
        
    def explain_clinical(self, features: np.ndarray, feature_names: List[str], predict_fn: Callable) -> Dict[str, Any]:
        shap_explainer = SHAPExplainer(predict_fn, feature_names)
        return shap_explainer.explain_tabular(features)

    def explain_all(self, image: torch.Tensor, original_image: np.ndarray, image_predict_fn: Callable, 
                    features: np.ndarray, feature_names: List[str], clinical_predict_fn: Callable, 
                    target_class: int = None) -> Dict[str, Any]:
        results = {}
        results['image'] = self.explain_image(image, original_image, image_predict_fn, target_class)
        results['clinical'] = self.explain_clinical(features, feature_names, clinical_predict_fn)
        return results

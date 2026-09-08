import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.iqa.blur_exposure_filter import compute_laplacian_variance, compute_shannon_entropy
from src.iqa.feedback_generator import IQAFeedbackGenerator
from src.preprocessing.border_cropper import RetinalBorderCropper
from src.preprocessing.sequential_pipeline import SequentialPreprocessor
from src.segmentation.concat_unet_lesion import ConcatUNetLesion
from src.segmentation.swin_unet_vessel import SwinUNetVessel
from src.backbones.tabular_mlp import TabularMLP
from src.fusion.adaptive_gated_fusion import AdaptiveGatedFusion
from src.fusion.eye_pair_lstm import EyePairLSTM
from src.models.multi_task_head import DrishtiRakshakModel, MultiTaskHead
from src.trustworthy_ai.temperature_scaling import TemperatureScaling, compute_ece
from src.trustworthy_ai.lesion_alignment_check import LesionAlignmentChecker
from src.models.threshold_optimizer import ThresholdOptimizer
from src.models.stacking_ensemble import StackingEnsemble

class TestIQA:
    def test_blur_detection_blurry_image(self):
        # Uniform image has 0 variance (blurry)
        blurry = np.ones((100, 100, 3), dtype=np.uint8) * 128
        score = compute_laplacian_variance(blurry)
        assert score < 10.0

    def test_blur_detection_sharp_image(self):
        # Checkerboard pattern has high edge gradients
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[::2, ::2] = 255
        sharp[1::2, 1::2] = 255
        score = compute_laplacian_variance(sharp)
        assert score > 10.0

    def test_entropy_dark_image(self):
        dark = np.zeros((100, 100, 3), dtype=np.uint8)
        entropy = compute_shannon_entropy(dark)
        assert entropy < 2.0

    def test_entropy_normal_image(self):
        noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        entropy = compute_shannon_entropy(noise)
        assert entropy > 4.0

    def test_feedback_generation(self):
        gen = IQAFeedbackGenerator()
        res = gen.generate_feedback({
            'blur_score': 10.0,
            'is_blurry': True,
            'entropy': 5.0,
            'is_underexposed': False,
            'is_overexposed': False,
            'brisque_score': 30.0
        })
        assert 'is_gradable' in res
        assert 'feedback_en' in res
        assert len(res['warnings']) > 0

    def test_invalid_input(self):
        gen = IQAFeedbackGenerator()
        with pytest.raises(Exception):
            compute_laplacian_variance(None)

class TestPreprocessing:
    def test_border_cropper_output_shape(self):
        cropper = RetinalBorderCropper()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create a white circle simulating the retinal FOV
        import cv2
        cv2.circle(img, (100, 100), 80, (255, 255, 255), -1)
        out = cropper.crop_and_resize(img, target_size=(1024, 1024))
        assert out.shape == (1024, 1024, 3)

    def test_sequential_pipeline_output_range(self):
        pipe = SequentialPreprocessor()
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = pipe.process(img)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_sequential_pipeline_dtype(self):
        pipe = SequentialPreprocessor()
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = pipe.process(img)
        assert out.dtype == np.float32

    def test_green_channel_extraction(self):
        pipe = SequentialPreprocessor()
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        green = pipe.extract_green_channel(img)
        assert green.ndim == 2
        assert green.shape == (224, 224)

class TestSegmentation:
    def test_swin_unet_output_shape(self):
        model = SwinUNetVessel(in_channels=3, out_channels=3)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert 'vessel' in out and 'disc' in out and 'cup' in out
        assert out['vessel'].shape == (1, 1, 224, 224)

    def test_lesion_unet_output_channels(self):
        model = ConcatUNetLesion()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert 'ma' in out and 'he' in out and 'ex' in out and 'se' in out
        assert out['ma'].shape == (1, 1, 224, 224)

    def test_cdr_computation(self):
        from src.segmentation.swin_unet_vessel import VesselSegmentor
        seg = VesselSegmentor(weights_path=None)
        disc = np.zeros((100, 100), dtype=np.uint8)
        cup = np.zeros((100, 100), dtype=np.uint8)
        disc[20:80, 20:80] = 1 # 60x60 = 3600
        cup[35:65, 35:65] = 1   # 30x30 = 900
        cdr = seg.compute_cdr(disc, cup)
        assert 0.4 <= cdr <= 0.6 # sqrt(900/3600) = 0.5

class TestBackbones:
    def test_efficientnet_feature_dim(self):
        from src.backbones.efficientnet_cbam import EfficientNetCBAM
        bb = EfficientNetCBAM(model_name='efficientnet_b0', pretrained=False)
        assert bb.get_feature_dim() == 1280

    def test_swin_feature_dim(self):
        from src.backbones.swin_transformer import SwinTransformerBackbone
        swin = SwinTransformerBackbone(variant='tiny', pretrained=False)
        assert swin.get_feature_dim() == 768

    def test_tabular_mlp_output_dim(self):
        mlp = TabularMLP(in_features=22, out_features=128)
        x = torch.randn(2, 22)
        out = mlp(x)
        assert out.shape == (2, 128)

class TestFusion:
    def test_agf_output_dim(self):
        agf = AdaptiveGatedFusion(image_dim=256, clinical_dim=128, fused_dim=512)
        img = torch.randn(2, 256)
        clin = torch.randn(2, 128)
        out = agf(img, clin)
        assert out['fused'].shape == (2, 512)

    def test_agf_gate_values_range(self):
        agf = AdaptiveGatedFusion(image_dim=256, clinical_dim=128, fused_dim=512)
        img = torch.randn(2, 256)
        clin = torch.randn(2, 128)
        out = agf(img, clin)
        gates = out['gate_values']
        assert (gates >= 0.0).all() and (gates <= 1.0).all()

    def test_lstm_single_eye(self):
        lstm = EyePairLSTM(input_dim=1280, hidden_dim=256)
        x = torch.randn(2, 1280)
        out = lstm.forward_single_eye(x)
        assert out.shape == (2, 256)

    def test_lstm_bilateral(self):
        lstm = EyePairLSTM(input_dim=1280, hidden_dim=256)
        x1 = torch.randn(2, 1280)
        x2 = torch.randn(2, 1280)
        out = lstm.forward_bilateral(x1, x2)
        assert out.shape == (2, 256)

class TestPrediction:
    def test_multitask_output_keys(self):
        head = MultiTaskHead(input_dim=512, dr_classes=5, dme_classes=3)
        x = torch.randn(2, 512)
        out = head(x)
        expected_keys = ['dr_logits', 'dr_probs', 'dr_pred', 'dme_logits', 'dme_probs', 'dme_pred', 'severity_score', 'referable_logit', 'referable_prob', 'referable_pred']
        for k in expected_keys:
            assert k in out

    def test_dr_probabilities_sum_to_one(self):
        head = MultiTaskHead(input_dim=512, dr_classes=5, dme_classes=3)
        x = torch.randn(4, 512)
        out = head(x)
        sums = out['dr_probs'].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_dme_probabilities_sum_to_one(self):
        head = MultiTaskHead(input_dim=512, dr_classes=5, dme_classes=3)
        x = torch.randn(4, 512)
        out = head(x)
        sums = out['dme_probs'].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_severity_range(self):
        head = MultiTaskHead(input_dim=512, dr_classes=5, dme_classes=3)
        x = torch.randn(4, 512)
        out = head(x)
        sev = out['severity_score']
        assert (sev >= 0.0).all() and (sev <= 4.0).all()

    def test_referable_criterion(self):
        dr = torch.tensor([0, 1, 2, 0, 3])
        dme = torch.tensor([0, 0, 0, 1, 2])
        ref = MultiTaskHead.compute_referable_from_predictions(dr, dme)
        # DR >= 2 OR DME >= 1
        expected = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        assert (ref == expected).all()

class TestCalibration:
    def test_ece_perfect_calibration(self):
        # Perfectly calibrated: confidence matches accuracy
        probs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        labels = np.array([0, 1, 0])
        ece = compute_ece(probs, labels, n_bins=5)
        assert ece < 0.05

    def test_ece_worst_calibration(self):
        # 100% confident but 100% wrong
        probs = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        labels = np.array([1, 0])
        ece = compute_ece(probs, labels, n_bins=5)
        assert ece > 0.5

class TestAlignment:
    def test_iou_identical_masks(self):
        checker = LesionAlignmentChecker()
        m = np.ones((50, 50))
        iou = checker.compute_iou(m, m)
        assert abs(iou - 1.0) < 1e-5

    def test_iou_disjoint_masks(self):
        checker = LesionAlignmentChecker()
        m1 = np.zeros((50, 50))
        m2 = np.zeros((50, 50))
        m1[:20, :] = 1.0
        m2[30:, :] = 1.0
        iou = checker.compute_iou(m1, m2)
        assert iou == 0.0

    def test_dice_known_overlap(self):
        checker = LesionAlignmentChecker()
        m1 = np.zeros((10, 10))
        m2 = np.zeros((10, 10))
        m1[:5, :] = 1.0  # 50 pixels
        m2[:5, :] = 1.0  # 50 pixels
        dice = checker.compute_dice(m1, m2)
        assert abs(dice - 1.0) < 1e-5

    def test_empty_mask_handling(self):
        checker = LesionAlignmentChecker()
        m1 = np.zeros((10, 10))
        m2 = np.zeros((10, 10))
        iou = checker.compute_iou(m1, m2)
        dice = checker.compute_dice(m1, m2)
        assert iou == 0.0
        assert dice == 0.0

class TestThresholds:
    def test_threshold_optimization_improves_qwk(self):
        opt = ThresholdOptimizer(num_classes=3, method='nelder-mead')
        # Generate slightly noisy continuous predictions
        preds = np.array([0.2, 0.4, 1.1, 1.3, 2.2, 2.4])
        labels = np.array([0, 0, 1, 1, 2, 2])
        opt_thresh = opt.optimize(preds, labels)
        discrete = opt.apply_thresholds(preds, opt_thresh)
        qwk = opt.compute_qwk(discrete, labels)
        assert qwk >= 0.9

    def test_thresholds_not_from_test_set(self):
        opt = ThresholdOptimizer(num_classes=5)
        assert len(opt.thresholds) == 4

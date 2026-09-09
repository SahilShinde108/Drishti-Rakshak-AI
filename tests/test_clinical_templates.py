import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.reporting.clinical_templates import ClinicalTemplateEngine

def test_all_dr_stages_have_templates():
    """Verify templates exist for all 5 ICDR DR stages (0 to 4)."""
    for stage in range(5):
        content = ClinicalTemplateEngine.get_patient_copy_content(
            dr_stage=stage,
            dme_grade=0,
            patient_id=f"TEST_STG_{stage}",
            eye_laterality="OD (Right Eye)"
        )
        assert content["dr_stage"] == stage
        assert len(content["stage_name_en"]) > 0
        assert len(content["stage_name_hi"]) > 0
        assert len(content["summary_en"]) > 0
        assert len(content["summary_hi"]) > 0
        assert len(content["timeline_en"]) > 0
        assert len(content["timeline_hi"]) > 0
        assert len(content["recommended_actions_en"]) >= 3
        assert len(content["recommended_actions_hi"]) >= 3

def test_csme_urgent_alert():
    """Verify that DME Grade 2 (CSME) triggers urgent alert in both languages."""
    content = ClinicalTemplateEngine.get_patient_copy_content(
        dr_stage=1,
        dme_grade=2,
        patient_id="TEST_CSME",
        eye_laterality="OS (Left Eye)"
    )
    assert content["dme_grade"] == 2
    assert "csme" in content["dme_name_en"].lower()
    assert "anti-vegf" in content["summary_en"].lower()
    assert "anti-vegf" in content["summary_hi"].lower() or "इंजेक्शन" in content["summary_hi"]
    assert content["urgency_en"] == "EMERGENCY (Within 1-2 Weeks)"
    assert content["badge_text"] == "EMERGENCY REFERRAL"

def test_glaucoma_suspect_flag():
    """Verify enlarged CDR (> 0.60) triggers glaucoma suspect warning."""
    # Normal CDR
    content_normal = ClinicalTemplateEngine.get_patient_copy_content(
        dr_stage=0,
        dme_grade=0,
        cdr_value=0.35
    )
    assert not content_normal["glaucoma_suspect"]
    assert "GLAUCOMA SUSPECT" not in content_normal["summary_en"]

    # Suspect CDR
    content_suspect = ClinicalTemplateEngine.get_patient_copy_content(
        dr_stage=0,
        dme_grade=0,
        cdr_value=0.68
    )
    assert content_suspect["glaucoma_suspect"]
    assert "GLAUCOMA SUSPECT" in content_suspect["summary_en"]
    assert "काला मोतिया" in content_suspect["summary_hi"]

def test_hypertensive_narrowing_flag():
    """Verify attenuated AVR (< 0.60) triggers hypertensive changes warning."""
    # Normal AVR
    content_normal = ClinicalTemplateEngine.get_patient_copy_content(
        dr_stage=0,
        dme_grade=0,
        avr_value=0.67
    )
    assert not content_normal["hypertensive_suspect"]

    # Attenuated AVR
    content_hypertensive = ClinicalTemplateEngine.get_patient_copy_content(
        dr_stage=0,
        dme_grade=0,
        avr_value=0.55
    )
    assert content_hypertensive["hypertensive_suspect"]
    assert "HYPERTENSIVE" in content_hypertensive["summary_en"]
    assert "रक्तचाप" in content_hypertensive["summary_hi"]

def test_edge_case_stage_clamping():
    """Verify out-of-range inputs are safely clamped."""
    content_neg = ClinicalTemplateEngine.get_patient_copy_content(dr_stage=-5, dme_grade=-1)
    assert content_neg["dr_stage"] == 0
    assert content_neg["dme_grade"] == 0

    content_overflow = ClinicalTemplateEngine.get_patient_copy_content(dr_stage=99, dme_grade=99)
    assert content_overflow["dr_stage"] == 4
    assert content_overflow["dme_grade"] == 2

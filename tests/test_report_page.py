import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.reporting.clinical_templates import ClinicalTemplateEngine
from src.reporting.pdf_report_generator import ClinicalPDFReport

def test_report_page_template_integration():
    """Verify that report page dependencies produce valid output."""
    sample_res = {
        'patient_id': 'TEST_PAGE_001',
        'eye_laterality': 'OD (Right Eye)',
        'dr_prediction': {'stage': 2, 'stage_name': 'Moderate NPDR', 'confidence': 0.88, 'probabilities': [0.02, 0.08, 0.88, 0.01, 0.01]},
        'dme_prediction': {'grade': 0, 'grade_name': 'No DME'},
        'severity_score': 2.15,
        'referable': True,
        'referral_status': 'REFERRAL REQUIRED',
        'biomarkers': {'avr': 0.52, 'cdr': 0.68, 'vessel_density': 0.082, 'exudate_fovea_distance': 2400.0},
        'uncertainty': {'variance': 0.0012, 'level': 'low'},
        'calibration': {'confidence_calibrated': 0.865, 'ece': 0.018},
        'xai': {'lesion_alignment': {'dice': 0.824, 'iou': 0.741}},
        'clinical_data_provided': False,
        'clinical_features': {}
    }
    
    # 1. Test template retrieval
    template_content = ClinicalTemplateEngine.get_patient_copy_content(
        dr_stage=sample_res['dr_prediction']['stage'],
        dme_grade=sample_res['dme_prediction']['grade'],
        cdr_value=sample_res['biomarkers']['cdr'],
        avr_value=sample_res['biomarkers']['avr'],
        patient_id=sample_res['patient_id'],
        eye_laterality=sample_res['eye_laterality']
    )
    
    assert template_content['dr_stage'] == 2
    assert "Within 3 Months" in template_content['timeline_en']
    assert template_content['glaucoma_suspect'] is True   # CDR = 0.68 > 0.60
    assert template_content['hypertensive_suspect'] is True # AVR = 0.52 < 0.60
    
    # 2. Test PDF regeneration on the fly
    reports_dir = BASE_DIR / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_pdf_path = str(reports_dir / "TEST_PAGE_REGEN_report.pdf")
    
    pdf_gen = ClinicalPDFReport()
    generated_path = pdf_gen.generate(sample_res, test_pdf_path)
    assert Path(generated_path).exists()
    assert Path(generated_path).stat().st_size > 10000

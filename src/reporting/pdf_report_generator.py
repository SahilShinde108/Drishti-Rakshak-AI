import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
import yaml
from pathlib import Path

# Try importing reportlab, fallback to basic text if not available
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ClinicalPDFReport:
    def __init__(self, config: dict = None):
        self.config = config or load_config().get('reporting', {})
        self.institution_name = self.config.get('institution_name', 'Drishti-Rakshak AI Screening')
        
    def generate(self, patient_data: Dict[str, Any], save_path: str) -> str:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. Falling back to JSON report.")
            json_path = save_path.replace('.pdf', '.json')
            with open(json_path, 'w') as f:
                json.dump(patient_data, f, indent=4)
            return json_path
            
        doc = SimpleDocTemplate(save_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Center', alignment=1))
        styles.add(ParagraphStyle(name='Right', alignment=2))
        
        story = []
        
        # Header
        story.append(Paragraph(f"<b>{self.institution_name}</b>", styles['Title']))
        story.append(Paragraph("AI-Assisted Diabetic Retinopathy Screening Report", styles['Heading2']))
        story.append(Spacer(1, 0.25*inch))
        
        # Clinical Screening Information
        story.append(Paragraph("<b>1. Clinical Screening Information</b>", styles['Heading3']))
        mode = patient_data.get('screening_mode', 'Autonomous Image-Only Screening')
        clin_prov = patient_data.get('clinical_data_provided', False)
        clin_data = patient_data.get('clinical_inputs', {})
        
        if clin_prov:
            age_str = f"{clin_data.get('age', 'N/A')} yrs"
            dur_str = f"{clin_data.get('diabetes_duration', 'N/A')} yrs"
            hba1c_str = f"{clin_data.get('hba1c', 'N/A')}%"
            bp_str = f"{clin_data.get('systolic_bp', 'N/A')}/{clin_data.get('diastolic_bp', 'N/A')} mmHg"
            p_info = [
                ["Patient ID:", str(patient_data.get('patient_id', 'N/A')), "Date / Time:", str(patient_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")))],
                ["Screening Mode:", mode, "Age / Duration:", f"{age_str} / {dur_str}"],
                ["HbA1c Level:", hba1c_str, "Blood Pressure:", bp_str]
            ]
        else:
            p_info = [
                ["Patient ID:", str(patient_data.get('patient_id', 'N/A')), "Date / Time:", str(patient_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")))],
                ["Screening Mode:", mode, "Clinical EMR:", "Not Provided (Autonomous Image-Only Mode)"]
            ]
        t = Table(p_info, colWidths=[1.3*inch, 2.2*inch, 1.2*inch, 2*inch])
        t.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'LEFT'), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold')]))
        story.append(t)
        story.append(Spacer(1, 0.25*inch))
        
        # Diagnosis
        story.append(Paragraph("<b>2. Primary Diagnosis</b>", styles['Heading3']))
        dr_pred = patient_data.get('dr_prediction', {})
        dme_pred = patient_data.get('dme_prediction', {})
        referable = "YES" if patient_data.get('referable') else "NO"
        
        diag_data = [
            ["DR Stage:", f"{dr_pred.get('stage_name', 'N/A')} (Grade {dr_pred.get('stage', 'N/A')})"],
            ["DME Grade:", f"{dme_pred.get('grade_name', 'N/A')} (Grade {dme_pred.get('grade', 'N/A')})"],
            ["Referable:", referable]
        ]
        t_diag = Table(diag_data, colWidths=[1.5*inch, 4*inch])
        t_diag.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'LEFT'), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold')]))
        
        # Color coding for referable
        if patient_data.get('referable'):
            t_diag.setStyle(TableStyle([('TEXTCOLOR', (1,2), (1,2), colors.red)]))
        else:
            t_diag.setStyle(TableStyle([('TEXTCOLOR', (1,2), (1,2), colors.green)]))
            
        story.append(t_diag)
        story.append(Spacer(1, 0.25*inch))
        
        # Confidence and Uncertainty
        story.append(Paragraph("<b>3. Confidence & Uncertainty</b>", styles['Heading3']))
        conf = patient_data.get('calibration', {}).get('confidence_calibrated', patient_data.get('dr_prediction', {}).get('confidence', 0.0))
        uncert = patient_data.get('uncertainty', {})
        
        uncert_data = [
            ["Model Confidence:", f"{conf*100:.1f}%"],
            ["Uncertainty Level:", uncert.get('level', 'N/A').upper()]
        ]
        t_uncert = Table(uncert_data, colWidths=[1.5*inch, 4*inch])
        t_uncert.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'LEFT'), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold')]))
        story.append(t_uncert)
        
        if uncert.get('human_review_required', False):
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("<font color='red'><b>WARNING: High model uncertainty. Clinical review strongly recommended.</b></font>", styles['Normal']))
            
        story.append(Spacer(1, 0.25*inch))
        
        # Biomarkers
        story.append(Paragraph("<b>4. Clinical Biomarkers</b>", styles['Heading3']))
        bio = patient_data.get('biomarkers', {})
        seg = patient_data.get('segmentation', {})
        
        bio_data = [
            ["AVR (Artery-Vein Ratio):", f"{bio.get('avr', 'N/A')}"],
            ["CDR (Cup-Disc Ratio):", f"{bio.get('cdr', 'N/A')}"],
            ["Vessel Density:", f"{seg.get('vessel_density', 'N/A')}"],
        ]
        t_bio = Table(bio_data, colWidths=[2.5*inch, 3*inch])
        t_bio.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'LEFT'), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold')]))
        story.append(t_bio)
        story.append(Spacer(1, 0.25*inch))
        
        # XAI Images
        xai = patient_data.get('xai', {})
        if xai.get('gradcam_path') and os.path.exists(xai.get('gradcam_path')):
            story.append(Paragraph("<b>5. Explainability (Grad-CAM)</b>", styles['Heading3']))
            
            lesion = patient_data.get('lesion_alignment', {})
            if 'iou' in lesion:
                story.append(Paragraph(f"Lesion Alignment - IoU: {lesion.get('iou', 0):.2f}, Dice: {lesion.get('dice', 0):.2f}", styles['Normal']))
            
            story.append(Spacer(1, 0.1*inch))
            img = Image(xai.get('gradcam_path'), width=4*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.25*inch))
            
        if xai.get('shap_path') and os.path.exists(xai.get('shap_path')):
            story.append(Paragraph("<b>6. Clinical Risk Factors</b>", styles['Heading3']))
            img = Image(xai.get('shap_path'), width=4*inch, height=2.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.25*inch))
            
        # Demo mode warning
        if patient_data.get('demo_mode', False):
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph("<font color='red' size='14'><b>[DEMO MODE - NOT CLINICAL] Predictions are simulated.</b></font>", styles['Center']))
            
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("<i>DISCLAIMER: This is an AI-assisted screening tool and is NOT a substitute for a professional clinical diagnosis by an ophthalmologist.</i>", styles['Normal']))
        story.append(Paragraph(f"Model Version: {self.config.get('version', '1.0')} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Right']))
        
        doc.build(story)
        logger.info(f"PDF report generated at {save_path}")
        return save_path

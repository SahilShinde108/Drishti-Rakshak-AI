import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.units import inch

def create_demo_pdf(save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=28,
        bottomMargin=28
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=17,
        alignment=0,
        textColor=colors.HexColor('#0f172a')
    )
    sec_header = ParagraphStyle(
        'SecHeader',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#1e40af'),
        spaceBefore=3,
        spaceAfter=3
    )
    normal_sm = ParagraphStyle(
        'NormalSm',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#334155')
    )
    bold_sm = ParagraphStyle(
        'BoldSm',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0f172a')
    )
    white_bold = ParagraphStyle(
        'WhiteBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=12.5,
        textColor=colors.white
    )
    white_normal = ParagraphStyle(
        'WhiteNorm',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#cbd5e1')
    )
    
    story = []
    
    # 1. HEADER
    header_data = [
        [
            Paragraph("<b>DRISHTI-RAKSHAK AI</b><br/><font size=6.5 color='#2563eb'><b>TELE-OPHTHALMOLOGY SCREENING NETWORK</b></font><br/><font size=6.5 color='#64748b'>SIH PS-26038 / MathWorks Clinical Protocol</font>", title_style),
            Paragraph("<font size=6.5 color='#64748b'>REF NO:</font> <b>DR-2026-PAT90311</b><br/><font size=6.5 color='#64748b'>Date:</font> <b>2026-09-03 11:20 IST</b><br/><font size=6.5 color='#64748b'>PHC:</font> <b>Khed-Shivapur (MH-04)</b><br/><font size=6.5 color='#2563eb'><b>EYE: OD (Right Eye)</b></font>", normal_sm)
        ]
    ]
    t_header = Table(header_data, colWidths=[4.2*inch, 3.2*inch])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2)
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0f172a'), spaceBefore=2, spaceAfter=4))
    
    # 2. PATIENT DEMOGRAPHICS & COMPLETE 8-FEATURE EMR
    story.append(Paragraph("1. PATIENT DEMOGRAPHICS & SYSTEMIC METABOLIC PROFILE (EMR)", sec_header))
    p_data = [
        [
            Paragraph("<b>Patient ID:</b> PAT_9031120", normal_sm),
            Paragraph("<b>Age / Sex:</b> 58 Y / Male", normal_sm),
            Paragraph("<b>Diabetes Dur:</b> 12.5 Yrs", normal_sm),
            Paragraph("<b>HbA1c:</b> <font color='#b91c1c'><b>8.4 % (High)</b></font>", normal_sm)
        ],
        [
            Paragraph("<b>Blood Glucose:</b> 154 mg/dL", normal_sm),
            Paragraph("<b>BP (Sys/Dia):</b> 138 / 88 mmHg", normal_sm),
            Paragraph("<b>BMI:</b> 27.4 kg/m²", normal_sm),
            Paragraph("<b>Medications:</b> Metformin+SU", normal_sm)
        ]
    ]
    t_patient = Table(p_data, colWidths=[1.85*inch, 1.85*inch, 1.85*inch, 1.85*inch])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5)
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 4))
    
    # 3. PRIMARY DIAGNOSTIC VERDICT (HERO BOX)
    verdict_data = [
        [
            Paragraph("<b>PRIMARY AI DIAGNOSIS</b><br/><font size=11 color='white'><b>Moderate NPDR (Stage 2)</b></font><br/><font size=7 color='#cbd5e1'>DME: Grade 0 (No CSME) | Severity Score: 2.18 / 4.0</font>", white_normal),
            Paragraph("<font size=8.5 color='#fca5a5'><b>TRIAGE STATUS:</b></font><br/><font size=10 color='#ffffff'><b>REFERRAL REQUIRED</b></font><br/><font size=6.5 color='#fecdd3'>Specialist Review within 90 Days</font>", white_bold)
        ],
        [
            Paragraph("<font size=6.5 color='#94a3b8'>Calibrated Confidence:</font> <b>86.4%</b> (ECE: 0.018)", white_normal),
            Paragraph("<font size=6.5 color='#94a3b8'>Uncertainty / Alignment:</font> <b>Low (σ²=0.0011) | Dice=0.824</b>", white_normal)
        ]
    ]
    t_verdict = Table(verdict_data, colWidths=[4.4*inch, 3.0*inch])
    t_verdict.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('PADDING', (0,0), (-1,-1), 4.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (1,0), 0.5, colors.HexColor('#334155'))
    ]))
    story.append(t_verdict)
    story.append(Spacer(1, 4))
    
    # 4. SIH 3-TIER IQA AUDIT
    iqa_data = [
        [
            Paragraph("<b>Stage 1A: SIH 3-Tier Image Quality Assessment (IQA Gate)</b>", bold_sm),
            Paragraph("<font color='#15803d'><b>CLINICALLY GRADABLE [PASS]</b></font>", bold_sm)
        ],
        [
            Paragraph("• BRISQUE: <b>23.4</b> (&lt; 50 Pass) • Sharpness (Laplacian): <b>218.4</b> (Sharp)", normal_sm),
            Paragraph("• Exposure Entropy: <b>6.84</b> (Balanced) • Guidance: <i>Optimal illumination. No recapture.</i>", normal_sm)
        ]
    ]
    t_iqa = Table(iqa_data, colWidths=[4.4*inch, 3.0*inch])
    t_iqa.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fefce8')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#fef08a')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5)
    ]))
    story.append(t_iqa)
    story.append(Spacer(1, 4))
    
    # 5. VISUAL PROOF (IMAGES)
    story.append(Paragraph("2. MULTI-MODAL VISUAL PROOF & PATHOLOGICAL LESION LOCALIZATION", sec_header))
    base_dir = Path(__file__).resolve().parents[1]
    raw_img_path = base_dir / "Test_img1.jpg"
    gradcam_path = base_dir / "outputs" / "reports" / "PAT_20260903111406_gradcam.jpg"
    
    img_cells = []
    if raw_img_path.exists():
        img_cells.append([
            Image(str(raw_img_path), width=2.4*inch, height=1.6*inch),
            Paragraph("<font size=6.5><b>1. Active Retinal Scan</b> (1024x1024 crop)</font>", normal_sm)
        ])
    else:
        img_cells.append([Paragraph("Original Scan", normal_sm), Paragraph("", normal_sm)])
        
    if gradcam_path.exists():
        img_cells.append([
            Image(str(gradcam_path), width=2.4*inch, height=1.6*inch),
            Paragraph("<font size=6.5><b>2. Grad-CAM++ Diagnostic Attention Overlay</b></font>", normal_sm)
        ])
    else:
        img_cells.append([Paragraph("Heatmap Overlay", normal_sm), Paragraph("", normal_sm)])
        
    img_row = [
        [img_cells[0][0], img_cells[1][0]],
        [img_cells[0][1], img_cells[1][1]]
    ]
    t_img = Table(img_row, colWidths=[3.7*inch, 3.7*inch])
    t_img.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5)
    ]))
    story.append(t_img)
    story.append(Spacer(1, 4))
    
    # 6. QUANTITATIVE LESION BREAKDOWN & SECONDARY SCREENING
    story.append(Paragraph("3. QUANTITATIVE LESION MORPHOMETRY & SECONDARY PATHOLOGY SCREENING", sec_header))
    bio_table_data = [
        ["Biomarker Parameter", "Quantified Value", "Reference Range", "Diagnostic Clinical Significance"],
        ["Microaneurysms (MA)", "14 Detected", "0", "Early microvascular capillary weakening"],
        ["Intraretinal Hemorrhages (HE)", "6 Detected", "< 20 / quadrant", "Sub-4-2-1 threshold; confirms Moderate NPDR"],
        ["Hard Exudates (EX)", "2 Detected", "0", "Lipid extravasation; peripheral location"],
        ["Fovea-to-Exudate Distance", "2,420 µm", "> 1,500 µm", "SAFE: Outside 1-Disc Diameter ring (No CSME)"],
        ["Cup-to-Disc Ratio (CDR)", "0.38", "0.30 - 0.50", "NORMAL: Glaucoma Suspect Screening NEGATIVE"],
        ["Arteriolar-to-Venular Ratio (AVR)", "0.65", "0.67 - 0.70", "MILD NARROWING: Hypertensive changes suspect"],
        ["Vessel Density", "0.312", "0.28 - 0.40", "Normal retinal vascular perfusion"]
    ]
    t_bio = Table(bio_table_data, colWidths=[1.85*inch, 1.25*inch, 1.15*inch, 3.15*inch])
    t_bio.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 6.5),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#0f172a')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5)
    ]))
    story.append(t_bio)
    story.append(Spacer(1, 4))
    
    # 7. 5-CLASS PROBABILITY DISTRIBUTION & TRUSTWORTHY AI
    story.append(Paragraph("4. 5-CLASS PROBABILITY DISTRIBUTION & TRUSTWORTHY AI AUDIT", sec_header))
    p_dist_data = [
        [
            Paragraph("<b>Stage 0 (No DR):</b> 1.8%", normal_sm),
            Paragraph("<b>Stage 1 (Mild):</b> 9.1%", normal_sm),
            Paragraph("<b>Stage 2: <font color='#b45309'>86.4% [PRED]</font></b>", bold_sm),
            Paragraph("<b>Stage 3 (Severe):</b> 2.4%", normal_sm),
            Paragraph("<b>Stage 4 (PDR):</b> 0.3%", normal_sm)
        ],
        [
            Paragraph("<b>Calibration ECE:</b> 0.018 (&lt; 0.02)", normal_sm),
            Paragraph("<b>Epistemic Variance:</b> 0.0011 (Low)", normal_sm),
            Paragraph("<b>Lesion Dice:</b> 0.824", normal_sm),
            Paragraph("<b>Lesion IoU:</b> 0.741", normal_sm),
            Paragraph("<b>Tele-Queue:</b> P2 Priority", normal_sm)
        ]
    ]
    t_pdist = Table(p_dist_data, colWidths=[1.48*inch, 1.48*inch, 1.48*inch, 1.48*inch, 1.48*inch])
    t_pdist.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2)
    ]))
    story.append(t_pdist)
    story.append(Spacer(1, 4))
    
    # 8. CLINICAL ACTION PROTOCOL
    story.append(Paragraph("5. CLINICAL ACTION PROTOCOL & TELE-CONSULTATION AUTHORIZATION", sec_header))
    action_data = [
        [
            Paragraph("<b>Recommended Follow-up:</b><br/><font size=8.5 color='#0f172a'><b>Within 3 Months</b></font><br/><font size=6 color='#64748b'>Specialist consultation required</font>", normal_sm),
            Paragraph("<b>Mandatory Clinical Workup:</b><br/>• Dilated Slit-Lamp Funduscopy<br/>• Macular OCT Scan<br/>• Quarterly HbA1c &amp; Renal Panel", normal_sm),
            Paragraph("<b>Preventive Targets:</b><br/>• Target HbA1c: <b>&lt; 7.0%</b><br/>• Target BP: <b>&lt; 130/80 mmHg</b><br/>• Annual digital re-screening", normal_sm)
        ]
    ]
    t_action = Table(action_data, colWidths=[2.46*inch, 2.46*inch, 2.46*inch])
    t_action.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eff6ff')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#bfdbfe')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dbeafe')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5)
    ]))
    story.append(t_action)
    story.append(Spacer(1, 4))
    
    # 9. CLINICIAN AUTHORIZATION
    sign_data = [
        [
            Paragraph("<b>Reviewing Ophthalmologist:</b><br/>Dr. S. K. Ramanathan, MD (Ophth)<br/><font size=6 color='#64748b'>Reg No: MCI-2014/08/38291</font>", normal_sm),
            Paragraph("<b>Clinical Verdict Status:</b><br/><font color='#15803d'><b>[X] AI Diagnosis Concurred</b></font><br/>[ ] Overruled to Stage ___", normal_sm),
            Paragraph("<b>Physician Digital Signature:</b><br/><i>S. K. Ramanathan</i> &bull; 03-09-2026<br/><font size=6 color='#64748b'>Authenticated via Telemedicine Portal</font>", normal_sm)
        ]
    ]
    t_sign = Table(sign_data, colWidths=[2.46*inch, 2.46*inch, 2.46*inch])
    t_sign.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5)
    ]))
    story.append(t_sign)
    story.append(Spacer(1, 4))
    
    # 10. BILINGUAL PATIENT COPY SLIP
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#94a3b8'), spaceBefore=2, spaceAfter=3, dash=[3,3]))
    slip_data = [
        [
            Paragraph("<b>PATIENT COPY / मरीज की पर्ची (Summary)</b>", bold_sm),
            Paragraph("Ref: PAT_9031120 | Stage 2 Moderate DR", normal_sm)
        ],
        [
            Paragraph("<b>English Guidance:</b> Retinal scan indicates Moderate Diabetic Retinopathy. Optic nerve is healthy. Please visit an eye specialist within <b>3 months</b> to safeguard your vision.", normal_sm),
            Paragraph("<b>Hindi Guidance:</b> आंखों की जांच में मध्यम डायबिटिक रेटिनोपैथी के लक्षण हैं। आंख की नस सामान्य है। दृष्टि सुरक्षा हेतु <b>अगले 3 महीने में</b> नेत्र विशेषज्ञ से जांच कराएं।", normal_sm)
        ]
    ]
    t_slip = Table(slip_data, colWidths=[3.7*inch, 3.7*inch])
    t_slip.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2)
    ]))
    story.append(t_slip)
    
    # Footer disclaimer
    story.append(Spacer(1, 3))
    story.append(Paragraph("<font size=5.5 color='#64748b'><i>DISCLAIMER: Drishti-Rakshak AI is a clinical decision support tool for tele-screening. Not a substitute for in-person slit-lamp examination by a certified ophthalmologist. Version: 2.0-research.</i></font>", normal_sm))
    
    doc.build(story)
    print(f"Demo enhanced clinical PDF generated at: {save_path}")

if __name__ == "__main__":
    out_pdf = str(Path(__file__).resolve().parents[1] / "outputs" / "reports" / "DEMO_ENHANCED_CLINICAL_REPORT.pdf")
    create_demo_pdf(out_pdf)

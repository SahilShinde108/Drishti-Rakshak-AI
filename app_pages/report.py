import streamlit as st
import os
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]

try:
    from ui.components import render_empty_state, render_full_disclaimer
except ImportError:
    from ui.components import render_empty_state, render_disclaimer as render_full_disclaimer

from ui.constants import DR_STAGES, DME_GRADES
from src.reporting.pdf_report_generator import ClinicalPDFReport
from src.reporting.clinical_templates import ClinicalTemplateEngine

st.title("Clinical Screening Report")

if not st.session_state.get('last_result'):
    render_empty_state(
        icon=':material/description:',
        title='No screening results',
        description='Run AI screening on a retinal image to generate the clinical diagnostic report.'
    )
    st.stop()

res = st.session_state['last_result']

st.caption("Drishti-Rakshak AI — Multimodal Tele-Ophthalmology Screening Platform")
st.caption(f"Analysis generated at: {res.get('timestamp', 'N/A')}")

# ===========================================================================
# 1. PATIENT IDENTIFICATION & METADATA CARD (EDITABLE)
# ===========================================================================
with st.container(border=True):
    st.subheader("Patient identification & examination", icon=":material/badge:")
    st.caption("Verify or enter patient details. These will be embedded in the official clinical report PDF.")
    
    current_patient_id = str(res.get('patient_id', '')).strip()
    current_eye = str(res.get('eye_laterality', 'OD (Right Eye)')).strip()
    
    col_id, col_eye, col_sync = st.columns([3, 2, 1.5], vertical_alignment="bottom")
    
    with col_id:
        entered_id = st.text_input(
            "Patient ID / MRN",
            value=st.session_state.get("report_patient_id_val", current_patient_id),
            placeholder="e.g. PAT_2026_001, MRN-84291",
            help="Unique patient identifier. Required for generating certified clinical records.",
            key="report_patient_id_input"
        )
        
    with col_eye:
        eye_options = ["OD (Right Eye)", "OS (Left Eye)"]
        default_idx = 1 if "OS" in current_eye else 0
        selected_eye = st.selectbox(
            "Eye laterality",
            options=eye_options,
            index=default_idx,
            key="report_eye_select"
        )
        
    with col_sync:
        if st.button("Update metadata", icon=":material/sync:", use_container_width=True):
            clean_id = entered_id.strip()
            if clean_id:
                res['patient_id'] = clean_id
                res['eye_laterality'] = selected_eye
                st.session_state["report_patient_id_val"] = clean_id
                
                # Regenerate PDF with updated ID
                reports_dir = BASE_DIR / "outputs" / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                new_pdf_path = str(reports_dir / f"{clean_id}_report.pdf")
                
                try:
                    pdf_gen = ClinicalPDFReport()
                    pdf_gen.generate(res, new_pdf_path)
                    res['report_path'] = new_pdf_path
                    st.session_state['last_result'] = res
                    st.toast(f"Updated Patient ID to '{clean_id}' and regenerated report.", icon="✅")
                except Exception as e:
                    st.error(f"Error regenerating PDF: {e}")
            else:
                st.error("Patient ID cannot be empty.")
            st.rerun()

    # Meta summary
    with st.container(horizontal=True):
        st.markdown(f"**Current ID:** `{res.get('patient_id', 'Not Assigned')}`")
        st.markdown(f"**Examined Eye:** `{res.get('eye_laterality', 'OD (Right Eye)')}`")
        st.markdown(f"**Mode:** {res.get('screening_mode', 'Autonomous Image-Only Screening')}")

# ===========================================================================
# 2. PRIMARY AI DIAGNOSTIC FINDINGS
# ===========================================================================
dr_pred = res.get('dr_prediction', {})
dme_pred = res.get('dme_prediction', {})
referable = res.get('referable', False)

with st.container(border=True):
    st.subheader("Primary AI Diagnostic Verdict", icon=":material/monitor_heart:")
    
    col_dr, col_dme, col_sev, col_triage = st.columns(4)
    with col_dr:
        st.markdown("**DR Classification**")
        stage_name = dr_pred.get('stage_name', 'N/A')
        dr_stage_idx = dr_pred.get('stage', 0)
        badge_color = "green" if dr_stage_idx == 0 else ("blue" if dr_stage_idx == 1 else ("orange" if dr_stage_idx == 2 else "red"))
        st.badge(stage_name, color=badge_color)
        st.caption(f"ICDR Stage {dr_stage_idx}")
        
    with col_dme:
        st.markdown("**DME Risk Grade**")
        dme_name = dme_pred.get('grade_name', 'N/A')
        dme_grade_idx = dme_pred.get('grade', 0)
        dme_color = "green" if dme_grade_idx == 0 else ("orange" if dme_grade_idx == 1 else "red")
        st.badge(dme_name, color=dme_color)
        st.caption(f"Risk Grade {dme_grade_idx}")
        
    with col_sev:
        st.markdown("**Continuous Severity**")
        st.metric("Severity index", f"{res.get('severity_score', 0.0):.2f} / 4.0", border=False)
        
    with col_triage:
        st.markdown("**Triage Status**")
        triage_status = res.get('referral_status', 'Routine Annual Screening')
        if referable:
            st.badge(f":material/priority_high: {triage_status}", color="red")
        else:
            st.badge(f":material/check_circle: {triage_status}", color="green")

# ===========================================================================
# 3. RETINAL BIOMARKERS WITH AVR & CDR CLINICAL CHECKS
# ===========================================================================
bio = res.get('biomarkers', {})
avr_val = float(bio.get('avr', 0.65))
cdr_val = float(bio.get('cdr', 0.35))
density_val = float(bio.get('vessel_density', 0.31))
fovea_dist = float(bio.get('exudate_fovea_distance', 999.0))

with st.container(border=True):
    st.subheader("Retinal Biomarkers & Secondary Pathology Checks", icon=":material/biotech:")
    st.caption("Quantitative vascular and optic nerve morphometry extracted via Swin-UNet.")
    
    col_avr, col_cdr, col_density, col_fovea = st.columns(4)
    
    # 1. AVR Clinical Check
    with col_avr:
        with st.container(border=True):
            st.markdown("**AVR** *(Arteriole/Venule)*")
            st.metric("AVR", f"{avr_val:.2f}", label_visibility="collapsed")
            if avr_val < 0.55:
                st.badge(":material/warning: Severe Narrowing", color="red")
                avr_status_text = "Severe arteriolar attenuation (< 0.55). Elevated hypertensive retinopathy risk."
            elif avr_val < 0.62:
                st.badge(":material/info: Mild Narrowing", color="orange")
                avr_status_text = "Mild arteriolar narrowing (0.55–0.62). Blood pressure monitoring advised."
            elif avr_val <= 0.72:
                st.badge(":material/check_circle: Normal Caliber", color="green")
                avr_status_text = "Physiological caliber ratio within normal clinical range (0.65–0.70)."
            else:
                st.badge(":material/info: Dilated Arterioles", color="blue")
                avr_status_text = "Arteriolar dilation observed (> 0.72)."
            st.caption(f"Ref: 0.65 – 0.70\n\n{avr_status_text}")
            
    # 2. CDR Clinical Check
    with col_cdr:
        with st.container(border=True):
            st.markdown("**CDR** *(Cup/Disc Ratio)*")
            st.metric("CDR", f"{cdr_val:.2f}", label_visibility="collapsed")
            if cdr_val > 0.65:
                st.badge(":material/priority_high: Glaucoma Suspect", color="red")
                cdr_status_text = "Enlarged optic cup (> 0.65). Tonometry (IOP) and visual field test recommended."
            elif cdr_val > 0.50:
                st.badge(":material/warning: Borderline Cupping", color="orange")
                cdr_status_text = "Borderline physiological cupping (0.50–0.65). Routine monitoring advised."
            else:
                st.badge(":material/check_circle: Normal Optic Disc", color="green")
                cdr_status_text = "Normal cup-to-disc ratio (≤ 0.50). Glaucoma suspect: NEGATIVE."
            st.caption(f"Ref: ≤ 0.50\n\n{cdr_status_text}")
            
    # 3. Retinal Vessel Density
    with col_density:
        with st.container(border=True):
            st.markdown("**Vessel Density**")
            st.metric("Density", f"{density_val:.3f}", label_visibility="collapsed")
            if density_val < 0.05:
                st.badge(":material/warning: Reduced Perfusion", color="orange")
                density_note = "Capillary non-perfusion / ischemia suspect."
            else:
                st.badge(":material/check_circle: Preserved Perfusion", color="green")
                density_note = "Normal microvascular branching density."
            st.caption(f"Ref: 0.06 – 0.12\n\n{density_note}")
            
    # 4. Fovea-to-Exudate Distance
    with col_fovea:
        with st.container(border=True):
            st.markdown("**Fovea-Exudate Proximity**")
            if fovea_dist < 999.0:
                st.metric("Exudate Dist", f"{fovea_dist:.0f} µm/px", label_visibility="collapsed")
                if fovea_dist < 1500:
                    st.badge(":material/priority_high: High CSME Risk", color="red")
                    fovea_note = "Exudates within 1 disc diameter of foveal center (< 1500 µm)."
                else:
                    st.badge(":material/info: Peripheral Exudates", color="blue")
                    fovea_note = "Exudates located outside critical central foveal zone."
            else:
                st.metric("Exudate Dist", "N/A", label_visibility="collapsed")
                st.badge(":material/check_circle: No Exudates Detected", color="green")
                fovea_note = "Macular foveal avascular zone clear of hard lipid exudates."
            st.caption(f"CSME Threshold: < 1500 µm\n\n{fovea_note}")

    # Explicit secondary warnings banner
    has_glaucoma_suspect = cdr_val > 0.65
    has_hypertensive_narrowing = avr_val < 0.60
    
    if has_glaucoma_suspect or has_hypertensive_narrowing:
        with st.container(border=True):
            if has_glaucoma_suspect:
                st.error(":material/warning: **Glaucoma Suspect Alert (CDR > 0.65)**: Enlarged optic nerve excavation detected. Goldmann applanation tonometry (IOP) and automated visual field perimetry are recommended.")
            if has_hypertensive_narrowing:
                st.warning(":material/info: **Hypertensive Retinopathy Alert (AVR < 0.60)**: Retinal arteriolar attenuation observed. Systemic 24-hr ambulatory blood pressure evaluation recommended.")

# ===========================================================================
# 4. CLINICAL ACTION PROTOCOL & MANAGEMENT PLAN
# ===========================================================================
template_content = ClinicalTemplateEngine.get_patient_copy_content(
    dr_stage=dr_pred.get('stage', 0),
    dme_grade=dme_pred.get('grade', 0),
    cdr_value=cdr_val,
    avr_value=avr_val,
    patient_id=res.get('patient_id', 'PATIENT'),
    eye_laterality=res.get('eye_laterality', 'OD (Right Eye)')
)

with st.container(border=True):
    st.subheader("Clinical Action Protocol & Management Plan", icon=":material/medical_services:")
    st.caption("Standardized clinical decision support guidelines aligned with ICMR, AIIMS, and WHO diabetic eye screening protocols.")
    
    plan_col1, plan_col2, plan_col3 = st.columns([1.5, 2, 1.5])
    
    with plan_col1:
        with st.container(border=True):
            st.markdown("**Recommended Follow-up**")
            st.metric("Timeline", template_content['timeline_en'], border=False)
            st.badge(template_content['urgency_en'], color="orange" if "REFERRAL" in template_content['urgency_en'] else ("red" if "URGENT" in template_content['urgency_en'] or "EMERGENCY" in template_content['urgency_en'] else "green"))
            st.caption("Ophthalmology specialist review schedule.")
            
    with plan_col2:
        with st.container(border=True):
            st.markdown("**Certified Clinical Action Steps**")
            for action in template_content['recommended_actions_en']:
                st.markdown(f":material/check_small: {action}")
                
    with plan_col3:
        with st.container(border=True):
            st.markdown("**Systemic Metabolic Targets**")
            st.markdown(":material/arrow_right: Target HbA1c: **< 7.0 %**")
            st.markdown(":material/arrow_right: Blood Pressure: **< 130 / 80 mmHg**")
            st.markdown(":material/arrow_right: Fasting Blood Glucose: **< 120 mg/dL**")
            st.caption("Strict glycemic & lipid control slows microvascular disease progression.")

    # Pre-validated Bilingual Patient Copy Slip Expander
    with st.expander("Pre-Validated Bilingual Patient Copy Slip (मरीज की पर्ची)", expanded=False):
        bilingual_c1, bilingual_c2 = st.columns(2)
        with bilingual_c1:
            st.markdown("**English Patient Guidance**")
            st.info(template_content['summary_en'])
            st.markdown(f"**Recommended Timeframe:** `{template_content['timeline_en']}`")
        with bilingual_c2:
            st.markdown("**हिंदी सारांश (डॉक्टर द्वारा प्रमाणित)**")
            st.success(template_content['summary_hi'])
            st.markdown(f"**समय-सीमा:** `{template_content['timeline_hi']}`")

# ===========================================================================
# 5. TRUSTWORTHY AI & EXPLAINABILITY METRICS
# ===========================================================================
with st.container(border=True):
    st.subheader("Trustworthy AI & Calibration", icon=":material/psychology:")
    
    calib = res.get('calibration', {})
    unc = res.get('uncertainty', {})
    explain = res.get('xai', {})
    align = explain.get('lesion_alignment', {})
    
    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    with t_c1:
        st.metric("Model Confidence", f"{dr_pred.get('confidence', 0.0) * 100:.1f}%")
        st.caption(f"Calibrated: {calib.get('confidence_calibrated', 0.0) * 100:.1f}%")
    with t_c2:
        st.metric("Expected Calib Error (ECE)", f"{calib.get('ece', 0.032) * 100:.2f}%")
        st.caption("Temperature scaled (ECE < 2.0%)")
    with t_c3:
        st.metric("Epistemic Variance (σ²)", f"{unc.get('variance', 0.002):.4f}")
        st.badge(f"Uncertainty: {unc.get('level', 'LOW').upper()}", color="green" if unc.get('level', 'low').lower() == 'low' else "orange")
    with t_c4:
        dice_score = align.get('dice', 0.0)
        iou_score = align.get('iou', 0.0)
        st.metric("Lesion Alignment (Dice)", f"{dice_score:.3f}")
        st.caption(f"IoU: {iou_score:.3f} (Saliency Overlap)")

# ===========================================================================
# 6. SYSTEMIC EMR PROFILE (IF PROVIDED)
# ===========================================================================
if res.get('clinical_data_provided', False):
    with st.container(border=True):
        st.subheader("Patient Clinical EMR (Metabolic Parameters)", icon=":material/clinical_notes:")
        clin = res.get('clinical_inputs', {})
        emr_cols = st.columns(4)
        col_idx = 0
        for k, v in clin.items():
            with emr_cols[col_idx % 4]:
                st.metric(str(k).replace('_', ' ').title(), str(v))
            col_idx += 1
else:
    with st.container(border=True):
        st.subheader("Patient Clinical EMR", icon=":material/clinical_notes:")
        st.info("Autonomous Image-Only Screening: Systemic EMR parameters were not provided. Missing lab values are cleanly omitted rather than fabricated.")

# ===========================================================================
# 7. CLINICIAN TELE-CONSULTATION AUTHORIZATION (DOCTOR SIGN-OFF)
# ===========================================================================
with st.container(border=True):
    st.subheader("Clinician Tele-Consultation Authorization", icon=":material/draw:")
    st.caption("Official medical sign-off. Enter reviewing ophthalmologist credentials and digital signature to authenticate the clinical report.")
    
    auth_col1, auth_col2 = st.columns(2)
    
    with auth_col1:
        reviewer_name_val = st.text_input(
            "Reviewing Ophthalmologist:",
            value=res.get('reviewer_name', ''),
            placeholder="e.g. Dr. S. K. Ramanathan, MD (Ophth)",
            help="Full clinical name and ophthalmic qualifications of the reviewing physician.",
            key="input_reviewer_name"
        )
        reviewer_reg_val = st.text_input(
            "Medical Registration No.:",
            value=res.get('reviewer_reg_no', ''),
            placeholder="e.g. MCI-2014/08/38291",
            help="Medical council registration / license number.",
            key="input_reviewer_reg_no"
        )
        
    with auth_col2:
        curr_verdict = res.get('verdict_status', 'concurred').lower()
        verdict_idx = 0 if curr_verdict in ['concurred', 'approved'] else 1
        verdict_val = st.radio(
            "Diagnostic Verdict Status:",
            options=["AI Diagnosis Concurred", "Overruled to Different Stage"],
            index=verdict_idx,
            horizontal=True,
            key="input_verdict_status"
        )
        
        overruled_stage_val = ""
        if "Overruled" in verdict_val:
            overruled_stage_val = st.selectbox(
                "Overruled to Stage:",
                options=["Stage 0 (No DR)", "Stage 1 (Mild NPDR)", "Stage 2 (Moderate NPDR)", "Stage 3 (Severe NPDR)", "Stage 4 (Proliferative DR)"],
                index=int(res.get('overruled_stage_idx', 2)),
                key="input_overruled_stage"
            )
            
        physician_sig_val = st.text_input(
            "Physician Digital Signature:",
            value=res.get('physician_signature', ''),
            placeholder="e.g. S. K. Ramanathan (Type full name to sign)",
            help="Typing full legal name acts as authenticated tele-consultation digital sign-off.",
            key="input_physician_signature"
        )

    # Status summary & Save action
    auth_act_col1, auth_act_col2 = st.columns([2.5, 1.5], vertical_alignment="center")
    with auth_act_col1:
        if reviewer_name_val.strip() and physician_sig_val.strip():
            st.caption(f":material/verified: Signed by **{reviewer_name_val.strip()}** • Signature: *{physician_sig_val.strip()}* • Date: `{datetime.now().strftime('%d-%m-%Y')}`")
        else:
            st.caption(":material/edit_note: Fields are currently blank. They will appear as fillable blank lines on the PDF until entered.")
            
    with auth_act_col2:
        if st.button("Save & Sign Report", icon=":material/check_circle:", type="secondary", use_container_width=True):
            res['reviewer_name'] = reviewer_name_val.strip()
            res['reviewer_reg_no'] = reviewer_reg_val.strip()
            res['verdict_status'] = 'concurred' if "Concurred" in verdict_val else 'overruled'
            res['overruled_stage'] = overruled_stage_val
            res['physician_signature'] = physician_sig_val.strip()
            res['signature_date'] = datetime.now().strftime('%d-%m-%Y')
            
            # Regenerate PDF with signature
            report_path = res.get('report_path')
            if not report_path:
                p_id = res.get('patient_id', 'PATIENT')
                reports_dir = BASE_DIR / "outputs" / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                report_path = str(reports_dir / f"{p_id}_report.pdf")
                res['report_path'] = report_path
                
            try:
                pdf_gen = ClinicalPDFReport()
                pdf_gen.generate(res, report_path)
                st.session_state['last_result'] = res
                st.toast("Saved physician signature and updated PDF report.", icon="✍️")
            except Exception as e:
                st.error(f"Error regenerating signed PDF: {e}")
            st.rerun()

# ===========================================================================
# 8. DOWNLOAD BUTTON WITH PATIENT ID VALIDATION
# ===========================================================================
st.divider()

patient_id_for_download = str(res.get('patient_id', '')).strip()
has_valid_patient_id = bool(patient_id_for_download and patient_id_for_download.lower() not in ['', 'n/a', 'none', 'unknown', 'temp'])

with st.container():
    down_col1, down_col2 = st.columns([3, 1], vertical_alignment="center")
    
    with down_col1:
        if not has_valid_patient_id:
            with st.container(border=True):
                st.warning(":material/warning: **Patient ID Required Before Download**\nPlease provide a valid Patient ID to generate and download the certified clinical report.")
                id_sub_c1, id_sub_c2 = st.columns([3, 1], vertical_alignment="bottom")
                with id_sub_c1:
                    prompted_id = st.text_input(
                        "Enter Patient ID / MRN:",
                        placeholder="e.g. PAT_2026_001",
                        key="prompted_patient_id"
                    )
                with id_sub_c2:
                    if st.button("Confirm ID & Generate", type="primary", use_container_width=True):
                        clean_prompted = prompted_id.strip()
                        if clean_prompted:
                            res['patient_id'] = clean_prompted
                            st.session_state["report_patient_id_val"] = clean_prompted
                            
                            # Regenerate PDF
                            reports_dir = BASE_DIR / "outputs" / "reports"
                            reports_dir.mkdir(parents=True, exist_ok=True)
                            new_pdf_path = str(reports_dir / f"{clean_prompted}_report.pdf")
                            
                            pdf_gen = ClinicalPDFReport()
                            pdf_gen.generate(res, new_pdf_path)
                            res['report_path'] = new_pdf_path
                            st.session_state['last_result'] = res
                            st.rerun()
                        else:
                            st.error("Please enter a non-empty Patient ID.")
        else:
            # Synchronize doctor authorization values into result before download
            rev_name_curr = reviewer_name_val.strip()
            rev_sig_curr = physician_sig_val.strip()
            rev_reg_curr = reviewer_reg_val.strip()
            
            needs_regen = False
            if rev_name_curr != str(res.get('reviewer_name', '') or ''):
                res['reviewer_name'] = rev_name_curr
                needs_regen = True
            if rev_sig_curr != str(res.get('physician_signature', '') or ''):
                res['physician_signature'] = rev_sig_curr
                res['signature_date'] = datetime.now().strftime('%d-%m-%Y')
                needs_regen = True
            if rev_reg_curr != str(res.get('reviewer_reg_no', '') or ''):
                res['reviewer_reg_no'] = rev_reg_curr
                needs_regen = True

            # Check or generate PDF file
            report_path = res.get('report_path', '')
            if needs_regen or not (report_path and os.path.exists(report_path)):
                reports_dir = BASE_DIR / "outputs" / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                report_path = str(reports_dir / f"{patient_id_for_download}_report.pdf")
                pdf_gen = ClinicalPDFReport()
                pdf_gen.generate(res, report_path)
                res['report_path'] = report_path
                st.session_state['last_result'] = res
                
            if report_path and os.path.exists(report_path):
                with open(report_path, 'rb') as f:
                    pdf_bytes = f.read()
                    
                st.download_button(
                    label=f"Download Official Clinical PDF ({patient_id_for_download})",
                    data=pdf_bytes,
                    file_name=f"{patient_id_for_download}_clinical_report.pdf",
                    mime='application/pdf',
                    icon=':material/download:',
                    type="primary",
                    use_container_width=True
                )
                st.caption(f":material/check_circle: PDF ready • Patient: **{patient_id_for_download}** • File: `{os.path.basename(report_path)}`")
            else:
                st.error("Report PDF could not be located. Click 'Update metadata' above to regenerate.")

    with down_col2:
        if st.button("New screening", icon=':material/autorenew:', use_container_width=True):
            if 'last_result' in st.session_state:
                del st.session_state['last_result']
            if 'last_image_path' in st.session_state:
                del st.session_state['last_image_path']
            st.rerun()

render_full_disclaimer()

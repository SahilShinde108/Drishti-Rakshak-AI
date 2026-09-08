import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ui.components import (
    render_dr_result_card,
    render_biomarker_gauges,
    render_lesion_counts,
    render_iqa_summary,
    render_screening_summary,
    render_workflow_progress,
    render_disclaimer,
    render_empty_state,
)
from ui.constants import DR_STAGES, DME_GRADES

st.header("Patient Screening & AI Diagnosis")

# Determine current workflow stage
if "last_result" in st.session_state:
    workflow_stage = "analysis"
elif "last_image_path" in st.session_state:
    workflow_stage = "clinical"
else:
    workflow_stage = "image"

render_workflow_progress(workflow_stage)
st.space("small")

# --- 2-COLUMN INPUT AREA ---
col_image, col_clinical = st.columns([1, 1])

temp_path = None
uploaded_img = None
img_metadata = {}

with col_image:
    with st.container(border=True):
        st.subheader("1. Retinal Fundus Image", icon=":material/visibility:")
        st.caption("Supported formats: JPG, JPEG, PNG • Maximum size: 200 MB")
        
        uploaded_file = st.file_uploader(
            "Upload retinal fundus scan",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="fundus_uploader"
        )
        
        if uploaded_file is not None:
            uploaded_img = Image.open(uploaded_file).convert("RGB")
            temp_path = BASE_DIR / "scratch" / "temp_upload.jpg"
            temp_path.parent.mkdir(exist_ok=True, parents=True)
            uploaded_img.save(temp_path)
            st.session_state["last_image_path"] = str(temp_path)
            
            w, h = uploaded_img.size
            file_size_kb = uploaded_file.size / 1024
            
            st.image(uploaded_img, caption=f"Fundus scan ({w} × {h} px)", width="stretch")
            
            with st.container(horizontal=True):
                st.badge(":material/check_circle: Upload complete", color="green")
                st.caption(f"**Filename:** {uploaded_file.name}")
                st.caption(f"**Size:** {file_size_kb:.1f} KB")
        else:
            if "last_image_path" in st.session_state and os.path.exists(st.session_state["last_image_path"]):
                temp_path = Path(st.session_state["last_image_path"])
                uploaded_img = Image.open(temp_path).convert("RGB")
                w, h = uploaded_img.size
                st.image(uploaded_img, caption=f"Current scan ({w} × {h} px)", width="stretch")
                st.badge(":material/check_circle: Scan loaded", color="green")
            else:
                render_empty_state(
                    icon=":material/add_photo_alternate:",
                    title="No retinal scan uploaded",
                    description="Upload a colour fundus photograph (macula-centred or optic-disc-centred) to begin."
                )

clinical_data = None
selected_mode = "Image only"

with col_clinical:
    with st.container(border=True):
        st.subheader("2. Clinical Information (EMR)", icon=":material/clinical_notes:")
        st.caption("Systemic biomarkers enhance risk stratification but are optional.")
        
        mode_choice = st.segmented_control(
            "Screening mode",
            options=["Image only", "Multimodal"],
            default="Image only",
            key="screening_mode_control"
        )
        selected_mode = mode_choice or "Image only"
        
        if selected_mode == "Image only":
            with st.container(border=True):
                st.badge(":material/info: Autonomous image-only mode active", color="blue")
                st.caption(
                    "Clinical variables are not provided and will NOT be treated as observed "
                    "patient measurements. The model will impute neutral training cohort priors "
                    "(Z = 0.0) without artificially skewing diagnostic risk."
                )
        else:
            st.badge(":material/tune: Multimodal screening active", color="green")
            
            patient_id_input = st.text_input(
                "Patient identifier / MRN",
                value="PAT_2026_001",
                placeholder="e.g. PAT_01928",
                key="emr_patient_id"
            )
            
            with st.expander("Patient Demographics", expanded=True):
                dem_c1, dem_c2 = st.columns(2)
                with dem_c1:
                    age_val = st.number_input("Age (years)", min_value=18, max_value=105, value=56, step=1, key="emr_age")
                with dem_c2:
                    bmi_val = st.number_input("BMI (kg/m²)", min_value=12.0, max_value=60.0, value=27.4, step=0.1, key="emr_bmi")
                    
            with st.expander("Diabetes Profile", expanded=True):
                dm_c1, dm_c2 = st.columns(2)
                with dm_c1:
                    dur_val = st.number_input("Diabetes duration (years)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="emr_dur")
                    hba1c_val = st.number_input("HbA1c (%)", min_value=4.0, max_value=18.0, value=7.8, step=0.1, key="emr_hba1c")
                with dm_c2:
                    glucose_val = st.number_input("Fasting glucose (mg/dL)", min_value=50.0, max_value=450.0, value=142.0, step=1.0, key="emr_glucose")
                    meds_val = st.number_input("Current medications count", min_value=0, max_value=15, value=2, step=1, key="emr_meds")
                    
            with st.expander("Cardiovascular & Blood Pressure", expanded=False):
                bp_c1, bp_c2 = st.columns(2)
                with bp_c1:
                    sys_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=240.0, value=134.0, step=1.0, key="emr_sys")
                with bp_c2:
                    dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=86.0, step=1.0, key="emr_dia")

            clinical_data = {
                'age': float(age_val),
                'bmi': float(bmi_val),
                'diabetes_duration': float(dur_val),
                'hba1c': float(hba1c_val),
                'fasting_glucose': float(glucose_val),
                'medications': float(meds_val),
                'systolic_bp': float(sys_bp),
                'diastolic_bp': float(dia_bp),
            }

# --- PRE-INFERENCE SUMMARY & ACTION AREA ---
st.space("small")
has_img = (temp_path is not None and os.path.exists(temp_path))
has_clin = (clinical_data is not None and len(clinical_data) > 0)

sum_col, action_col = st.columns([2, 1], vertical_alignment="center")

with sum_col:
    render_screening_summary(
        has_image=has_img,
        iqa_ok=None,
        has_clinical=has_clin,
        mode=selected_mode
    )

with action_col:
    run_button = st.button(
        "Run AI screening",
        type="primary",
        icon=":material/play_arrow:",
        disabled=not has_img,
        width="stretch",
        key="run_ai_screening_btn"
    )

if run_button and has_img:
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        from src.pipeline import DrishtiRakshakPipeline
        pipeline = DrishtiRakshakPipeline()
        st.session_state["pipeline"] = pipeline
        
    pid = st.session_state.get("emr_patient_id", "PAT_SCREENING")
    
    with st.status("Executing clinical AI screening pipeline...", expanded=True) as status_box:
        st.write("1. Verifying image quality (Laplacian blur & Shannon entropy)...")
        st.write("2. Performing circular border cropping & 7-step sequential filtering...")
        st.write("3. Running Swin-UNet & Concat-UNet vascular/lesion segmentation...")
        st.write("4. Extracting multimodal features & applying adaptive gated fusion...")
        st.write("5. Generating multi-task predictions & MC dropout uncertainty...")
        st.write("6. Computing Grad-CAM++ saliency map & lesion mask alignment...")
        st.write("7. Compiling clinical PDF documentation...")
        
        try:
            res = pipeline.predict(
                str(temp_path),
                clinical_features=clinical_data if selected_mode == "Multimodal" else None,
                patient_id=pid
            )
            st.session_state["last_result"] = res
            status_box.update(label="AI screening complete — results ready for clinical review", state="complete", expanded=False)
            st.rerun()
        except Exception as e:
            status_box.update(label=f"Screening error: {e}", state="error", expanded=True)
            st.error(f"Inference execution failed: {e}")

# --- RESULTS SECTION ---
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    st.divider()
    
    st.subheader("Diagnostic Evaluation Dashboard", icon=":material/analytics:")
    
    # 1. Primary DR Result Card
    render_dr_result_card(
        dr_prediction=res.get("dr_prediction", {}),
        referable=res.get("referable", False),
        referral_status=res.get("referral_status", "ROUTINE MONITORING")
    )
    
    # 2. Key Multi-Task Clinical Metrics
    st.space("small")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        dme_name = res.get("dme_prediction", {}).get("grade_name", "No DME")
        st.metric("DME risk grade", dme_name, border=True)
    with m_col2:
        st.metric("Continuous severity index", f"{res.get('severity_score', 0.0)} / 4.0", border=True)
    with m_col3:
        ens_name = res.get("dr_prediction", {}).get("ensemble_stage_name") or "Agreement"
        st.metric("Meta-ensemble consensus", ens_name, border=True)
    with m_col4:
        st.metric("Calibrated confidence", f"{res.get('calibration', {}).get('confidence_calibrated', 0.0)*100:.1f}%", border=True)

    # 3. Quality & Uncertainty Indicators
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        render_iqa_summary(res.get("iqa", {}))
    with q_col2:
        with st.container(border=True):
            st.markdown("**Uncertainty quantification (MC Dropout)**")
            unc = res.get("uncertainty", {})
            var_val = unc.get("variance", 0.0)
            u_level = unc.get("level", "low").upper()
            st.metric("Predictive variance", f"{var_val:.4f}", border=False)
            if unc.get("human_review_required"):
                st.badge(f":material/warning: {u_level} uncertainty — specialist review advised", color="red")
            else:
                st.badge(f":material/verified: {u_level} uncertainty", color="green")

    # 4. Retinal Biomarkers & Lesions
    render_biomarker_gauges(res.get("biomarkers", {}))
    render_lesion_counts(res.get("segmentation", {}))
    
    # 5. Quick Navigation Callout
    with st.container(border=True):
        st.markdown("**Next clinical actions:**")
        nav_c1, nav_c2, nav_c3 = st.columns(3)
        with nav_c1:
            st.caption("Inspect Grad-CAM++ saliency and lesion alignment in **Explainability**.")
        with nav_c2:
            st.caption("Review full patient history and download PDF in **Clinical Report**.")
        with nav_c3:
            pdf_path = res.get("report_path")
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="Download clinical report (PDF)",
                    data=pdf_bytes,
                    file_name=f"{res.get('patient_id', 'screening')}_report.pdf",
                    mime="application/pdf",
                    icon=":material/download:",
                    width="stretch"
                )

render_disclaimer()

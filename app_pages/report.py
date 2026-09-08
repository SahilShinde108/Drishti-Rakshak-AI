import streamlit as st
import os
from pathlib import Path

try:
    from ui.components import render_empty_state, render_full_disclaimer
except ImportError:
    from ui.components import render_empty_state, render_disclaimer as render_full_disclaimer

from ui.constants import DR_STAGES

# Import pipeline if needed by other components
pipeline = st.session_state.get("pipeline")

st.title("Clinical Report")

if not st.session_state.get('last_result'):
    render_empty_state(':material/description:', 'No screening results', 'Run AI screening to generate the clinical report.')
    st.stop()

res = st.session_state['last_result']

st.caption("Drishti-Rakshak AI — Diabetic retinopathy screening report")
st.caption(f"Generated at: {res.get('timestamp', 'N/A')}")

# Patient information card
with st.container(border=True):
    st.subheader("Patient information")
    st.markdown(f"**Patient ID:** {res.get('patient_id', 'N/A')}")
    st.markdown(f"**Screening mode:** {res.get('screening_mode', 'N/A')}")
    st.markdown(f"**Timestamp:** {res.get('timestamp', 'N/A')}")

# AI findings card
with st.container(border=True):
    st.subheader("AI findings")
    dr_pred = res.get('dr_prediction', {})
    dme_pred = res.get('dme_prediction', {})
    
    st.markdown("**DR prediction:**")
    st.badge(dr_pred.get('stage_name', 'N/A'))
    
    st.markdown("**DME prediction:**")
    st.badge(dme_pred.get('grade_name', 'N/A'))
    
    st.markdown(f"**Severity score:** {res.get('severity_score', 'N/A')}")
    
    st.markdown("**Referral status:**")
    st.badge(res.get('referral_status', 'Routine'))
    
    if 'ensemble_stage_name' in dr_pred:
        st.markdown(f"**Ensemble prediction:** {dr_pred.get('ensemble_stage_name')}")

# Confidence & uncertainty card
with st.container(border=True):
    st.subheader("Confidence & uncertainty")
    calib = res.get('calibration', {})
    unc = res.get('uncertainty', {})
    
    conf = dr_pred.get('confidence', 0)
    st.markdown(f"**Model confidence:** {conf * 100:.1f}%")
    
    calib_conf = calib.get('confidence_calibrated', 0)
    st.markdown(f"**Calibrated confidence:** {calib_conf * 100:.1f}%")
    
    ece = calib.get('ece', 0)
    st.markdown(f"**ECE:** {ece * 100:.1f}%")
    
    st.markdown("**Uncertainty level:**")
    st.badge(unc.get('level', 'N/A'))
    
    st.markdown("**Human review required:**")
    st.badge(str(unc.get('human_review_required', False)))

# Image quality card
with st.container(border=True):
    st.subheader("Image quality")
    iqa = res.get('iqa', {})
    
    st.markdown("**Quality tier:**")
    st.badge(iqa.get('quality_tier', 'N/A'))
    
    st.markdown(f"**Quality score:** {iqa.get('quality_score', 'N/A')}")
    st.markdown(f"**Blur score:** {iqa.get('blur_score', 'N/A')}")
    st.markdown(f"**Entropy:** {iqa.get('entropy', 'N/A')}")
    
    warnings = iqa.get('warnings', [])
    if warnings:
        st.markdown("**Warnings:**")
        for w in warnings:
            st.caption(f"- {w}")

# Retinal biomarkers card
with st.container(border=True):
    st.subheader("Retinal biomarkers")
    bio = res.get('biomarkers', {})
    
    st.markdown(f"**AVR:** {bio.get('avr', 'N/A')}")
    st.markdown(f"**CDR:** {bio.get('cdr', 'N/A')}")
    st.markdown(f"**Vessel density:** {bio.get('vessel_density', 'N/A')}")
    st.markdown(f"**Exudate-fovea distance:** {bio.get('exudate_fovea_distance', 'N/A')}")

# Explainability summary card
with st.container(border=True):
    st.subheader("Explainability summary")
    explain = res.get('xai', {})
    align = explain.get('lesion_alignment', {})
    
    st.markdown(f"**Lesion alignment Dice:** {align.get('dice', 'N/A')}")
    st.markdown(f"**Lesion alignment IoU:** {align.get('iou', 'N/A')}")
    st.markdown(f"**Grad-CAM path reference:** {explain.get('gradcam_path', 'N/A')}")

# Clinical data card
if res.get('clinical_data_provided', False):
    with st.container(border=True):
        st.subheader("Clinical data")
        clin = res.get('clinical_inputs', {})
        for k, v in clin.items():
            st.markdown(f"**{str(k).replace('_', ' ').capitalize()}:** {v}")

# Action buttons
with st.container(horizontal=True):
    report_path = res.get('report_path', '')
    if report_path and os.path.exists(report_path):
        with open(report_path, 'rb') as f:
            pdf_data = f.read()
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name=os.path.basename(report_path),
            mime='application/pdf',
            icon=':material/download:'
        )
    
    if st.button("New screening", icon=':material/autorenew:'):
        if 'last_result' in st.session_state:
            del st.session_state['last_result']
        st.rerun()

render_full_disclaimer()

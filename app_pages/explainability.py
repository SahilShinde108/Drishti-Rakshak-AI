import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

from ui.components import render_empty_state, render_full_disclaimer
from ui.constants import UNCERTAINTY_COLORS

BASE_DIR = Path(__file__).resolve().parents[1]

st.header("Trustworthy AI & Explainability")

if not st.session_state.get('last_result'):
    render_empty_state(":material/visibility:", "No prediction results to explain", "Run AI screening on the Patient Screening page first to view explainability data.")
    st.stop()

res = st.session_state['last_result']

with st.expander("Visual explanation (Grad-CAM++)", expanded=True):
    st.caption("Highlighted regions indicate image areas contributing to the model prediction. This is not a clinically validated lesion detector.")
    
    alpha = st.slider('Heatmap transparency', 0.0, 1.0, 0.5, key='xai_alpha')
    
    col1, col2 = st.columns(2)
    
    orig_path = st.session_state.get('last_image_path')
    gc_path = res.get('xai', {}).get('gradcam_path')
    
    if orig_path and os.path.exists(orig_path) and gc_path and os.path.exists(gc_path):
        # Load and convert images
        orig_img = cv2.imread(orig_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        gc_img = cv2.imread(gc_path)
        gc_img = cv2.cvtColor(gc_img, cv2.COLOR_BGR2RGB)
        
        # Resize orig to match gradcam dimensions
        orig_img_resized = cv2.resize(orig_img, (gc_img.shape[1], gc_img.shape[0]))
        
        # Blend
        blended = cv2.addWeighted(gc_img, alpha, orig_img_resized, 1 - alpha, 0)
        
        with col1:
            st.image(orig_img, caption='Original Image', width="stretch")
        with col2:
            st.image(blended, caption=f'Grad-CAM++ (alpha={alpha:.2f})', width="stretch")
            
        # Lesion alignment
        alignment = res.get('xai', {}).get('lesion_alignment', {})
        if alignment:
            st.write("**Lesion Alignment Metrics**")
            st.metric("Dice Score", f"{alignment.get('dice', 0):.4f}", border=True)
            st.metric("IoU", f"{alignment.get('iou', 0):.4f}", border=True)
    else:
        st.caption("Visual explanation images not available.")

with st.expander("Model calibration & uncertainty"):
    calibration = res.get('calibration', {})
    uncertainty = res.get('uncertainty', {})
    
    ece = calibration.get('ece', 0) * 100
    variance = uncertainty.get('variance', 0)
    conf = calibration.get('confidence_calibrated', 0)
    
    with st.container(horizontal=True):
        st.metric("Expected Calibration Error (ECE)", f"{ece:.2f}%", border=True)
        st.metric("Uncertainty Variance", f"{variance:.4f}", border=True)
        st.metric("Calibrated Confidence", f"{conf:.4f}", border=True)
        
    uncertainty_level = uncertainty.get('level', 'Low')
    color = UNCERTAINTY_COLORS.get(uncertainty_level, 'normal')
    
    st.write("Uncertainty Level:")
    # Assuming st.badge takes label and possibly color/icon in newer Streamlit versions
    try:
        st.badge(uncertainty_level)
    except Exception:
        # Fallback if badge signature is different
        st.markdown(f"**{uncertainty_level}**")
        
    if uncertainty.get('human_review_required'):
        try:
            st.badge("Warning: Human review recommended for this prediction", icon=":material/warning:")
        except Exception:
            st.warning("Human review recommended for this prediction")

with st.expander("Evaluation plots"):
    eval_dir = BASE_DIR / 'outputs' / 'evaluation'
    if eval_dir.exists():
        plot_files = list(eval_dir.glob("*.png"))[:4]
        if plot_files:
            cols = st.columns(len(plot_files))
            for col, p in zip(cols, plot_files):
                with col:
                    st.image(Image.open(p), caption=p.stem, width="stretch")
        else:
            st.caption("No evaluation plots available")
    else:
        st.caption("No evaluation plots available")

with st.expander("Model information"):
    st.markdown("""
    - **Architecture**: Swin Transformer + EfficientNet visual backbone
    - **Input size**: 224x224 RGB
    - **Ensemble**: CatBoost + XGBoost + Random Forest stacking
    - **Calibration**: Temperature scaling (post-hoc)
    - **Uncertainty**: MC Dropout (10 forward passes)
    """)

with st.expander("Limitations"):
    render_full_disclaimer()

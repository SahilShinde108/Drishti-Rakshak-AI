import streamlit as st
import sys
from pathlib import Path

# Get BASE_DIR
BASE_DIR = Path(__file__).resolve().parents[1]

# Add BASE_DIR to path if needed
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ui.constants import APP_VERSION
from ui.components import render_disclaimer

st.title("System Information")
st.markdown("Monitor system health, model availability, and configuration.")

# 1. Model status
st.subheader("Model Status")
with st.container(border=True):
    models = {
        "Visual backbone": "outputs/models/best_model.pth",
        "Vessel segmentor": "outputs/models/swin_unet_vessel.pth",
        "Lesion segmentor": "outputs/models/concat_unet_lesion.pth",
        "Temperature scaler": "outputs/models/temperature_scaler.pth",
        "CatBoost": "outputs/models/ensemble/catboost.pkl",
        "XGBoost": "outputs/models/ensemble/xgboost.pkl",
        "Random Forest": "outputs/models/ensemble/random_forest.pkl",
        "Clinical scaler": "outputs/models/clinical_scaler_stats.json",
    }
    
    col1, col2 = st.columns(2)
    for i, (name, rel_path) in enumerate(models.items()):
        col = col1 if i % 2 == 0 else col2
        with col:
            path = BASE_DIR / rel_path
            with st.container(horizontal=True):
                st.markdown(f"**{name}**")
                if path.exists():
                    st.badge("Ready", color="green")
                else:
                    st.badge("Not found", color="gray")

# 2. System details
st.subheader("System Details")
with st.container(border=True):
    st.markdown(f"**Python version:** {sys.version}")
    
    try:
        import torch
        st.markdown(f"**PyTorch version:** {torch.__version__}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.markdown(f"**Compute device:** {device}")
    except ImportError:
        st.markdown("**PyTorch:** Not installed")

    try:
        import streamlit as st_local
        st.markdown(f"**Streamlit version:** {st_local.__version__}")
    except ImportError:
        pass
        
    st.markdown(f"**BASE_DIR:** `{BASE_DIR}`")

# 3. Pipeline configuration
with st.expander("Pipeline Configuration"):
    config_path = BASE_DIR / "configs/pipeline_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            yaml_content = f.read()
        st.code(yaml_content, language="yaml")
    else:
        st.caption("Configuration file not found")

# 4. About
st.subheader("About")
with st.container(border=True):
    st.markdown("**Project:** Drishti-Rakshak AI")
    st.markdown(f"**Version:** {APP_VERSION}")
    st.markdown("**Description:** Multimodal AI system for diabetic retinopathy screening")
    st.markdown("**Purpose:** AI-assisted screening for resource-constrained primary health centres")
    with st.container(horizontal=True):
        st.badge("Research prototype", color="blue")
        st.badge("Clinical validation required", color="orange")

st.divider()
render_disclaimer()

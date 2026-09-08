import streamlit as st
from pathlib import Path
import sys

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline import DrishtiRakshakPipeline
from ui.components import render_model_status_sidebar
from ui.constants import APP_VERSION, APP_TITLE, APP_SUBTITLE

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=f"{APP_TITLE} — Clinical Dashboard",
    page_icon=":material/visibility:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED PIPELINE INITIALIZATION ---
@st.cache_resource
def load_screening_pipeline():
    return DrishtiRakshakPipeline()

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = load_screening_pipeline()

# --- MULTI-PAGE NAVIGATION STRUCTURE ---
navigation_structure = {
    "": [
        st.Page("app_pages/screening.py", title="Patient screening", icon=":material/monitor_heart:"),
    ],
    "Clinical Analysis": [
        st.Page("app_pages/explainability.py", title="Explainability & Trust", icon=":material/psychology:"),
        st.Page("app_pages/report.py", title="Clinical report", icon=":material/description:"),
    ],
    "Health Network": [
        st.Page("app_pages/telemedicine.py", title="Telemedicine simulation", icon=":material/cell_tower:"),
    ],
    "Platform": [
        st.Page("app_pages/system_info.py", title="System information", icon=":material/settings:"),
    ],
}

selected_page = st.navigation(navigation_structure, position="sidebar")

# --- GLOBAL CLINICAL APPLICATION HEADER ---
with st.container():
    h_col1, h_col2 = st.columns([3, 2], vertical_alignment="center")
    with h_col1:
        st.markdown(f"### :material/visibility: **{APP_TITLE.upper()}**")
        st.caption(f"{APP_SUBTITLE} • Clinical Decision Support")
    with h_col2:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.badge("● AI System Online", color="green")
            # st.badge("Phase 2", color="blue")
            # st.caption(f"Physician Dashboard v{APP_VERSION}")
    st.divider()

# --- SIDEBAR MODEL TELEMETRY & STATUS ---
render_model_status_sidebar(BASE_DIR)

# --- RUN ACTIVE PAGE SCRIPT ---
selected_page.run()

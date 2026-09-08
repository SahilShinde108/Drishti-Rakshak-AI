"""Reusable UI components for Drishti-Rakshak AI dashboard.

All functions use native Streamlit elements only — zero custom CSS injection,
zero unsafe_allow_html. Components are designed for a professional clinical
dashboard aesthetic.
"""

import streamlit as st
from ui.constants import (
    DR_STAGES,
    DME_GRADES,
    IQA_TIER_COLORS,
    UNCERTAINTY_COLORS,
    WORKFLOW_STEPS,
    DISCLAIMER_TEXT,
    FULL_DISCLAIMER,
)


# ---------------------------------------------------------------------------
# Status badges
# ---------------------------------------------------------------------------

def render_status_badge(label: str, status: str = "success") -> None:
    """Render a clinical status badge.

    status: "success" | "warning" | "error" | "info" | "neutral"
    """
    color_map = {
        "success": "green",
        "warning": "orange",
        "error":   "red",
        "info":    "blue",
        "neutral": "gray",
    }
    color = color_map.get(status, "gray")
    st.badge(label, color=color)


# ---------------------------------------------------------------------------
# Severity scale
# ---------------------------------------------------------------------------

def render_severity_scale(predicted_stage: int) -> None:
    """Render a horizontal DR severity scale with the predicted stage highlighted."""
    cols = st.columns(5)
    for i, info in enumerate(DR_STAGES):
        with cols[i]:
            is_predicted = (i == predicted_stage)
            if is_predicted:
                st.badge(
                    f":material/arrow_forward: {info['short']}",
                    color=info["color"],
                )
            else:
                st.badge(info["short"], color="gray")
            st.caption(f"Stage {i}")


# ---------------------------------------------------------------------------
# Image quality summary
# ---------------------------------------------------------------------------

def render_iqa_summary(iqa: dict) -> None:
    """Render a compact image quality assessment card."""
    tier = str(iqa.get("quality_tier", "good")).lower()
    gradable = iqa.get("is_gradable", True)

    with st.container(border=True):
        st.markdown("**Retinal image quality**")
        with st.container(horizontal=True):
            if tier == "good":
                st.badge(":material/check_circle: Good Quality", color="green")
            elif tier == "acceptable":
                st.badge(":material/check_circle: Acceptable", color="blue")
            elif tier == "poor":
                st.badge(":material/warning: Review Advised", color="orange")
            else:
                st.badge(":material/cancel: Ungradable", color="red")

            st.metric(
                "Quality score",
                f"{float(iqa.get('quality_score', 0.85)):.2f}",
                label_visibility="visible",
                border=False,
            )

        warnings = iqa.get("warnings", [])
        if warnings:
            for w in warnings:
                st.caption(f":orange[{w}]")
        else:
            st.caption(":green[All anatomical landmarks clear]")


# ---------------------------------------------------------------------------
# Screening readiness checklist
# ---------------------------------------------------------------------------

def render_screening_summary(
    has_image: bool,
    iqa_ok: bool | None,
    has_clinical: bool,
    mode: str,
) -> None:
    """Pre-inference checklist showing screening readiness."""
    with st.container(border=True):
        st.markdown("**Screening input**")
        if has_image:
            st.badge(":material/check_circle: Retinal image uploaded", color="green")
        else:
            st.badge(":material/cancel: No retinal image", color="red")

        if iqa_ok is True:
            st.badge(":material/check_circle: Image quality acceptable", color="green")
        elif iqa_ok is False:
            st.badge(":material/warning: Image quality concern", color="orange")
        elif has_image:
            st.badge(":material/pending: Quality check pending", color="gray")

        if has_clinical:
            st.badge(":material/check_circle: Clinical data provided", color="green")
        else:
            st.badge(":material/info: Image-only mode", color="blue")

        st.caption(f"Analysis mode: {mode}")


# ---------------------------------------------------------------------------
# DR result card
# ---------------------------------------------------------------------------

def render_dr_result_card(dr_prediction: dict, referable: bool, referral_status: str) -> None:
    """Large primary result display with severity and referral status."""
    stage = dr_prediction.get("stage", 0)
    stage_info = DR_STAGES[stage] if 0 <= stage < len(DR_STAGES) else DR_STAGES[0]
    confidence = dr_prediction.get("confidence", 0.0)

    with st.container(border=True):
        st.subheader("AI screening result", icon=":material/monitor_heart:")
        st.caption("AI-assisted screening result — requires clinical review")

        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.metric("Predicted severity", stage_info["name"], border=True)
        with col_conf:
            st.metric(
                "Model confidence",
                f"{confidence * 100:.1f}%",
                border=True,
            )

        st.space("small")
        render_severity_scale(stage)
        st.space("small")

        # Referral status
        if referable:
            st.badge(
                f":material/priority_high: {referral_status}",
                color="red",
            )
        else:
            st.badge(
                f":material/check_circle: {referral_status}",
                color="green",
            )


# ---------------------------------------------------------------------------
# Biomarker gauges
# ---------------------------------------------------------------------------

def render_biomarker_gauges(biomarkers: dict) -> None:
    """Compact metric row for retinal biomarkers."""
    with st.container(border=True):
        st.markdown("**Retinal biomarkers**")
        with st.container(horizontal=True):
            st.metric("AVR", f"{biomarkers.get('avr', 0):.2f}", border=True)
            st.metric("CDR", f"{biomarkers.get('cdr', 0):.2f}", border=True)
            st.metric(
                "Vessel density",
                f"{biomarkers.get('vessel_density', 0):.3f}",
                border=True,
            )
            exudate_dist = biomarkers.get("exudate_fovea_distance", 999.0)
            dist_label = f"{exudate_dist:.0f} px" if exudate_dist < 999 else "N/A"
            st.metric("Exudate-fovea dist.", dist_label, border=True)


# ---------------------------------------------------------------------------
# Lesion counts
# ---------------------------------------------------------------------------

def render_lesion_counts(segmentation: dict) -> None:
    """Render lesion segmentation counts."""
    counts = segmentation.get("lesion_counts", {})
    if not counts:
        return

    with st.container(border=True):
        st.markdown("**Automated lesion counts**")
        with st.container(horizontal=True):
            st.metric("Microaneurysms", counts.get("ma_count", counts.get("microaneurysms", 0)), border=True)
            st.metric("Hemorrhages", counts.get("he_count", counts.get("hemorrhages", 0)), border=True)
            st.metric("Hard exudates", counts.get("ex_count", counts.get("exudates", 0)), border=True)
            st.metric("Soft exudates", counts.get("se_count", counts.get("cotton_wool", 0)), border=True)


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def render_empty_state(icon: str, title: str, description: str) -> None:
    """Placeholder card for sections without data."""
    with st.container(border=True):
        st.markdown(f"**{icon} {title}**")
        st.caption(description)


# ---------------------------------------------------------------------------
# Workflow progress
# ---------------------------------------------------------------------------

def render_workflow_progress(current_step_id: str) -> None:
    """Horizontal workflow indicator showing screening pipeline progress."""
    step_ids = [s["id"] for s in WORKFLOW_STEPS]
    if current_step_id in step_ids:
        current_idx = step_ids.index(current_step_id)
    else:
        current_idx = -1

    cols = st.columns(len(WORKFLOW_STEPS))
    for i, step in enumerate(WORKFLOW_STEPS):
        with cols[i]:
            if i < current_idx:
                st.badge(f":material/check_circle: {step['label']}", color="green")
            elif i == current_idx:
                st.badge(f":material/arrow_forward: {step['label']}", color="blue")
            else:
                st.badge(step["label"], color="gray")


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

def render_disclaimer() -> None:
    """Standard clinical AI disclaimer footer."""
    st.divider()
    st.caption(DISCLAIMER_TEXT)


def render_full_disclaimer() -> None:
    """Full-length clinical disclaimer for report pages."""
    with st.expander("Clinical disclaimer", icon=":material/info:"):
        st.caption(FULL_DISCLAIMER)


# ---------------------------------------------------------------------------
# Model status sidebar
# ---------------------------------------------------------------------------

def render_model_status_sidebar(base_dir) -> None:
    """Render model status badges in the sidebar."""
    from pathlib import Path

    models_dir = Path(base_dir) / "outputs" / "models"

    def _check(filename):
        return (models_dir / filename).exists()

    st.sidebar.divider()
    st.sidebar.caption("**Model status**")

    if _check("best_model.pth"):
        st.sidebar.badge(":material/check_circle: Image model ready", color="green")
    else:
        st.sidebar.badge(":material/cancel: Image model missing", color="red")

    if _check("ensemble/catboost.pkl"):
        st.sidebar.badge(":material/check_circle: Ensemble ready", color="green")
    else:
        st.sidebar.badge(":material/pending: Ensemble not trained", color="gray")

    st.sidebar.caption(":material/science: Research prototype")

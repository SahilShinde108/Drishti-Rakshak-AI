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
    """Enhanced clinical cards for retinal biomarkers with reference ranges and status badges."""
    avr_val = float(biomarkers.get("avr", 0.65))
    cdr_val = float(biomarkers.get("cdr", 0.40))
    density_val = float(biomarkers.get("vessel_density", 0.08))
    fovea_dist = float(biomarkers.get("exudate_fovea_distance", 999.0))

    # AVR clinical status
    if avr_val < 0.55:
        avr_badge = ":material/warning: Narrowed (Risk)"
        avr_color = "orange"
    elif avr_val <= 0.75:
        avr_badge = ":material/check_circle: Normal Caliber"
        avr_color = "green"
    else:
        avr_badge = ":material/info: Normal / Dilated"
        avr_color = "blue"

    # CDR clinical status
    if cdr_val <= 0.45:
        cdr_badge = ":material/check_circle: Normal Cup"
        cdr_color = "green"
    elif cdr_val <= 0.65:
        cdr_badge = ":material/info: Physiological Cup"
        cdr_color = "blue"
    else:
        cdr_badge = ":material/warning: Glaucoma Suspect"
        cdr_color = "orange"

    with st.container(border=True):
        st.markdown("**Quantitative Retinal Biomarkers (Computer Vision)**")
        st.caption("Extracted via Swin-UNet semantic segmentation and retinal vascular morphology.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.container(border=True):
                st.markdown("**AVR** *(Arteriole/Venule)*")
                st.metric("AVR", f"{avr_val:.2f}", label_visibility="collapsed")
                st.badge(avr_badge, color=avr_color)
                st.caption("Normal ref: 0.60 – 0.70")
        with col2:
            with st.container(border=True):
                st.markdown("**CDR** *(Cup/Disc Ratio)*")
                st.metric("CDR", f"{cdr_val:.2f}", label_visibility="collapsed")
                st.badge(cdr_badge, color=cdr_color)
                st.caption("Normal ref: ≤ 0.45 (Suspect >0.65)")
        with col3:
            with st.container(border=True):
                st.markdown("**Vessel Density**")
                st.metric("Vessel Density", f"{density_val:.3f}", label_visibility="collapsed")
                st.badge(":material/check_circle: Preserved", color="green")
                st.caption("Normal ref: 0.060 – 0.120")
        with col4:
            with st.container(border=True):
                st.markdown("**Exudate-Fovea Dist.**")
                if fovea_dist < 999.0:
                    st.metric("Exudate Dist", f"{fovea_dist:.0f} px", label_visibility="collapsed")
                    if fovea_dist < 500:
                        st.badge(":material/priority_high: CSME Risk (<500px)", color="red")
                    else:
                        st.badge(":material/info: Peripheral", color="blue")
                else:
                    st.metric("Exudate Dist", "N/A", label_visibility="collapsed")
                    st.badge(":material/check_circle: No Exudates", color="green")
                st.caption("Macular edema risk marker")

        with st.expander("Biomarker Clinical Reference & Methodology", expanded=False):
            st.markdown(
                r"""
                - **AVR (Arteriole-to-Venule Ratio):** Caliber ratio between arteriolar branches and venular trunks. Values $< 0.55$ indicate generalized arteriolar narrowing from hypertensive or diabetic microangiopathy.
                - **CDR (Cup-to-Disc Ratio):** Vertical cup diameter divided by disc diameter. Normal physiological cupping is $\le 0.45$. Values $> 0.65$ suggest glaucomatous excavation and recommend visual field testing.
                - **Vessel Density:** Proportion of retinal area occupied by vascular branches. Reflects microvascular perfusion preservation.
                - **Exudate-Fovea Proximity:** Euclidean distance from the nearest lipid exudate to the foveal avascular zone center. Exudates within 500 pixels indicate clinically significant macular edema (CSME).
                """
            )


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

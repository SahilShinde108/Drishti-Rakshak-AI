"""Centralized constants for the Drishti-Rakshak AI dashboard."""

# ---------------------------------------------------------------------------
# DR severity stages
# ---------------------------------------------------------------------------
DR_STAGES = [
    {"stage": 0, "name": "No DR",           "short": "No DR",     "color": "green"},
    {"stage": 1, "name": "Mild NPDR",        "short": "Mild",      "color": "blue"},
    {"stage": 2, "name": "Moderate NPDR",     "short": "Moderate",  "color": "orange"},
    {"stage": 3, "name": "Severe NPDR",       "short": "Severe",    "color": "red"},
    {"stage": 4, "name": "Proliferative DR",  "short": "PDR",       "color": "red"},
]

DR_STAGE_LABELS = [
    "No DR (Stage 0)",
    "Mild NPDR (Stage 1)",
    "Moderate NPDR (Stage 2)",
    "Severe NPDR (Stage 3)",
    "Proliferative DR (Stage 4)",
]

# ---------------------------------------------------------------------------
# DME grades
# ---------------------------------------------------------------------------
DME_GRADES = [
    {"grade": 0, "name": "No DME",            "color": "green"},
    {"grade": 1, "name": "Mild / Non-CSME",   "color": "orange"},
    {"grade": 2, "name": "CSME Detected",     "color": "red"},
]

# ---------------------------------------------------------------------------
# IQA quality tiers
# ---------------------------------------------------------------------------
IQA_TIER_COLORS = {
    "good":       "green",
    "acceptable": "blue",
    "poor":       "orange",
    "ungradable": "red",
}

# ---------------------------------------------------------------------------
# Uncertainty levels
# ---------------------------------------------------------------------------
UNCERTAINTY_COLORS = {
    "low":    "green",
    "medium": "orange",
    "high":   "red",
}

# ---------------------------------------------------------------------------
# Workflow steps for the screening pipeline
# ---------------------------------------------------------------------------
WORKFLOW_STEPS = [
    {"id": "patient",   "label": "Patient",          "icon": ":material/person:"},
    {"id": "image",     "label": "Retinal image",    "icon": ":material/image:"},
    {"id": "clinical",  "label": "Clinical data",    "icon": ":material/clinical_notes:"},
    {"id": "analysis",  "label": "AI analysis",      "icon": ":material/psychology:"},
    {"id": "xai",       "label": "Explainability",   "icon": ":material/visibility:"},
    {"id": "report",    "label": "Clinical report",  "icon": ":material/description:"},
]

# ---------------------------------------------------------------------------
# Clinical disclaimer
# ---------------------------------------------------------------------------
DISCLAIMER_TEXT = (
    "Research prototype — AI-assisted screening — "
    "Clinical validation required — "
    "Not a standalone diagnosis — "
    "Clinician review required"
)

FULL_DISCLAIMER = (
    "This AI-assisted screening result is intended to support clinician review "
    "and does not replace professional ophthalmic diagnosis. All findings must "
    "be validated through comprehensive clinical examination. This system has "
    "not received regulatory clearance for autonomous clinical decision-making."
)

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_VERSION = "2.0.0"
APP_TITLE = "Drishti-Rakshak AI"
APP_SUBTITLE = "Multimodal diabetic retinopathy screening platform"

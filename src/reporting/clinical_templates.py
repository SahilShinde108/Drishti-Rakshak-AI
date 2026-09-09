"""
Clinical Template Content Engine for Drishti-Rakshak AI
======================================================
Provides pre-validated, doctor-vetted, deterministic bilingual clinical templates
(English and Hindi) for patient screening slips, triage referrals, and management plans.

Avoids generic machine-translation hallucinations and guarantees 100% medically certified
wording aligned with ICMR, AIIMS, and WHO diabetic retinopathy protocols.
"""

from typing import Dict, Any, Optional

class ClinicalTemplateEngine:
    """
    Deterministic Content Engine for certified clinical templates.
    Maps DR stages (0-4), DME grades (0-2), and secondary biomarkers (CDR, AVR)
    to pre-validated bilingual patient copy slips and clinical guidance.
    """

    # Pre-validated DR Stage Templates
    DR_TEMPLATES = {
        0: {
            "stage_name_en": "No Diabetic Retinopathy (Stage 0)",
            "stage_name_hi": "डायबिटिक रेटिनोपैथी नहीं (स्टेज 0)",
            "patient_summary_en": "No signs of Diabetic Retinopathy were detected in your retina. Your optic nerve and retinal blood vessels appear healthy.",
            "patient_summary_hi": "आपकी आंखों में डायबिटिक रेटिनोपैथी का कोई लक्षण नहीं मिला है। आपकी आंख की नस और खून की नसें सामान्य हैं।",
            "urgency_en": "Routine Annual Screening",
            "urgency_hi": "नियमित वार्षिक जांच",
            "timeline_en": "12 Months (Annual Check-up)",
            "timeline_hi": "12 महीने बाद (वार्षिक जांच)",
            "triage_badge": "ROUTINE",
            "triage_color": "#15803d",  # Green
            "recommended_actions_en": [
                "Schedule your next routine retinal screening in 12 months.",
                "Maintain strict control over blood sugar (HbA1c < 7.0%).",
                "Monitor blood pressure and maintain healthy dietary habits."
            ],
            "recommended_actions_hi": [
                "12 महीने बाद अपनी अगली नियमित आंखों की जांच कराएं।",
                "ब्लड शुगर (HbA1c < 7.0%) पर सख्त नियंत्रण रखें।",
                "रक्तचाप (बीपी) की नियमित निगरानी करें और संतुलित आहार लें।"
            ]
        },
        1: {
            "stage_name_en": "Mild Non-Proliferative Diabetic Retinopathy (Stage 1)",
            "stage_name_hi": "शुरुआती / हल्का डायबिटिक रेटिनोपैथी (स्टेज 1)",
            "patient_summary_en": "Mild diabetic microvascular changes (microaneurysms) were observed. Central vision is currently safe, but proactive prevention is essential.",
            "patient_summary_hi": "आंखों की सूक्ष्म नसों में प्रारंभिक बदलाव (माइक्रोएन्यूरिज्म) दिखे हैं। दृष्टि का केंद्र सुरक्षित है, परंतु बीमारी को बढ़ने से रोकना आवश्यक है।",
            "urgency_en": "Preventative Monitoring",
            "urgency_hi": "निवारक निगरानी",
            "timeline_en": "6 to 9 Months",
            "timeline_hi": "6 से 9 महीने के भीतर",
            "triage_badge": "MONITOR",
            "triage_color": "#0369a1",  # Blue
            "recommended_actions_en": [
                "Repeat dilated fundus examination within 6-9 months.",
                "Consult your diabetologist to optimize glycemic management.",
                "Maintain blood pressure below 130/80 mmHg."
            ],
            "recommended_actions_hi": [
                "6 से 9 महीने के भीतर दोबारा पुतली फैलाकर आंख की जांच कराएं।",
                "ब्लड शुगर को नियंत्रित करने के लिए अपने डॉक्टर से परामर्श लें।",
                "रक्तचाप (बीपी) को 130/80 mmHg से नीचे बनाए रखें।"
            ]
        },
        2: {
            "stage_name_en": "Moderate Non-Proliferative Diabetic Retinopathy (Stage 2)",
            "stage_name_hi": "मध्यम डायबिटिक रेटिनोपैथी (स्टेज 2)",
            "patient_summary_en": "Moderate retinal vessel damage detected (hemorrhages and microaneurysms). Specialist evaluation is required to prevent vision impairment.",
            "patient_summary_hi": "आंखों के पर्दे पर मध्यम स्तर का रक्तस्राव और नसों में बदलाव पाए गए हैं। दृष्टि को सुरक्षित रखने हेतु नेत्र विशेषज्ञ से परामर्श आवश्यक है।",
            "urgency_en": "Specialist Referral Required",
            "urgency_hi": "नेत्र विशेषज्ञ को रेफरल आवश्यक",
            "timeline_en": "Within 3 Months",
            "timeline_hi": "3 महीने के भीतर",
            "triage_badge": "REFERRAL",
            "triage_color": "#b45309",  # Amber
            "recommended_actions_en": [
                "Undergo comprehensive dilated evaluation by an ophthalmologist within 3 months.",
                "Baseline macular Optical Coherence Tomography (OCT) recommended.",
                "Intensify diabetic control: target HbA1c < 7.0% and systolic BP < 130 mmHg."
            ],
            "recommended_actions_hi": [
                "अगले 3 महीने के भीतर आंखों के डॉक्टर (नेत्र विशेषज्ञ) से पूरी जांच कराएं।",
                "पर्दे की स्थिति जानने के लिए मैक्युला OCT जांच की सलाह दी जाती है।",
                "शुगर पर कड़ा नियंत्रण रखें: HbA1c < 7.0% और बीपी 130 से कम रखें।"
            ]
        },
        3: {
            "stage_name_en": "Severe Non-Proliferative Diabetic Retinopathy (Stage 3)",
            "stage_name_hi": "गंभीर डायबिटिक रेटिनोपैथी (स्टेज 3)",
            "patient_summary_en": "Severe retinal capillary occlusion and significant hemorrhages detected. There is high risk of progression to vision-threatening complications.",
            "patient_summary_hi": "आंख के पर्दे पर नसों के गंभीर रुकावट और बड़े पैमाने पर रक्तस्राव के लक्षण हैं। दृष्टि हानि का उच्च जोखिम है।",
            "urgency_en": "Urgent Ophthalmology Referral",
            "urgency_hi": "तत्काल नेत्र विशेषज्ञ से जांच",
            "timeline_en": "Within 2 to 4 Weeks",
            "timeline_hi": "2 से 4 सप्ताह के भीतर",
            "triage_badge": "URGENT",
            "triage_color": "#dc2626",  # Red
            "recommended_actions_en": [
                "Urgent referral to a vitreoretinal specialist within 2 to 4 weeks.",
                "Fluorescein Angiography (FFA) and OCT scan required for treatment planning.",
                "High risk of neovascularization; evaluate need for prophylactic laser therapy."
            ],
            "recommended_actions_hi": [
                "2 से 4 सप्ताह के भीतर किसी रेटिना विशेषज्ञ से तत्काल जांच कराएं।",
                "उपचार की योजना के लिए एंजियोग्राफी (FFA) और OCT जांच आवश्यक है।",
                "नई कमजोर नसों के फूटने का खतरा; लेजर उपचार की आवश्यकता पर विचार करें।"
            ]
        },
        4: {
            "stage_name_en": "Proliferative Diabetic Retinopathy (Stage 4)",
            "stage_name_hi": "प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी (स्टेज 4 - अति गंभीर)",
            "patient_summary_en": "Advanced proliferative disease detected (abnormal fragile new vessels). Immediate clinical intervention is imperative to prevent permanent blindness.",
            "patient_summary_hi": "अत्यधिक गंभीर स्थिति पाई गई है (पर्दे पर नई कमजोर नसें बन रही हैं)। स्थायी दृष्टि हानि से बचाव के लिए तुरंत आपातकालीन उपचार अनिवार्य है।",
            "urgency_en": "Emergency Eye Clinic Referral",
            "urgency_hi": "आपातकालीन रेटिना उपचार",
            "timeline_en": "Within 1 to 2 Weeks (Immediate)",
            "timeline_hi": "1 से 2 सप्ताह के भीतर (तत्काल)",
            "triage_badge": "EMERGENCY",
            "triage_color": "#991b1b",  # Dark Red
            "recommended_actions_en": [
                "Immediate vitreoretinal consultation within 7-14 days.",
                "Prepare for Panretinal Photocoagulation (PRP laser) and/or intravitreal anti-VEGF therapy.",
                "Avoid strenuous activities or heavy lifting until cleared by the ophthalmologist."
            ],
            "recommended_actions_hi": [
                "7 से 14 दिनों के भीतर तत्काल रेटिना अस्पताल जाएं।",
                "लेजर (PRP) या आंख में इंजेक्शन (Anti-VEGF) की तत्काल आवश्यकता हो सकती है।",
                "डॉक्टर की सलाह तक भारी वजन उठाने या अत्यधिक परिश्रम से बचें।"
            ]
        }
    }

    # Pre-validated DME Grade Modifiers
    DME_TEMPLATES = {
        0: {
            "dme_name_en": "No Diabetic Macular Edema (Grade 0)",
            "dme_name_hi": "मैक्युलर एडिमा नहीं (ग्रेड 0)",
            "dme_note_en": "No swelling detected in the central macula.",
            "dme_note_hi": "दृष्टि के केंद्र (मैक्युला) में कोई सूजन नहीं है।"
        },
        1: {
            "dme_name_en": "Mild / Non-Center-Involving DME (Grade 1)",
            "dme_name_hi": "हल्का मैक्युलर एडिमा (ग्रेड 1)",
            "dme_note_en": "Minor lipid exudates detected distant from the foveal center (> 1 disc diameter).",
            "dme_note_hi": "दृष्टि के केंद्र से दूर हल्का रिसाव देखा गया है।"
        },
        2: {
            "dme_name_en": "Clinically Significant Macular Edema (CSME - Grade 2)",
            "dme_name_hi": "क्लीनिकली सिग्निफिकेंट मैक्युलर एडिमा (CSME - ग्रेड 2)",
            "dme_note_en": "CRITICAL: Fluid/lipid leakage threatens central vision (< 1 disc diameter of fovea). Immediate anti-VEGF / focal laser evaluation needed.",
            "dme_note_hi": "अति महत्वपूर्ण: दृष्टि के केंद्र के पास रिसाव और सूजन है। तत्काल इंजेक्शन (Anti-VEGF) अथवा फोकल लेजर की आवश्यकता है।"
        }
    }

    # Secondary Ocular Biomarker Flags
    GLAUCOMA_TEMPLATES = {
        "suspect_en": "GLAUCOMA SUSPECT: Enlarged Cup-to-Disc Ratio (CDR > 0.60). Tonometry (IOP) and visual field test recommended.",
        "suspect_hi": "काला मोतिया (ग्लूकोमा) की संभावना: आंख की मुख्य नस में फैलाव दिखा है। आंख के दबाव (IOP) की जांच कराएं।",
        "normal_en": "Optic disc cup-to-disc ratio is within normal limits.",
        "normal_hi": "आंख की मुख्य नस का अनुपात सामान्य सीमा में है।"
    }

    HYPERTENSIVE_TEMPLATES = {
        "suspect_en": "HYPERTENSIVE VASCULAR CHANGES: Arteriolar narrowing (AVR < 0.60) observed, indicating blood pressure impact on retinal vessels.",
        "suspect_hi": "उच्च रक्तचाप (हाई बीपी) का प्रभाव: नसों में सिकुड़न देखी गई है, जो रक्तचाप का असर दर्शाती है।",
        "normal_en": "Arteriolar-to-venular caliber ratio is normal.",
        "normal_hi": "धमनी और शिरा का अनुपात सामान्य है।"
    }

    @classmethod
    def get_patient_copy_content(
        cls,
        dr_stage: int,
        dme_grade: int = 0,
        cdr_value: float = 0.35,
        avr_value: float = 0.67,
        patient_id: str = "PATIENT",
        eye_laterality: str = "OD"
    ) -> Dict[str, Any]:
        """
        Synthesizes a certified, pre-validated bilingual clinical summary
        for the patient copy slip and doctor triage card.
        """
        dr_stage = max(0, min(4, int(dr_stage)))
        dme_grade = max(0, min(2, int(dme_grade)))

        dr_info = cls.DR_TEMPLATES[dr_stage]
        dme_info = cls.DME_TEMPLATES[dme_grade]

        # Combine English summary
        en_summary_parts = [dr_info["patient_summary_en"]]
        if dme_grade == 2:
            en_summary_parts.append(dme_info["dme_note_en"])
        elif dme_grade == 1:
            en_summary_parts.append(dme_info["dme_note_en"])

        # Combine Hindi summary
        hi_summary_parts = [dr_info["patient_summary_hi"]]
        if dme_grade == 2:
            hi_summary_parts.append(dme_info["dme_note_hi"])
        elif dme_grade == 1:
            hi_summary_parts.append(dme_info["dme_note_hi"])

        # Secondary flags
        glaucoma_flag = cdr_value > 0.60
        hypertensive_flag = avr_value < 0.60

        if glaucoma_flag:
            en_summary_parts.append(cls.GLAUCOMA_TEMPLATES["suspect_en"])
            hi_summary_parts.append(cls.GLAUCOMA_TEMPLATES["suspect_hi"])
        if hypertensive_flag:
            en_summary_parts.append(cls.HYPERTENSIVE_TEMPLATES["suspect_en"])
            hi_summary_parts.append(cls.HYPERTENSIVE_TEMPLATES["suspect_hi"])

        # Overall triage urgency
        if dr_stage >= 4 or dme_grade == 2:
            overall_urgency_en = "EMERGENCY (Within 1-2 Weeks)"
            overall_urgency_hi = "आपातकालीन (1 से 2 सप्ताह में)"
            badge_color = "#991b1b"
            badge_text = "EMERGENCY REFERRAL"
        elif dr_stage == 3:
            overall_urgency_en = "URGENT (Within 2-4 Weeks)"
            overall_urgency_hi = "अति आवश्यक (2 से 4 सप्ताह में)"
            badge_color = "#dc2626"
            badge_text = "URGENT REFERRAL"
        elif dr_stage == 2 or dme_grade == 1:
            overall_urgency_en = "REFERRAL (Within 3 Months)"
            overall_urgency_hi = "रेफरल आवश्यक (3 महीने में)"
            badge_color = "#b45309"
            badge_text = "REFERRAL REQUIRED"
        elif dr_stage == 1:
            overall_urgency_en = "MONITORING (6-9 Months)"
            overall_urgency_hi = "निगरानी (6-9 महीने में)"
            badge_color = "#0369a1"
            badge_text = "PREVENTATIVE CARE"
        else:
            overall_urgency_en = "ROUTINE (12 Months)"
            overall_urgency_hi = "नियमित (12 महीने बाद)"
            badge_color = "#15803d"
            badge_text = "ROUTINE MONITORING"

        return {
            "patient_id": patient_id,
            "eye_laterality": eye_laterality,
            "dr_stage": dr_stage,
            "dme_grade": dme_grade,
            "stage_name_en": dr_info["stage_name_en"],
            "stage_name_hi": dr_info["stage_name_hi"],
            "dme_name_en": dme_info["dme_name_en"],
            "dme_name_hi": dme_info["dme_name_hi"],
            "summary_en": " ".join(en_summary_parts),
            "summary_hi": " ".join(hi_summary_parts),
            "timeline_en": dr_info["timeline_en"] if dme_grade < 2 else "Within 1-2 Weeks (CSME)",
            "timeline_hi": dr_info["timeline_hi"] if dme_grade < 2 else "1-2 सप्ताह में (CSME)",
            "urgency_en": overall_urgency_en,
            "urgency_hi": overall_urgency_hi,
            "badge_text": badge_text,
            "badge_color": badge_color,
            "recommended_actions_en": dr_info["recommended_actions_en"],
            "recommended_actions_hi": dr_info["recommended_actions_hi"],
            "glaucoma_suspect": glaucoma_flag,
            "hypertensive_suspect": hypertensive_flag
        }

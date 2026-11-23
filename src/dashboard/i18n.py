"""
i18n.py - English / Tamil text for the farmer-facing screens.

t("key", **fmt) returns the text in the language chosen in the sidebar. Missing Tamil falls back to
English. The Tamil was written for this app and should be reviewed by a native speaker (ideally an
agricultural extension officer) before it is shown to farmers; NEEDS_REVIEW marks that.
"""
from __future__ import annotations

import streamlit as st

LANGS = {"en": "English", "ta": "தமிழ் (Tamil)"}
NEEDS_REVIEW = {"ta": True}

T = {
    # ---- shell
    "view": {"en": "View", "ta": "பார்வை"},
    "role_farmer": {"en": "Farmer", "ta": "விவசாயி"},
    "role_agronomist": {"en": "Agronomist", "ta": "வேளாண் நிபுணர்"},
    "role_research": {"en": "Researcher", "ta": "ஆராய்ச்சியாளர்"},
    "language": {"en": "Language", "ta": "மொழி"},
    "review_note": {"en": "", "ta": "தமிழ் மொழிபெயர்ப்பு சரிபார்க்கப்பட வேண்டும்."},
    # ---- navigation (farmer-facing pages and groups)
    "nav_farm": {"en": "Farm home", "ta": "வயல் முகப்பு"},
    "nav_upload": {"en": "Upload field image", "ta": "வயல் படம் பதிவேற்று"},
    "nav_optimizer": {"en": "Treatment plan", "ta": "சிகிச்சைத் திட்டம்"},
    "nav_overview": {"en": "Live twin & drones", "ta": "நேரலை இரட்டை & ட்ரோன்கள்"},
    "nav_leafcheck": {"en": "Leaf disease check", "ta": "இலை நோய் சோதனை"},
    "grp_Home": {"en": "Home", "ta": "முகப்பு"},
    "grp_Workflow": {"en": "Workflow", "ta": "பணிப்படிகள்"},
    "grp_Live operations": {"en": "Live operations", "ta": "நேரலை செயல்பாடுகள்"},
    # ---- farmer home
    "farm_title": {"en": "Your field today", "ta": "இன்று உங்கள் வயல்"},
    "farm_sub": {"en": "What the digital twin sees, and what to do next.",
                 "ta": "டிஜிட்டல் இரட்டை பார்ப்பது, அடுத்து செய்ய வேண்டியது."},
    "no_twin": {"en": "The field twin is not running. Ask your operator to start it.",
                "ta": "வயல் இரட்டை இயங்கவில்லை. இயக்குபவரிடம் தொடங்கச் சொல்லுங்கள்."},
    "health": {"en": "Crop health", "ta": "பயிர் ஆரோக்கியம்"},
    "health_val": {"en": "{pct:.0f}% of the field is healthy", "ta": "வயலின் {pct:.0f}% ஆரோக்கியமாக உள்ளது"},
    "disease": {"en": "Disease risk (next 7 days)", "ta": "நோய் அபாயம் (அடுத்த 7 நாட்கள்)"},
    "risk_low": {"en": "Low", "ta": "குறைவு"},
    "risk_medium": {"en": "Medium", "ta": "நடுத்தரம்"},
    "risk_high": {"en": "High", "ta": "அதிகம்"},
    "harvest": {"en": "Expected harvest", "ta": "எதிர்பார்க்கும் அறுவடை"},
    "harvest_val": {"en": "{t:.1f} tonnes per hectare", "ta": "ஹெக்டேருக்கு {t:.1f} டன்"},
    "weather": {"en": "Weather today", "ta": "இன்றைய வானிலை"},
    "rain_yes": {"en": "Rain {mm:.0f} mm", "ta": "மழை {mm:.0f} மி.மீ"},
    "rain_no": {"en": "No rain", "ta": "மழை இல்லை"},
    "todo": {"en": "What to do", "ta": "என்ன செய்ய வேண்டும்"},
    "todo_spray": {"en": "Spray zone(s) {zones} within 2 days. Spraying protects about {gain:.2f} t/ha of harvest.",
                   "ta": "2 நாட்களுக்குள் மண்டலம் {zones} இல் மருந்து தெளிக்கவும். இது ஹெக்டேருக்கு சுமார் "
                         "{gain:.2f} டன் அறுவடையைக் காக்கும்."},
    "todo_nospray": {"en": "No spraying needed now. Keep the water level steady and check again in 3 days.",
                     "ta": "இப்போது மருந்து தெளிக்கத் தேவையில்லை. நீர்மட்டத்தை சீராக வைத்து, 3 நாட்களில் மீண்டும் "
                           "பாருங்கள்."},
    "todo_rain": {"en": "Rain is expected: wait for a dry spell before spraying.",
                  "ta": "மழை எதிர்பார்க்கப்படுகிறது: மருந்து தெளிக்க வறண்ட நேரம் வரை காத்திருங்கள்."},
    "todo_scout": {"en": "The twin is unsure about parts of the field: fly a drone scout or check leaves there.",
                   "ta": "வயலின் சில பகுதிகள் பற்றி இரட்டை உறுதியாக இல்லை: ட்ரோன் ஆய்வு செய்யுங்கள் அல்லது அங்கு "
                         "இலைகளைப் பாருங்கள்."},
    "saved": {"en": "Fungicide saved so far: {pct:.0f}% compared with spraying the whole field.",
              "ta": "இதுவரை சேமித்த பூஞ்சைக் கொல்லி: முழு வயலிலும் தெளிப்பதுடன் ஒப்பிட்டால் {pct:.0f}%."},
    "btn_leaf": {"en": "Check a leaf", "ta": "இலையைச் சோதி"},
    "btn_upload": {"en": "Upload drone image", "ta": "ட்ரோன் படத்தைப் பதிவேற்று"},
    "btn_tasks": {"en": "Treatment plan", "ta": "சிகிச்சைத் திட்டம்"},
    "btn_live": {"en": "See the field live", "ta": "வயலை நேரலையில் பார்"},
    "map_title": {"en": "Field map (green = healthy, brown = disease)",
                  "ta": "வயல் வரைபடம் (பச்சை = ஆரோக்கியம், பழுப்பு = நோய்)"},
    # ---- leaf check
    "leaf_title": {"en": "Leaf disease check", "ta": "இலை நோய் சோதனை"},
    "leaf_intro": {"en": "Take or upload a close-up photo of one rice leaf.",
                   "ta": "ஒரு நெல் இலையின் அருகாமைப் புகைப்படத்தை எடுங்கள் அல்லது பதிவேற்றுங்கள்."},
    "leaf_most_likely": {"en": "Most likely", "ta": "பெரும்பாலும்"},
    "leaf_unsure": {"en": "The AI is unsure about this photo. Take another, closer photo in daylight.",
                    "ta": "இந்தப் புகைப்படம் பற்றி AI உறுதியாக இல்லை. பகலில் மீண்டும் அருகில் எடுங்கள்."},
    "leaf_not_diagnosis": {"en": "A photo is not a diagnosis: confirm with an agronomist before treating.",
                           "ta": "புகைப்படம் மட்டும் நோயறிதல் அல்ல: சிகிச்சைக்கு முன் வேளாண் நிபுணரிடம் உறுதி செய்யுங்கள்."},
    # ---- diseases
    "Healthy": {"en": "Healthy", "ta": "ஆரோக்கியமானது"},
    "Blast": {"en": "Blast", "ta": "குலை நோய் (Blast)"},
    "Brown spot": {"en": "Brown spot", "ta": "பழுப்புப் புள்ளி நோய்"},
    "Narrow brown spot": {"en": "Narrow brown spot", "ta": "குறுகிய பழுப்புப் புள்ளி நோய்"},
    "Sheath blight": {"en": "Sheath blight", "ta": "இலையுறை கருகல் நோய்"},
    "Bacterial leaf blight": {"en": "Bacterial leaf blight", "ta": "பாக்டீரியா இலைக் கருகல் நோய்"},
    "Tungro": {"en": "Tungro", "ta": "துங்ரோ நோய்"},
    # ---- alerts
    "alerts": {"en": "Alerts", "ta": "எச்சரிக்கைகள்"},
    "alert_disease": {"en": "Disease risk is high in zone(s) {zones}. Spray within 2 days.",
                      "ta": "மண்டலம் {zones} இல் நோய் அபாயம் அதிகம். 2 நாட்களுக்குள் மருந்து தெளிக்கவும்."},
    "alert_rain_spray": {"en": "Rain expected ({mm:.0f} mm): do not spray today.",
                         "ta": "மழை எதிர்பார்க்கப்படுகிறது ({mm:.0f} மி.மீ): இன்று மருந்து தெளிக்க வேண்டாம்."},
    "alert_water": {"en": "Parts of the field are drying out: check the water inlet and bunds.",
                    "ta": "வயலின் சில பகுதிகள் வறண்டு வருகின்றன: நீர் வரத்தையும் வரப்புகளையும் சரிபாருங்கள்."},
    "alert_none": {"en": "No alerts. Your field looks fine.", "ta": "எச்சரிக்கைகள் இல்லை. உங்கள் வயல் நன்றாக உள்ளது."},
}


def lang() -> str:
    return st.session_state.get("pt_lang", "en")


def t(key: str, **fmt) -> str:
    entry = T.get(key, {})
    text = entry.get(lang()) or entry.get("en") or key
    return text.format(**fmt) if fmt else text


def text_in(key: str, language: str, **fmt) -> str:
    """The same text in a given language (for alerts sent to someone in their own language)."""
    entry = T.get(key, {})
    text = entry.get(language) or entry.get("en") or key
    return text.format(**fmt) if fmt else text

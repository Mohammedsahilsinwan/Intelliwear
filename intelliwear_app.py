"""
╔══════════════════════════════════════════════════════════════════╗
║          IntelliWear — AI Hydration Monitoring System            ║
║  Fitbit API → Feature Engineering → LightGBM → Dashboard + Bot  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import json
import time
import os
import pickle
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💧 IntelliWear",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  GLOBAL STYLING  — mirrors the dark teal/purple palette from
#  the original HTML project
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700&display=swap');

:root {
  --bg:    #030B16;  --bg2: #060F1C;  --bg3: #0A1628;
  --panel: #0D1E35;  --border: #0F2238; --border2: #1A3050;
  --text:  #E2ECF8;  --muted: #4A6080;
  --teal:  #0ECECE;  --mint:  #14D4AC;  --green: #20D490;
  --red:   #F04060;  --yellow:#F0B040;  --purple:#9060E0;
}

html, body, [data-testid="stApp"] {
    background-color: #030B16 !important;
    color: #E2ECF8 !important;
    font-family: 'Manrope', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #060F1C !important;
    border-right: 1px solid #0F2238 !important;
}
[data-testid="stSidebar"] * { color: #E2ECF8 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #0D1E35;
    border: 1px solid #1A3050;
    border-radius: 14px;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: #4A6080 !important; font-family:'DM Mono',monospace !important; font-size:11px !important; }
[data-testid="stMetricValue"] { color: #0ECECE !important; font-family:'Playfair Display',serif !important; font-size:28px !important; font-weight:900 !important; }
[data-testid="stMetricDelta"] { font-size:11px !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #9060E0, #5030B0) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(144,96,224,.4) !important;
    transition: all .3s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(144,96,224,.55) !important;
}

/* Chat input */
.stChatInput > div { background:#0A1628 !important; border:1px solid #1A3050 !important; border-radius:12px !important; }
.stChatMessage { background:#0D1E35 !important; border:1px solid #1A3050 !important; border-radius:14px !important; }

/* Expanders / info boxes */
.stAlert { border-radius: 12px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #060F1C !important; border-radius:10px; gap:2px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color:#4A6080 !important; border-radius:8px !important; }
.stTabs [aria-selected="true"] { background: #0D1E35 !important; color:#0ECECE !important; }

/* Divider */
hr { border-color: #0F2238 !important; }

/* Selectbox / text input */
.stSelectbox > div > div, .stTextInput > div > div {
    background: #0A1628 !important;
    border: 1px solid #1A3050 !important;
    border-radius: 10px !important;
    color: #E2ECF8 !important;
}

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg,#0ECECE,#9060E0) !important; border-radius:4px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  FITBIT OAUTH 2.0  LAYER
# ─────────────────────────────────────────────────────────────────
FITBIT_AUTH_URL  = "https://www.fitbit.com/oauth2/authorize"
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
FITBIT_API_BASE  = "https://api.fitbit.com/1/user/-"

def build_auth_url(client_id: str, redirect_uri: str, scope: str = "heartrate activity profile") -> str:
    """Build Fitbit OAuth2 authorization URL."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "expires_in": "604800",
    }
    from urllib.parse import urlencode
    return f"{FITBIT_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    """Exchange auth code for access + refresh tokens (Authorization Code Flow)."""
    import base64
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        FITBIT_TOKEN_URL,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token: str, client_id: str, client_secret: str) -> dict:
    """Refresh an expired access token."""
    import base64
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        FITBIT_TOKEN_URL,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_heart_rate(access_token: str) -> dict:
    """
    Fetch today's heart rate data from Fitbit API.
    Returns dict with keys: avg_hr, resting_hr, zones, raw_response
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    today = datetime.now().strftime("%Y-%m-%d")

    # Heart rate time series (intraday)
    url = f"{FITBIT_API_BASE}/activities/heart/date/{today}/1d.json"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    heart_data = data.get("activities-heart", [{}])[0].get("value", {})
    resting_hr  = heart_data.get("restingHeartRate", 60)
    zones       = heart_data.get("heartRateZones", [])

    # Compute weighted average HR from zones
    total_min = sum(z.get("minutes", 0) for z in zones)
    if total_min > 0:
        avg_hr = sum(
            ((z.get("min", 0) + z.get("max", 220)) / 2) * z.get("minutes", 0)
            for z in zones
        ) / total_min
    else:
        avg_hr = resting_hr

    return {
        "avg_hr": round(avg_hr, 1),
        "resting_hr": resting_hr,
        "zones": zones,
        "raw": data,
    }


def fetch_activity_summary(access_token: str) -> dict:
    """Fetch today's activity summary for context features."""
    headers = {"Authorization": f"Bearer {access_token}"}
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"{FITBIT_API_BASE}/activities/date/{today}.json"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_profile(access_token: str) -> dict:
    """Fetch user profile (age, weight, gender)."""
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{FITBIT_API_BASE}/profile.json"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json().get("user", {})


# ─────────────────────────────────────────────────────────────────
#  DEMO / SIMULATED DATA  (used when no real token is present)
# ─────────────────────────────────────────────────────────────────
DEMO_SCENARIOS = {
    "🔴 Critical — Intense workout, hot day":  dict(avg_hr=188, resting_hr=62, age=28, weight=72, temperature=40, duration=100, gender="male"),
    "🟡 Moderate — Jogging session":           dict(avg_hr=135, resting_hr=60, age=24, weight=70, temperature=33, duration=50,  gender="male"),
    "🟠 Low Risk — Light walk":                dict(avg_hr=108, resting_hr=58, age=35, weight=75, temperature=28, duration=30,  gender="female"),
    "🟢 Optimal — Rest / recovery":            dict(avg_hr=72,  resting_hr=58, age=22, weight=65, temperature=24, duration=0,   gender="female"),
}


# ─────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING  LAYER
# ─────────────────────────────────────────────────────────────────
def engineer_features(hr: float, resting_hr: float, age: int,
                      weight: float, temperature: float,
                      duration: float, gender: str) -> dict:
    """
    Compute all physiological features from raw sensor data.
    No manual user input — everything derived from Fitbit + context.
    """
    # 1. HR metrics
    hr_max  = 220 - age
    hr_norm = (hr - resting_hr) / max(hr_max - resting_hr, 1)
    hr_norm = float(np.clip(hr_norm, 0, 1))

    # 2. RPE (Rate of Perceived Exertion) — 1-10 scale from HR
    rpe = round(hr_norm * 10, 2)

    # 3. Activity level classification
    if hr_norm < 0.4:
        activity_level = 1   # Low
    elif hr_norm < 0.7:
        activity_level = 2   # Moderate
    else:
        activity_level = 3   # High

    # 4. Sweat Rate  (L/hr) — combines RPE and ambient temperature
    sweat_rate = round((0.2 * rpe) + (0.01 * temperature), 3)
    sweat_rate = float(np.clip(sweat_rate, 0.05, 3.5))

    # 5. Total fluid loss over the session (L)
    fluid_loss = round(sweat_rate * (duration / 60), 3)

    # 6. Stress score (0-100) based on HR_norm and duration
    stress = round(min(100, hr_norm * 70 + (duration / 300) * 30), 1)

    # 7. Fatigue score
    fatigue = round(min(100, (duration / 300) * 60 + hr_norm * 40), 1)

    # 8. Recovery score (inverse of fatigue + stress)
    recovery = round(max(0, 100 - (stress + fatigue) / 2), 1)

    # 9. Readiness score
    readiness = round((recovery * 0.6) + ((1 - hr_norm) * 40), 1)

    # 10. BMI proxy from weight and assumed height
    # We assume a generic height of 170 cm for demo; real Fitbit profile has height
    height_m = 1.70
    bmi = round(weight / (height_m ** 2), 1)

    # 11. Gender encoding
    gender_enc = 0 if gender.lower() == "male" else 1

    return {
        "hr":             hr,
        "resting_hr":     resting_hr,
        "hr_max":         hr_max,
        "hr_norm":        hr_norm,
        "rpe":            rpe,
        "activity_level": activity_level,
        "sweat_rate":     sweat_rate,
        "fluid_loss":     fluid_loss,
        "temperature":    temperature,
        "duration":       duration,
        "stress":         stress,
        "fatigue":        fatigue,
        "recovery":       recovery,
        "readiness":      readiness,
        "age":            age,
        "weight":         weight,
        "bmi":            bmi,
        "gender_enc":     gender_enc,
    }


# ─────────────────────────────────────────────────────────────────
#  ELECTROLYTE  CALCULATIONS
# ─────────────────────────────────────────────────────────────────
def compute_electrolytes(feats: dict) -> dict:
    """
    Sodium loss:    sweat_rate × 1000  mg/hr
    Potassium loss: scaled by activity level
    """
    sweat_rate     = feats["sweat_rate"]
    activity_level = feats["activity_level"]

    sodium_loss    = round(sweat_rate * 1000, 0)          # mg/hr
    potassium_base = {1: 80, 2: 180, 3: 290}
    potassium_loss = round(potassium_base[activity_level] * sweat_rate / 0.9, 0)

    return {
        "sodium_loss":    int(sodium_loss),
        "potassium_loss": int(potassium_loss),
    }


# ─────────────────────────────────────────────────────────────────
#  LIGHTGBM  MODEL  (load or create synthetic stand-in)
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "intelliwear_lgbm.pkl"

FEATURE_COLS = [
    "hr_norm", "rpe", "sweat_rate", "temperature", "duration",
    "stress", "fatigue", "recovery", "age", "bmi", "gender_enc", "activity_level",
]

LABEL_MAP = {0: "Hydrated", 1: "Mild Dehydration", 2: "Severe Dehydration"}
LABEL_COLOR = {0: "#20D490", 1: "#F0B040", 2: "#F04060"}


def _build_synthetic_model():
    """
    Build and save a lightweight synthetic LightGBM model trained on
    rule-based labels so the app runs without a pre-trained .pkl file.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None  # Fallback to rule-based prediction

    rng = np.random.default_rng(42)
    n = 3000

    hr_norm      = rng.uniform(0, 1, n)
    rpe          = hr_norm * 10 + rng.normal(0, 0.3, n)
    sweat_rate   = (0.2 * rpe + 0.01 * rng.uniform(15, 42, n)).clip(0.05, 3.5)
    temperature  = rng.uniform(15, 45, n)
    duration     = rng.uniform(0, 300, n)
    stress       = (hr_norm * 70 + duration / 300 * 30).clip(0, 100)
    fatigue      = (duration / 300 * 60 + hr_norm * 40).clip(0, 100)
    recovery     = (100 - (stress + fatigue) / 2).clip(0, 100)
    age          = rng.integers(18, 65, n)
    bmi          = rng.uniform(16, 35, n)
    gender_enc   = rng.integers(0, 2, n)
    activity_lv  = rng.integers(1, 4, n)

    # Rule-based label
    risk_score = (1 - hr_norm) * 30 + sweat_rate * 20 + (temperature - 20) * 0.5 + fatigue * 0.2
    label = np.where(risk_score < 20, 0, np.where(risk_score < 35, 1, 2))

    X = np.stack([hr_norm, rpe.clip(0, 10), sweat_rate, temperature,
                  duration, stress, fatigue, recovery, age, bmi, gender_enc, activity_lv], axis=1)

    ds = lgb.Dataset(X, label=label)
    params = {
        "objective": "multiclass", "num_class": 3,
        "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 120,
        "verbosity": -1,
    }
    model = lgb.train(params, ds, num_boost_round=120)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource
def load_model():
    """Load model from disk (or build synthetic one if not present)."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), "loaded"
        except Exception:
            pass
    model = _build_synthetic_model()
    return model, "synthetic"


def predict_hydration(feats: dict, model) -> tuple[str, float, dict]:
    """
    Returns (label_str, confidence_pct, all_probs_dict).
    Falls back to rule-based if model unavailable.
    """
    if model is None:
        # Rule-based fallback
        risk = (feats["sweat_rate"] * 25) + ((feats["temperature"] - 20) * 0.5) + (feats["hr_norm"] * 20) + (feats["fatigue"] * 0.15)
        if risk < 18:
            idx = 0
        elif risk < 32:
            idx = 1
        else:
            idx = 2
        probs = {0: 0.0, 1: 0.0, 2: 0.0}
        probs[idx] = 1.0
        return LABEL_MAP[idx], 100.0, probs

    X = np.array([[feats[c] for c in FEATURE_COLS]])
    raw = model.predict(X)[0]           # shape (3,) — class probabilities
    idx = int(np.argmax(raw))
    conf = round(float(raw[idx]) * 100, 1)
    probs = {i: round(float(raw[i]) * 100, 1) for i in range(3)}
    return LABEL_MAP[idx], conf, probs


# ─────────────────────────────────────────────────────────────────
#  AI RECOMMENDATION  LAYER
# ─────────────────────────────────────────────────────────────────
def generate_recommendation(feats: dict, elytes: dict, label: str) -> dict:
    """
    Rule-based AI recommendations based on predicted hydration status,
    sweat rate, RPE, and electrolyte loss.
    """
    rpe          = feats["rpe"]
    sweat_rate   = feats["sweat_rate"]
    activity_lv  = feats["activity_level"]
    sodium       = elytes["sodium_loss"]
    potassium    = elytes["potassium_loss"]

    # Water intake recommendation (ml/hr)
    base_water   = {1: 150, 2: 350, 3: 600}[activity_lv]
    if label == "Severe Dehydration":
        water_ml = base_water + int(sweat_rate * 600)
        urgency  = "🚨 DRINK IMMEDIATELY — every minute counts"
        action   = "Stop activity. Drink 500ml of water + electrolytes right now."
        icon     = "🔴"
    elif label == "Mild Dehydration":
        water_ml = base_water + int(sweat_rate * 350)
        urgency  = "⚠️ Drink within the next 10 minutes"
        action   = "Slow down. Sip 250–500ml water over the next 15 minutes."
        icon     = "🟡"
    else:
        water_ml = base_water
        urgency  = "✅ Maintain your current hydration rhythm"
        action   = "Continue drinking at your current pace. Great work!"
        icon     = "🟢"

    # Electrolyte advice
    if sodium > 600:
        electrolyte_advice = "Add an electrolyte tablet or sports drink. High sodium loss detected."
    elif sodium > 300:
        electrolyte_advice = "Consider a banana + pinch of salt in water for potassium + sodium."
    else:
        electrolyte_advice = "Plain water is sufficient. Normal electrolyte levels."

    return {
        "water_ml_hr": water_ml,
        "urgency":     urgency,
        "action":      action,
        "icon":        icon,
        "electrolyte": electrolyte_advice,
        "sodium_mg":   sodium,
        "potassium_mg": potassium,
    }


# ─────────────────────────────────────────────────────────────────
#  CHATBOT  (rule-based with full session context)
# ─────────────────────────────────────────────────────────────────
def chatbot_reply(query: str, context: dict) -> str:
    """
    Contextual rule-based chatbot that uses live model output.
    context keys: label, feats, reco, elytes, conf
    """
    q     = query.lower().strip()
    label = context.get("label", "Unknown")
    reco  = context.get("reco", {})
    feats = context.get("feats", {})
    elytes= context.get("elytes", {})
    conf  = context.get("conf", 0)

    # ── Hydration status ──
    if any(k in q for k in ["hydration", "dehydrat", "status", "how am i", "how is my"]):
        return (
            f"**💧 Your Hydration Status: {label}** (Confidence: {conf:.0f}%)\n\n"
            f"- Heart Rate: **{feats.get('hr', 0):.0f} bpm** (normalised: {feats.get('hr_norm', 0):.2f})\n"
            f"- Sweat Rate: **{feats.get('sweat_rate', 0):.2f} L/hr**\n"
            f"- Fatigue Score: **{feats.get('fatigue', 0):.0f}/100**\n\n"
            f"{reco.get('urgency', '')}\n\n"
            f"_{reco.get('action', '')}_"
        )

    # ── Water intake ──
    if any(k in q for k in ["drink", "water", "intake", "how much", "ml", "litre", "liter"]):
        return (
            f"**🥤 Water Recommendation**\n\n"
            f"Based on your current activity (**RPE {feats.get('rpe', 0):.1f}**) "
            f"and sweat rate (**{feats.get('sweat_rate', 0):.2f} L/hr**), you should drink:\n\n"
            f"### **{reco.get('water_ml_hr', 0)} ml/hour**\n\n"
            f"That's roughly **{reco.get('water_ml_hr', 0) // 250} glasses per hour**.\n\n"
            f"{reco.get('urgency', '')}"
        )

    # ── Electrolytes ──
    if any(k in q for k in ["electrolyte", "sodium", "salt", "potassium", "banana", "mineral"]):
        return (
            f"**🧂 Electrolyte Status**\n\n"
            f"- Sodium loss: **{elytes.get('sodium_loss', 0)} mg/hr**\n"
            f"- Potassium loss: **{elytes.get('potassium_loss', 0)} mg/hr**\n\n"
            f"**Advice:** {reco.get('electrolyte', 'Normal levels — water is fine.')}\n\n"
            f"*Sodium loss = sweat_rate × 1000 mg/hr as per IntelliWear formula.*"
        )

    # ── Sweat rate ──
    if any(k in q for k in ["sweat", "perspir", "fluid loss"]):
        fl = feats.get("fluid_loss", 0)
        return (
            f"**💦 Sweat Analysis**\n\n"
            f"- Sweat Rate: **{feats.get('sweat_rate', 0):.2f} L/hr**\n"
            f"- Formula: `Sweat Rate = (0.2 × RPE) + (0.01 × Temperature)`\n"
            f"- RPE: **{feats.get('rpe', 0):.1f}** | Temperature: **{feats.get('temperature', 0):.1f}°C**\n"
            f"- Session Fluid Loss: **{fl:.2f} L** over {feats.get('duration', 0):.0f} min\n\n"
            f"{'⚠️ High sweat rate — replenish actively.' if feats.get('sweat_rate', 0) > 1.5 else '✅ Sweat rate within manageable range.'}"
        )

    # ── AI / LightGBM ──
    if any(k in q for k in ["lgbm", "lightgbm", "model", "ai", "ml", "machine", "algorithm", "predict"]):
        return (
            "**🧠 How the AI Model Works**\n\n"
            "IntelliWear uses a **LightGBM** (gradient-boosted decision trees) classifier trained on "
            "12 engineered physiological features:\n\n"
            "`hr_norm, rpe, sweat_rate, temperature, duration, stress, fatigue, recovery, age, bmi, gender, activity_level`\n\n"
            "**Pipeline:**\n"
            "Fitbit API → Feature Engineering → LightGBM → 3-class prediction\n\n"
            "**Classes:** Hydrated · Mild Dehydration · Severe Dehydration\n\n"
            f"Current prediction confidence: **{conf:.0f}%**"
        )

    # ── Fitbit / API ──
    if any(k in q for k in ["fitbit", "api", "oauth", "token", "wearable", "sensor", "device"]):
        return (
            "**⌚ Fitbit API Integration**\n\n"
            "IntelliWear uses **OAuth 2.0 Authorization Code Flow**:\n\n"
            "1. User clicks **'Connect Fitbit'** → redirected to Fitbit login\n"
            "2. Fitbit returns an auth code → exchanged for `access_token`\n"
            "3. App calls `/activities/heart/date/today/1d.json` for real-time HR\n"
            "4. HR data flows into the feature engineering layer automatically\n\n"
            "No manual input required — all physiological data comes from your wearable."
        )

    # ── Heart rate ──
    if any(k in q for k in ["heart rate", "bpm", "hr ", "pulse", "cardiac"]):
        return (
            f"**❤️ Heart Rate Data**\n\n"
            f"- Current HR: **{feats.get('hr', 0):.0f} bpm**\n"
            f"- Resting HR: **{feats.get('resting_hr', 0):.0f} bpm**\n"
            f"- Max HR (220 - age): **{feats.get('hr_max', 0):.0f} bpm**\n"
            f"- HR_norm = (HR - HR_rest) / (HR_max - HR_rest) = **{feats.get('hr_norm', 0):.3f}**\n"
            f"- RPE (HR_norm × 10): **{feats.get('rpe', 0):.1f}**\n\n"
            f"Source: Fitbit heartRateZones weighted average"
        )

    # ── Stress/fatigue ──
    if any(k in q for k in ["stress", "fatigue", "tired", "recovery", "readiness", "score"]):
        return (
            f"**📊 Biometric Scores**\n\n"
            f"| Score | Value |\n"
            f"|-------|-------|\n"
            f"| Stress | {feats.get('stress', 0):.0f}/100 |\n"
            f"| Fatigue | {feats.get('fatigue', 0):.0f}/100 |\n"
            f"| Recovery | {feats.get('recovery', 0):.0f}/100 |\n"
            f"| Readiness | {feats.get('readiness', 0):.0f}/100 |\n\n"
            f"These are computed from HR_norm, session duration, and activity level."
        )

    # ── Greetings ──
    if any(k in q for k in ["hello", "hi", "hey", "hola", "greet"]):
        return (
            f"👋 **Hello! I'm IntelliWear AI.**\n\n"
            f"Your live dashboard shows: **{label}** (HR: {feats.get('hr', 0):.0f} bpm, Sweat: {feats.get('sweat_rate', 0):.2f} L/hr)\n\n"
            f"Ask me about:\n"
            f"- 💧 Your hydration status\n"
            f"- 🥤 How much water to drink\n"
            f"- 🧂 Electrolyte advice\n"
            f"- ❤️ Your heart rate data\n"
            f"- 🧠 How the AI model works\n"
            f"- ⌚ Fitbit API integration"
        )

    # ── Default ──
    return (
        f"I didn't quite catch that. I can help with:\n\n"
        f"- **Hydration status** → *\"How is my hydration?\"*\n"
        f"- **Water intake** → *\"What should I drink?\"*\n"
        f"- **Electrolytes** → *\"What about sodium?\"*\n"
        f"- **AI model** → *\"Explain LightGBM\"*\n"
        f"- **Fitbit API** → *\"How does the API work?\"*\n\n"
        f"Current status: **{label}** — {reco.get('urgency', '')}"
    )


# ─────────────────────────────────────────────────────────────────
#  SESSION STATE  INIT
# ─────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "access_token":    None,
        "refresh_token":   None,
        "token_expiry":    None,
        "fitbit_profile":  {},
        "last_hr_data":    None,
        "feats":           None,
        "elytes":          None,
        "label":           None,
        "conf":            None,
        "probs":           None,
        "reco":            None,
        "chat_history":    [],
        "demo_scenario":   list(DEMO_SCENARIOS.keys())[1],
        "client_id":       "",
        "client_secret":   "",
        "redirect_uri":    "http://localhost:8501",
        "auth_code":       "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
model, model_status = load_model()


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────
def run_full_pipeline(hr: float, resting_hr: float, age: int,
                      weight: float, temperature: float,
                      duration: float, gender: str):
    """End-to-end pipeline: raw data → dashboard outputs."""
    feats = engineer_features(hr, resting_hr, age, weight, temperature, duration, gender)
    elytes = compute_electrolytes(feats)
    label, conf, probs = predict_hydration(feats, model)
    reco = generate_recommendation(feats, elytes, label)

    st.session_state.feats  = feats
    st.session_state.elytes = elytes
    st.session_state.label  = label
    st.session_state.conf   = conf
    st.session_state.probs  = probs
    st.session_state.reco   = reco
    return feats, elytes, label, conf, probs, reco


RISK_COLOR = {
    "Hydrated":           "#20D490",
    "Mild Dehydration":   "#F0B040",
    "Severe Dehydration": "#F04060",
}
RISK_EMOJI = {
    "Hydrated":           "🟢",
    "Mild Dehydration":   "🟡",
    "Severe Dehydration": "🔴",
}


def status_badge(label: str) -> str:
    color = RISK_COLOR.get(label, "#4A6080")
    emoji = RISK_EMOJI.get(label, "⚪")
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'background:{color}18;border:1px solid {color}55;'
        f'border-radius:18px;padding:6px 16px;">'
        f'<span style="font-size:16px">{emoji}</span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:13px;'
        f'font-weight:700;color:{color}">{label.upper()}</span></div>'
    )


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR  — Auth + Demo controls
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 10px;">
      <span style="font-family:'Playfair Display',serif;font-size:24px;font-weight:900;
        background:linear-gradient(135deg,#0ECECE,#9060E0);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        💧 IntelliWear
      </span>
      <div style="font-family:'DM Mono',monospace;font-size:10px;color:#4A6080;margin-top:4px;letter-spacing:.1em;">
        AI HYDRATION MONITOR
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Connection Mode ──
    mode = st.selectbox(
        "Data Source",
        ["🎭 Demo Mode (Simulated)", "⌚ Live Fitbit API"],
        key="data_mode"
    )

    # ─── DEMO MODE ───
    if "Demo" in mode:
        st.markdown("**Demo Scenario**")
        scenario = st.selectbox("Select scenario", list(DEMO_SCENARIOS.keys()), key="demo_scenario")

        st.markdown("**Context Overrides**")
        temperature = st.slider("🌡 Temperature (°C)", -10, 55, 33)
        duration    = st.slider("⏱ Duration (min)",   0, 300, 50)

        if st.button("▶ Run Analysis", use_container_width=True):
            d = DEMO_SCENARIOS[scenario]
            with st.spinner("Running pipeline…"):
                run_full_pipeline(
                    hr=d["avg_hr"], resting_hr=d["resting_hr"],
                    age=d["age"], weight=d["weight"],
                    temperature=temperature, duration=duration,
                    gender=d["gender"],
                )
            st.success("✅ Analysis complete!")

    # ─── LIVE FITBIT ───
    else:
        st.markdown("**Fitbit Credentials**")
        st.session_state.client_id     = st.text_input("Client ID",     value=st.session_state.client_id,     type="default")
        st.session_state.client_secret = st.text_input("Client Secret", value=st.session_state.client_secret, type="password")
        st.session_state.redirect_uri  = st.text_input("Redirect URI",  value=st.session_state.redirect_uri)

        if st.session_state.client_id:
            auth_url = build_auth_url(st.session_state.client_id, st.session_state.redirect_uri)
            st.markdown(f"**Step 1 →** [🔗 Authorize on Fitbit]({auth_url})", unsafe_allow_html=False)
            st.caption("After authorizing, copy the `code` from the redirect URL")

            st.session_state.auth_code = st.text_input("Paste Auth Code here", value=st.session_state.auth_code)

            if st.button("🔑 Get Token", use_container_width=True) and st.session_state.auth_code:
                try:
                    tok = exchange_code_for_token(
                        st.session_state.auth_code,
                        st.session_state.client_id,
                        st.session_state.client_secret,
                        st.session_state.redirect_uri,
                    )
                    st.session_state.access_token  = tok["access_token"]
                    st.session_state.refresh_token = tok.get("refresh_token", "")
                    st.success("✅ Authenticated with Fitbit!")
                except Exception as e:
                    st.error(f"Auth error: {e}")

        if st.session_state.access_token:
            st.success("🟢 Fitbit Connected")

            # Context sliders (non-physiological — user provides these)
            temperature = st.slider("🌡 Temperature (°C)", -10, 55, 30)
            duration    = st.slider("⏱ Duration (min)",   0, 300, 30)

            if st.button("🔄 Fetch & Analyse", use_container_width=True):
                try:
                    with st.spinner("Fetching from Fitbit…"):
                        hr_data = fetch_heart_rate(st.session_state.access_token)
                        profile = fetch_profile(st.session_state.access_token)
                        st.session_state.last_hr_data   = hr_data
                        st.session_state.fitbit_profile = profile

                    age    = profile.get("age", 25)
                    weight = profile.get("weight", 70)
                    gender = profile.get("gender", "MALE").lower().replace("male", "male").replace("female", "female")

                    with st.spinner("Running ML pipeline…"):
                        run_full_pipeline(
                            hr=hr_data["avg_hr"],
                            resting_hr=hr_data["resting_hr"],
                            age=age, weight=weight,
                            temperature=temperature,
                            duration=duration,
                            gender=gender,
                        )
                    st.success("✅ Live analysis complete!")
                except requests.HTTPError as e:
                    if e.response.status_code == 401:
                        # Try refresh
                        try:
                            tok = refresh_access_token(
                                st.session_state.refresh_token,
                                st.session_state.client_id,
                                st.session_state.client_secret,
                            )
                            st.session_state.access_token  = tok["access_token"]
                            st.session_state.refresh_token = tok.get("refresh_token", st.session_state.refresh_token)
                            st.warning("Token refreshed — please click Fetch again.")
                        except Exception as re:
                            st.error(f"Re-auth failed: {re}")
                    else:
                        st.error(f"Fitbit API error: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown(
        f'<div style="font-family:\'DM Mono\',monospace;font-size:9px;color:#2A4060;text-align:center;">'
        f'Model: {"✅ LightGBM" if model else "⚠️ Rule-based"} · {model_status}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
#  MAIN  LAYOUT
# ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align:center;padding:24px 0 18px;">
  <h1 style="font-family:'Playfair Display',serif;font-size:42px;font-weight:900;margin:0;
    background:linear-gradient(135deg,#0ECECE,#9060E0,#14D4AC);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    💧 IntelliWear
  </h1>
  <p style="font-family:'DM Mono',monospace;font-size:12px;color:#4A6080;margin:6px 0 0;letter-spacing:.12em;">
    FITBIT API → FEATURE ENGINEERING → LIGHTGBM → AI RECOMMENDATIONS
  </p>
</div>
""", unsafe_allow_html=True)

# If no analysis yet — show prompt
if st.session_state.feats is None:
    st.markdown("""
    <div style="text-align:center;border:2px dashed #1A3050;border-radius:20px;padding:60px 40px;margin:30px 0;">
      <div style="font-size:52px;margin-bottom:14px;opacity:.4">💧</div>
      <div style="font-family:'Playfair Display',serif;font-size:22px;color:#4A6080;margin-bottom:8px;">
        No analysis yet
      </div>
      <div style="font-size:13px;color:#2A4060;">
        Use the sidebar to connect Fitbit or select a Demo scenario, then click <strong style="color:#9060E0">Run Analysis</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    feats = st.session_state.feats
    elytes = st.session_state.elytes
    label  = st.session_state.label
    conf   = st.session_state.conf
    probs  = st.session_state.probs
    reco   = st.session_state.reco
    color  = RISK_COLOR.get(label, "#4A6080")

    # ── Status Banner ──
    col_l, col_m, col_r = st.columns([1.2, 2, 1.2])
    with col_m:
        st.markdown(
            f'<div style="text-align:center;background:{color}10;border:2px solid {color}40;'
            f'border-radius:20px;padding:22px;margin-bottom:18px;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#4A6080;letter-spacing:.15em;margin-bottom:6px;">HYDRATION STATUS</div>'
            f'{status_badge(label)}'
            f'<div style="font-family:\'Playfair Display\',serif;font-size:36px;font-weight:900;color:{color};margin:8px 0 4px;">{conf:.0f}%</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#4A6080;">MODEL CONFIDENCE</div>'
            f'<div style="margin-top:10px;font-size:13px;color:#A0B8D0;font-style:italic;">{reco["urgency"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Top metrics ──
    tabs = st.tabs(["📊 Dashboard", "🧠 ML Scores", "💧 Recommendations", "🤖 Chatbot"])

    # ════════════════════════════════════════════════════
    #  TAB 1 — DASHBOARD
    # ════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Live Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("❤️ Heart Rate",    f"{feats['hr']:.0f} bpm",      delta=f"Resting: {feats['resting_hr']:.0f}")
        c2.metric("💦 Sweat Rate",    f"{feats['sweat_rate']:.2f} L/hr")
        c3.metric("🧂 Sodium Loss",   f"{elytes['sodium_loss']} mg/hr")
        c4.metric("🏃 RPE",           f"{feats['rpe']:.1f} / 10")

        st.divider()
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("🌡 Temperature",    f"{feats['temperature']}°C")
        c6.metric("⏱ Duration",        f"{feats['duration']:.0f} min")
        c7.metric("💪 Activity Level", f"Level {feats['activity_level']}")
        c8.metric("🍌 Potassium Loss", f"{elytes['potassium_loss']} mg/hr")

        st.divider()
        # HR_norm gauge via progress bar
        st.markdown("**HR Normalisation (HR_norm)**")
        st.caption(f"HR_norm = (HR - HR_rest) / (HR_max - HR_rest) = **{feats['hr_norm']:.3f}**")
        st.progress(feats["hr_norm"])

        st.markdown("**Fluid Loss this Session**")
        fl_pct = min(feats["fluid_loss"] / 3.0, 1.0)
        st.caption(f"Total fluid loss = Sweat Rate × Duration = **{feats['fluid_loss']:.2f} L** (over {feats['duration']:.0f} min)")
        st.progress(fl_pct)

        # Fitbit raw HR zones (if live data)
        if st.session_state.last_hr_data and st.session_state.last_hr_data.get("zones"):
            st.subheader("Fitbit Heart Rate Zones")
            zones = st.session_state.last_hr_data["zones"]
            df_zones = pd.DataFrame(zones)[["name", "min", "max", "minutes", "caloriesOut"]]
            df_zones.columns = ["Zone", "Min BPM", "Max BPM", "Minutes", "Calories"]
            st.dataframe(df_zones, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════
    #  TAB 2 — ML SCORES
    # ════════════════════════════════════════════════════
    with tabs[1]:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Biometric Scores")
            for name, val, color_code in [
                ("🧠 Stress",     feats["stress"],     "#F04060"),
                ("😓 Fatigue",    feats["fatigue"],    "#F0B040"),
                ("🔋 Recovery",   feats["recovery"],   "#20D490"),
                ("⚡ Readiness",  feats["readiness"],  "#0ECECE"),
            ]:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                    f'<span style="font-size:13px;color:#A0B8D0">{name}</span>'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:700;color:{color_code}">{val:.0f}/100</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(val / 100)

        with col_b:
            st.subheader("Prediction Probabilities")
            for idx, lbl in LABEL_MAP.items():
                p = probs.get(idx, 0)
                c = LABEL_COLOR[idx]
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                    f'<span style="font-size:13px;color:#A0B8D0">{lbl}</span>'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:700;color:{c}">{p:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(p / 100)

        st.divider()
        st.subheader("Feature Vector (12 inputs to LightGBM)")
        feat_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Value": [round(feats[c], 4) for c in FEATURE_COLS],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════
    #  TAB 3 — RECOMMENDATIONS
    # ════════════════════════════════════════════════════
    with tabs[2]:
        col_p, col_e = st.columns(2)

        with col_p:
            st.subheader("💧 Water Intake")
            st.markdown(
                f'<div style="background:#0ECECE18;border:1px solid #0ECECE40;border-radius:14px;padding:20px;text-align:center;">'
                f'<div style="font-family:\'Playfair Display\',serif;font-size:48px;font-weight:900;color:#0ECECE;">'
                f'{reco["water_ml_hr"]}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:#4A6080;margin-top:4px;letter-spacing:.1em;">ML / HOUR</div>'
                f'<div style="margin-top:12px;font-size:13px;color:#A0B8D0;font-style:italic;">{reco["action"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Urgency:** {reco['urgency']}")

        with col_e:
            st.subheader("🧂 Electrolytes")
            ec1, ec2 = st.columns(2)
            ec1.metric("Sodium Loss",    f"{reco['sodium_mg']} mg/hr")
            ec2.metric("Potassium Loss", f"{reco['potassium_mg']} mg/hr")
            st.info(f"**{reco['electrolyte']}**")

            st.subheader("🍽 Food Tips")
            act = feats["activity_level"]
            if act == 3:
                st.markdown("- 🍌 Banana + peanut butter (potassium + protein)\n- 🧃 Coconut water (natural electrolytes)\n- 🥨 Pretzels (sodium replenishment)\n- 💊 Electrolyte tab in 500ml water")
            elif act == 2:
                st.markdown("- 🍉 Watermelon (hydration + potassium)\n- 🥒 Cucumber slices (water-rich)\n- 🧃 Diluted sports drink (30% water 70%)\n- 🍊 Orange slices (Vitamin C + hydration)")
            else:
                st.markdown("- 💧 Plain water throughout the day\n- 🥗 Green vegetables (natural minerals)\n- 🍵 Herbal tea (hydrating, no caffeine)\n- 🫚 Avoid alcohol and excess caffeine")

    # ════════════════════════════════════════════════════
    #  TAB 4 — CHATBOT
    # ════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("🤖 IntelliWear AI Chatbot")
        st.caption("Powered by rule-based AI with full session context. Ask about your hydration, water intake, electrolytes, or the ML model.")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="💧" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

        # Welcome message on first load
        if not st.session_state.chat_history:
            welcome = chatbot_reply("hello", {
                "label": label, "feats": feats, "reco": reco, "elytes": elytes, "conf": conf
            })
            with st.chat_message("assistant", avatar="💧"):
                st.markdown(welcome)
            st.session_state.chat_history.append({"role": "assistant", "content": welcome})

        # Suggested queries
        st.markdown("**Quick questions:**")
        qcols = st.columns(3)
        quick_qs = [
            "How is my hydration?",
            "What should I drink?",
            "How much sodium am I losing?",
            "Explain the LightGBM model",
            "What is my sweat rate?",
            "How does Fitbit integration work?",
        ]
        for i, q in enumerate(quick_qs):
            if qcols[i % 3].button(q, key=f"qq_{i}", use_container_width=True):
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": q})
                reply = chatbot_reply(q, {
                    "label": label, "feats": feats, "reco": reco, "elytes": elytes, "conf": conf
                })
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

        # Chat input
        if user_input := st.chat_input("Ask anything about IntelliWear or your hydration…"):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="💧"):
                with st.spinner("Thinking…"):
                    time.sleep(0.4)
                    reply = chatbot_reply(user_input, {
                        "label": label, "feats": feats, "reco": reco, "elytes": elytes, "conf": conf
                    })
                st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        if st.button("🗑 Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────
#  SYSTEM  ARCHITECTURE  (always visible at bottom)
# ─────────────────────────────────────────────────────────────────
st.divider()
with st.expander("⚙️ System Architecture & Flow", expanded=False):
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        IntelliWear Pipeline                     │
    └─────────────────────────────────────────────────────────────────┘

    ① FITBIT API (OAuth 2.0)
       └── /activities/heart/date/today/1d.json
           → avg_hr (weighted from heartRateZones)
           → resting_hr

    ② FEATURE ENGINEERING
       ├── HR_max  = 220 - age
       ├── HR_norm = (HR - HR_rest) / (HR_max - HR_rest)
       ├── RPE     = HR_norm × 10
       ├── Sweat Rate = (0.2 × RPE) + (0.01 × temperature)
       ├── Fluid Loss = sweat_rate × (duration / 60)
       ├── Stress, Fatigue, Recovery, Readiness scores
       └── BMI, gender encoding

    ③ LIGHTGBM MODEL  (12 features → 3 classes)
       └── Hydrated · Mild Dehydration · Severe Dehydration
           + confidence scores per class

    ④ AI RECOMMENDATION ENGINE
       ├── Water intake (ml/hr)
       ├── Electrolyte advice (sodium / potassium)
       └── Actionable guidance based on label + RPE + sweat rate

    ⑤ DASHBOARD + CHATBOT
       └── Live metrics, ML scores, recommendations, contextual Q&A
    ```
    """)

st.markdown(
    '<div style="text-align:center;font-family:\'DM Mono\',monospace;font-size:10px;color:#1A3050;padding:20px 0;">'
    'IntelliWear © 2025 · Fitbit API + LightGBM + Streamlit · No manual physiological input'
    '</div>',
    unsafe_allow_html=True,
)

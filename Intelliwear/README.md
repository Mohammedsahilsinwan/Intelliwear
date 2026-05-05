# 💧 IntelliWear — AI Hydration Monitoring System

## Architecture
```
Fitbit API (OAuth 2.0)
    → Feature Engineering (HR_norm, RPE, Sweat Rate, Stress/Fatigue/Recovery)
        → LightGBM Classifier (3 classes: Hydrated / Mild / Severe)
            → AI Recommendations (water ml/hr, electrolytes, advice)
                → Streamlit Dashboard + Chatbot
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run intelliwear_app.py
```

The app runs at **http://localhost:8501**

---

## Fitbit OAuth 2.0 Setup

### You've already registered IntelliWear on Fitbit — here's how to connect:

1. Go to **https://dev.fitbit.com/apps** and open your IntelliWear app
2. Note your **Client ID** and **Client Secret**
3. Set **Redirect URI** to: `http://localhost:8501` (must match what you registered)
4. Set **OAuth 2.0 Application Type** to: `Personal` (for heart rate access)
5. Make sure **Scopes** include: `heartrate`, `activity`, `profile`

### In the app:
1. Sidebar → select **"⌚ Live Fitbit API"**
2. Enter your **Client ID** and **Client Secret**
3. Click the **"Authorize on Fitbit"** link → log in → you'll be redirected
4. Copy the `code=XXXX` value from the redirect URL
5. Paste it in the **"Paste Auth Code here"** box
6. Click **"🔑 Get Token"** — you're connected!
7. Set temperature and duration sliders
8. Click **"🔄 Fetch & Analyse"**

---

## Demo Mode (no Fitbit required)

Use the sidebar → **"🎭 Demo Mode"** to try 4 pre-built scenarios:
- 🔴 Critical — Intense workout, hot day
- 🟡 Moderate — Jogging session
- 🟠 Low Risk — Light walk
- 🟢 Optimal — Rest / recovery

---

## Feature Engineering

| Feature | Formula |
|---------|---------|
| HR_max | `220 - age` |
| HR_norm | `(HR - HR_rest) / (HR_max - HR_rest)` |
| RPE | `HR_norm × 10` |
| Sweat Rate | `(0.2 × RPE) + (0.01 × temperature)` L/hr |
| Sodium Loss | `sweat_rate × 1000` mg/hr |
| Stress | `HR_norm × 70 + (duration/300) × 30` |
| Fatigue | `(duration/300) × 60 + HR_norm × 40` |
| Recovery | `100 - (stress + fatigue) / 2` |

---

## ML Model

- **Algorithm:** LightGBM (gradient-boosted decision trees)
- **Input:** 12 engineered features
- **Output:** 3-class probability distribution
  - 0 = Hydrated
  - 1 = Mild Dehydration  
  - 2 = Severe Dehydration
- **Model file:** `intelliwear_lgbm.pkl` (auto-generated on first run if missing)

---

## File Structure

```
intelliwear_app.py       ← Main Streamlit application
intelliwear_lgbm.pkl     ← Pre-trained LightGBM model (auto-generated)
requirements.txt         ← Python dependencies
README.md                ← This file
```

---

## Chatbot Commands

| Ask | Response |
|-----|---------|
| "How is my hydration?" | Current status + confidence |
| "What should I drink?" | Water ml/hr recommendation |
| "What about sodium?" | Electrolyte breakdown |
| "Explain LightGBM" | Model architecture details |
| "How does Fitbit work?" | OAuth 2.0 flow explanation |
| "What is my sweat rate?" | Sweat analysis with formula |
| "Show my scores" | Stress / Fatigue / Recovery / Readiness |

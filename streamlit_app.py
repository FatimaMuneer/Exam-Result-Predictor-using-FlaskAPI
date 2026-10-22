import os
import textwrap

import joblib
import numpy as np
import streamlit as st

# --- Page configuration ---
st.set_page_config(
    page_title="ScholarIQ • Academic Success Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# --- Load Model Directly ---
@st.cache_resource
def load_predictor_model():
    return joblib.load("model.pkl")


try:
    model = load_predictor_model()
except Exception as e:
    st.error(
        "🔌 Could not find or load `model.pkl`. Please verify that `model.pkl` is pushed to your GitHub repository root."
    )
    st.stop()


# --- Styling: soft pastel palette, rounded cards, gentle motion ---
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #2b2350;
    --muted: #6b6490;
    --lavender: #8b5cf6;
    --pink: #f472b6;
    --mint: #34d399;
    --rose: #fb7185;
    --line: #ebe6fb;
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
    color: var(--ink);
}

/* Pastel background with soft colour glows */
.stApp {
    background:
        radial-gradient(900px 520px at 8% -8%, #efe8ff 0%, transparent 60%),
        radial-gradient(800px 480px at 100% 0%, #ffe7f3 0%, transparent 55%),
        radial-gradient(700px 420px at 50% 110%, #e5faf2 0%, transparent 60%),
        #fbfaff;
}
.block-container { max-width: 760px; padding-top: 2rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Widget labels readability */
.stApp label, .stApp [data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 600; }
.stApp [data-testid="stTickBarMin"], .stApp [data-testid="stTickBarMax"] { color: var(--muted); }

/* ---------- Hero ---------- */
.hero { text-align: center; margin: 4px 0 26px; animation: rise .6s ease both; }
.hero-emoji { font-size: 3rem; line-height: 1; display: inline-block; animation: bob 3.2s ease-in-out infinite; }
.badge {
    display: inline-block; margin: 10px 0 6px; padding: 5px 14px;
    background: #f3efff; color: #6d28d9; border: 1px solid #ddd6fe; border-radius: 999px;
    font-size: .74rem; font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
}
.hero-title { font-size: 2.5rem; font-weight: 800; letter-spacing: -1px; margin: 0; color: var(--ink); }
.hero-title span {
    background: linear-gradient(135deg, var(--lavender), var(--pink));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-desc { color: var(--muted); font-size: .98rem; max-width: 520px; margin: 8px auto 0; }

.section-title { font-size: 1.02rem; font-weight: 700; margin: 6px 0 10px; color: var(--ink); }

/* Bordered container that holds the sliders */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, .85);
    border: 1px solid var(--line);
    border-radius: 20px;
    box-shadow: 0 10px 30px -12px rgba(139, 92, 246, .22);
}

/* ---------- Live summary chips ---------- */
.chips { display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 4px; }
.chip {
    flex: 1 1 220px; background: #fff; border: 1px solid var(--line); border-radius: 18px;
    padding: 14px 18px; box-shadow: 0 6px 18px -10px rgba(139, 92, 246, .3);
}
.chip-label { font-size: .78rem; font-weight: 700; color: var(--muted); letter-spacing: .3px; }
.chip-value { font-size: 1.5rem; font-weight: 800; margin: 2px 0 6px; color: var(--ink); }
.chip-tag {
    display: inline-block; font-size: .74rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
    background: #f3efff; color: #6d28d9;
}
.chip-tag.pink { background: #ffeaf5; color: #be185d; }

/* ---------- Result cards ---------- */
.result { border-radius: 22px; padding: 22px 24px; margin-top: 8px; animation: rise .5s ease both; }
.result.pass { background: linear-gradient(135deg, #ecfdf5, #f0fdfa); border: 1.5px solid #86efac; color: #14532d; }
.result.risk { background: linear-gradient(135deg, #fff7ed, #fff1f2); border: 1.5px solid #fdba9b; color: #7c2d12; }
.result-row { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
.gauge {
    width: 108px; height: 108px; border-radius: 50%; display: grid; place-items: center; flex: none;
}
.gauge-inner {
    width: 80px; height: 80px; border-radius: 50%; background: #fff; display: grid; place-items: center;
    font-weight: 800; font-size: 1.25rem; color: var(--ink); line-height: 1.1; text-align: center;
}
.gauge-inner small { display: block; font-size: .62rem; font-weight: 700; color: var(--muted); letter-spacing: .4px; }
.result-text { flex: 1 1 260px; }
.result-title { font-size: 1.45rem; font-weight: 800; margin: 0 0 4px; }
.result-msg { font-size: .96rem; margin: 0; opacity: .92; }

.tips { margin-top: 16px; padding-top: 14px; border-top: 1px dashed rgba(124, 45, 18, .25); }
.tips-title { font-weight: 800; font-size: .92rem; margin-bottom: 6px; }
.tips ul { margin: 0; padding-left: 1.1rem; }
.tips li { margin: 3px 0; font-size: .92rem; }

.footnote { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 26px; }

/* ---------- Primary button ---------- */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
    color: #ffffff; font-weight: 700; font-size: 1rem; border-radius: 16px; border: none;
    padding: .75rem 1.4rem; transition: transform .2s ease, box-shadow .2s ease;
    box-shadow: 0 8px 22px -6px rgba(236, 72, 153, .45);
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px); color: #ffffff;
    box-shadow: 0 12px 28px -6px rgba(139, 92, 246, .5);
}
div.stButton > button:first-child:active { transform: translateY(0); }

@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes bob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""


def html(markup: str) -> None:
    cleaned = "\n".join(
        line.strip()
        for line in textwrap.dedent(markup).splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def card():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def study_vibe(hours: float) -> str:
    if hours < 2:
        return "🌙 Light"
    if hours < 4:
        return "☕ Steady"
    if hours < 6:
        return "📖 Focused"
    if hours < 8:
        return "🔥 Dedicated"
    return "🚀 Marathon"


def score_band(score: float) -> str:
    if score < 40:
        return "🌱 Growing"
    if score < 60:
        return "🌤️ Building"
    if score < 75:
        return "⭐ Solid"
    if score < 90:
        return "🏅 Strong"
    return "👑 Outstanding"


def friendly_tips(hours: float, score: float) -> list[str]:
    tips = []
    if hours < 3:
        tips.append(
            "⏰ Add a steady 1–2 focused hours a day. Consistency beats last-minute cramming."
        )
    if score < 50:
        tips.append(
            "🧩 Revisit the topics that cost you marks last time and redo a few past questions."
        )
    tips.append(
        "🧠 Quiz yourself instead of re-reading. Active recall helps knowledge stick."
    )
    tips.append(
        "💤 Protect your sleep and take short breaks. Rested minds learn faster."
    )
    return tips


def render_result(result: dict, study_hours: float, prev_score: float) -> None:
    is_pass = int(result["prediction"]) == 1
    conf = max(0.0, min(1.0, float(result["confidence"]))) * 100

    if is_pass:
        st.balloons()
        kind, emoji, ring = "pass", "🎉", "#34d399"
        title = "Predicted outcome: PASS"
        message = "Great news! The candidate displays robust readiness for examination clearance."
        tips_html = ""
    else:
        kind, emoji, ring = "risk", "🌱", "#fb923c"
        title = "Predicted outcome: AT RISK"
        message = "Additional targeted review and focused hours are recommended to reach passing thresholds."
        items = "".join(
            f"<li>{t}</li>" for t in friendly_tips(study_hours, prev_score)
        )
        tips_html = f'<div class="tips"><div class="tips-title">💡 Friendly study tips</div><ul>{items}</ul></div>'

    html(
        f"""
        <div class="result {kind}">
            <div class="result-row">
                <div class="gauge" style="background: conic-gradient({ring} {conf:.1f}%, #ece8f7 0);">
                    <div class="gauge-inner"><div>{conf:.0f}%<small>CERTAINTY</small></div></div>
                </div>
                <div class="result-text">
                    <div class="result-title">{emoji} {title}</div>
                    <p class="result-msg">{message}</p>
                </div>
            </div>
            {tips_html}
        </div>
        """
    )

    with st.expander("🔍 View technical model diagnostics"):
        st.markdown(f"**Model certainty score:** `{conf:.1f}%`")
        st.json(result)


# ------------------------------------------------------------------ Page ---
html(STYLE)

html(
    """
    <div class="hero">
        <div class="hero-emoji">🎓</div><br>
        <span class="badge">✨ Machine learning engine</span>
        <h1 class="hero-title">Scholar<span>IQ</span></h1>
        <p class="hero-desc">Estimate your academic outcome from your study habits, powered by a logistic regression model.</p>
    </div>
    """
)

html('<div class="section-title">📚 Your study profile</div>')
with card():
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider(
            "⏱️ Study hours per day",
            min_value=0.0,
            max_value=10.0,
            value=4.5,
            step=0.5,
            help="Daily dedicated academic study time",
        )
    with col2:
        prev_score = st.slider(
            "📝 Previous exam score",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            help="Benchmark grade achieved in previous examination cycle",
        )

# Live summary of current inputs
html(
    f"""
    <div class="chips">
        <div class="chip">
            <div class="chip-label">⏱️ DAILY COMMITMENT</div>
            <div class="chip-value">{study_hours:g} hrs/day</div>
            <span class="chip-tag">{study_vibe(study_hours)}</span>
        </div>
        <div class="chip">
            <div class="chip-label">📈 PRIOR PERFORMANCE</div>
            <div class="chip-value">{int(prev_score)} / 100</div>
            <span class="chip-tag pink">{score_band(prev_score)}</span>
        </div>
    </div>
    """
)

st.write("")

if st.button("✨ Analyze my academic outcome", use_container_width=True):
    try:
        with st.spinner("Crunching the numbers…"):
            # Direct model inference (no localhost server needed)
            input_data = np.array([[float(study_hours), float(prev_score)]])
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            confidence = float(probabilities[prediction])

            result_payload = {
                "prediction": prediction,
                "result_label": "Pass" if prediction == 1 else "Fail",
                "confidence": round(confidence, 3),
            }

            render_result(result_payload, study_hours, prev_score)

    except Exception as error:
        st.error(f"Something went wrong while generating the outcome: {error}")

html(
    """
    <div class="footnote">
        ScholarIQ gives statistical estimates, not guarantees. It cannot see effort, health or circumstances,
        so treat the result as a friendly nudge, not a verdict. 💜
    </div>
    """
)
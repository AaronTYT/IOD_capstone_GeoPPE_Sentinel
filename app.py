# ui.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
import os
import pandas as pd
import plotly.express as px
from scipy.stats import norm

st.set_page_config(page_title="GeoPPE Sentinel Dashboard", 
                   page_icon="🛡️", 
                   layout="wide"
                )

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "https://fatigue-risk-api.onrender.com/predict"))

# EXEC_PBI_EMBED = st.secrets.get("EXEC_PBI_EMBED", os.getenv("EXEC_PBI_EMBED", ""))     # Power BI embed URL
# CCTV_PBI_EMBED = st.secrets.get("CCTV_PBI_EMBED", os.getenv("CCTV_PBI_EMBED", ""))     # Power BI embed URL
# GEOF_PBI_EMBED = st.secrets.get("GEOF_PBI_EMBED", os.getenv("GEOF_PBI_EMBED", ""))     # Power BI embed URL

# # Optional fallback "open in new tab" links (Power BI report links)
# EXEC_PBI_LINK = st.secrets.get("EXEC_PBI_LINK", os.getenv("EXEC_PBI_LINK", ""))
# CCTV_PBI_LINK = st.secrets.get("CCTV_PBI_LINK", os.getenv("CCTV_PBI_LINK", ""))
# GEOF_PBI_LINK = st.secrets.get("GEOF_PBI_LINK", os.getenv("GEOF_PBI_LINK", ""))

# -----------------------------
# HELPERS
# -----------------------------
def embed_powerbi(embed_url: str, height: int = 820):
    """Embeds a Power BI report using an iframe."""
    if not embed_url:
        st.warning("Power BI embed URL not set yet. Add it to secrets/env and reload.")
        return

    iframe = f"""
    <iframe
        width="100%"
        height="{height}"
        src="{embed_url}"
        frameborder="0"
        allowFullScreen="true">
    </iframe>
    """
    st.components.v1.html(iframe, height=height)

def open_link_button(label: str, url: str):
    if url:
        st.link_button(label, url)
    else:
        st.caption("No external report link set.")

@st.cache_data
def load_population():
    return pd.read_csv("data/fatigue_powerbi_2000_workers.csv")

population_df = load_population()

def call_fatigue_api(payload: dict) -> dict:
    """Calls your FastAPI fatigue endpoint."""
    r = requests.post(API_URL, json=payload, timeout=20)
    # Raise useful error if non-200
    r.raise_for_status()
    return r.json()

# -----------------------------
# HEADER
# -----------------------------
st.title("🛡️ GeoPPE Sentinel - Mine Safety Control Room")
st.caption("Caption here")

# TABS
executive_tab, cctv_tab, geo_fence_tab, fatigue_tab = st.tabs([
    "1) Exectuive Dashboard 📊",
    "2) CCTV - OPF & PPE Dashboard 🎥",
    "3) GeoFence Map - Spatial Risk View 🗺️",
    "4) Fatigue Risk Predictor 🔮"
])

# -----------------------------
# TAB 1: EXECUTIVE
# -----------------------------
with executive_tab:
    st.subheader("Executive Dashboard 📊")
    colA, colB = st.columns([3,1], vertical_alignment="top")

    with colB:
        st.markdown("### Report access")
        # open_link_button("Open Executive Report", EXEC_PBI_LINK)
        st.markdown("---")
        st.caption("Tip: use the embed URL for in-app viewing, and the report link for opening in a new tab.")

    with colA:
        st.caption("colA - Executive")
        # embed_powerbi(EXEC_PBI_EMBED, height=850)

# -----------------------------
# TAB 2: CCTV
# -----------------------------
with cctv_tab:
    st.subheader("CCTV - OPF & PPE Dashboard 🎥")
    colA, colB = st.columns([3,1], vertical_alignment="top")

    with colB:
        st.markdown("### Report access")
        # open_link_button("Open Executive Report", EXEC_PBI_LINK)
        st.markdown("---")
        st.caption("Tip: use the embed URL for in-app viewing, and the report link for opening in a new tab.")

    with colA:
        st.caption("colA - CCTV")
        # embed_powerbi(EXEC_PBI_EMBED, height=850)

# -----------------------------
# TAB 3: GEOFENCE
# -----------------------------
with geo_fence_tab:
    st.subheader("GeoFence Map - Spatial Risk View 🗺️")
    colA, colB = st.columns([3,1], vertical_alignment="top")

    with colB:
        st.markdown("### Report access")
        # open_link_button("Open Executive Report", EXEC_PBI_LINK)
        st.markdown("---")
        st.caption("Tip: use the embed URL for in-app viewing, and the report link for opening in a new tab.")

    with colA:
        st.caption("colA - GeoFence")
        # embed_powerbi(EXEC_PBI_EMBED, height=850)

# -----------------------------
# TAB 4: FATIGUE PREDICTOR
# -----------------------------
with fatigue_tab:
    st.subheader("Fatigue Risk Predictor 🔮")
    st.caption("Enter work conditions:")

    left, right = st.columns([1,1], vertical_alignment="top")

    with left:
        with st.form("fatigue_form", clear_on_submit=False):
            sleep_hours = st.slider("Sleep hours", 1.0, 10.0, 6.0, 0.1)
            shift_hours = st.slider("Shift hours", 1.0, 14.0, 11.0, 0.1)
            opf_minutes = st.slider("OPF minutes (exposure)", 0.0, 300.0, 45.0, 0.1)
            ppe_violations = st.number_input("PPE violations", 0, 10, 1, 1)
            high_risk_events = st.number_input("High-risk events", 0, 20, 2, 1)
            break_compliance = st.slider("Break compliance", 0.0, 1.1, 0.7, 0.1)
            movement_score = st.slider("Movement score", 0.1, 1.0, 0.5, 0.1)

            submitted = st.form_submit_button("Predict fatigue risk")

    with right:
        if submitted:
            payload = {
                "sleep_hours": float(sleep_hours),
                "shift_hours": float(shift_hours),
                "opf_minutes": float(opf_minutes),
                "ppe_violations": int(ppe_violations),
                "high_risk_events": int(high_risk_events),
                "break_compliance": float(break_compliance),
                "movement_score": float(movement_score),
            }

            try:
                result = call_fatigue_api(payload)
                label = result.get("fatigue_label", "Unknown")
                probs = result.get("probabilities", {})

                st.markdown("## Prediction:")

                if label == "Severe":
                    emoji = "🚨"
                elif label == "High":
                    emoji = "🔴"
                elif label == "Moderate":
                    emoji = "🟡"
                else:
                    emoji = "🟢"
                
                percentage = probs[label] * 100
                
                # Expand the prediction recommendation more
                st.metric(
                    "Predicted fatigue risk:",
                    f"{emoji} {label} ({percentage:.1f}%)"
                )

                # Risk proxy
                worker_risk_score = probs["High"] + probs["Severe"]

                # -----------------------------
                # POPULATION RISK DISTRIBUTION
                # -----------------------------
                # Create population proxy if not already present
                if "risk_proxy" not in population_df.columns:
                    # Approximate proxy using fatigue_label
                    population_df["risk_proxy"] = population_df["fatigue_label"].map({
                        "Low": 0.1,
                        "Moderate": 0.3,
                        "High": 0.6,
                        "Severe": 0.9
                    })

                pop_scores = population_df["risk_proxy"]
                percentile = (pop_scores < worker_risk_score).mean() * 100
                
                # -----------------------------
                # PREPARE DISTRIBUTION
                # -----------------------------
                population_scores = pop_scores.values  # from your CSV
                worker_score = worker_risk_score        # from API

                # Fit normal distribution to population
                mu, sigma = norm.fit(population_scores)

                # X range for bell curve
                x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
                pdf = norm.pdf(x, mu, sigma)

                # Percentile calculation
                percentile = norm.cdf(worker_score, mu, sigma) * 100

                # -----------------------------
                # CREATE FIGURE
                # -----------------------------
                fig = go.Figure()

                # Bell curve
                fig.add_trace(go.Scatter(
                    x=x,
                    y=pdf,
                    mode="lines",
                    name="Workforce Distribution",
                    line=dict(color="#607D8B", width=3)
                ))

                # Shaded area (up to worker)
                mask = x <= worker_score
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x[mask], [worker_score]]),
                    y=np.concatenate([pdf[mask], [0]]),
                    fill="tozeroy",
                    mode="lines",
                    name="Percentile Area",
                    line=dict(color="rgba(244,67,54,0.3)"),
                    fillcolor="rgba(244,67,54,0.3)"
                ))

                # Vertical worker line
                fig.add_trace(go.Scatter(
                    x=[worker_score, worker_score],
                    y=[0, norm.pdf(worker_score, mu, sigma)],
                    mode="lines",
                    name="Current Worker",
                    line=dict(color="red", width=3, dash="dash")
                ))

                # -----------------------------
                # LAYOUT
                # -----------------------------
                fig.update_layout(
                    title="Worker Fatigue Risk Compared to Workforce",
                    xaxis_title="Fatigue Risk (Population)",
                    yaxis_title="Density",
                    showlegend=True,
                    template="simple_white",
                    height=420
                )

                # -----------------------------
                # DISPLAY
                # -----------------------------
                st.plotly_chart(fig, use_container_width=True)


                # -----------------------------
                # INTERPRETATION
                # -----------------------------
                st.markdown("### Interpretation")

                st.info(
                    f"This worker is more fatigued than **{percentile:.1f}%** of the workforce. "
                    "Higher percentiles indicate elevated fatigue risk compared to peers."
                )


            except requests.HTTPError as e:
                st.error(f"API returned an error: {e}")
                st.caption("Check Render logs if this persists.")
            except Exception as e:
                st.error(f"Request failed: {e} Rerun render")

st.markdown("---")
st.caption("© 2025 GeoPPE Sentinel: AI-Powered PPE & Fatigue Safety for WA MineSites Project — Aaron Tan")
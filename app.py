# app.py 
# Import python libraries:
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
import os
import pandas as pd
import plotly.express as px
from scipy.stats import norm
from ultralytics import YOLO

st.set_page_config(page_title="GeoPPE Sentinel Dashboard", 
                   page_icon="🛡️", 
                   layout="wide"
                )


# Load trained YOLOv8 model
model = YOLO("model/best.pt")  # path to your .pt file

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "https://fatigue-risk-api.onrender.com/predict"))

# -----------------------------
# HELPERS
# -----------------------------
# -----------------------------
# PPE Compliance Evaluation
# -----------------------------

REQUIRED_PPE = ["helmet", "vest"]
CONF_THRESHOLD = 0.6


def evaluate_ppe_compliance(detections):
    """
    Evaluates PPE compliance based on model detections.

    Parameters:
    detections (list): List of dicts with keys 'class' and 'conf'

    Returns:
    status (str): 'Compliant' or 'Non-Compliant'
    confidence (float): Aggregated confidence score
    missing_ppe (list): List of missing PPE items
    """

    detected = {}

    # Keep highest confidence per PPE item
    for det in detections:
        cls = det["class"].lower()
        conf = det["conf"]

        if cls not in detected or conf > detected[cls]:
            detected[cls] = conf

    missing_ppe = []
    confidence_scores = []

    for ppe in REQUIRED_PPE:
        if ppe not in detected or detected[ppe] < CONF_THRESHOLD:
            missing_ppe.append(ppe)
        else:
            confidence_scores.append(detected[ppe])

    if missing_ppe:
        status = "Non-Compliant"
    else:
        status = "Compliant"

    overall_confidence = round(
        sum(confidence_scores) / len(confidence_scores), 2
    ) if confidence_scores else 0.0

    return status, overall_confidence, missing_ppe


@st.cache_data
def load_population():
    return pd.read_csv("data/fatigue_powerbi_2000_workers.csv")

population_df = load_population()

def call_fatigue_api(payload: dict) -> dict:
    """Calls your FastAPI fatigue endpoint."""
    r = requests.post(API_URL, json=payload, timeout=10)
    # Raise useful error if non-200
    r.raise_for_status()
    return r.json()

# -----------------------------
# HEADER
# -----------------------------
st.title("🛡️ GeoPPE Sentinel - Mine Safety Control Room")
st.markdown("""
**GeoPPE Sentinel** is an AI-powered mine safety monitoring proof of concept designed to 
demonstrate how visual monitoring and spatial awareness can be combined to support safer 
operations in mining environments.

The system simulates a **centralised safety control room** by bringing together:
- **Executive-level dashboards** to support informed decision-making 
- **CCTV monitoring** to assess PPE compliance  
- **Location-based risk awareness** using geofencing concepts  
- **Fatigue Risk** to assess a worker's fatigue 

This project focuses on **practical safety insights**, showing how visual observations 
can be converted into clear safety outcomes rather than technical model outputs.

""")



# TABS
# Executive, CCTV, and Geo-Fence dashboards
# Power BI dashboards were developed in Power BI Desktop.
# Deployment to Power BI Service requires an organisational Microsoft account.
# Due to tenant eligibility restrictions, enterprise-style Power BI screenshots
# are used to demonstrate the proof of concept.
executive_tab, cctv_tab, geo_fence_tab, fatigue_tab = st.tabs([
    "1) Exectuive Dashboard 📊",
    "2) CCTV – PPE Monitoring 🎥",
    "3) GeoFence Map - Spatial Risk View 🗺️",
    "4) Fatigue Risk Predictor 🔮"
])

# -----------------------------
# TAB 1: EXECUTIVE
# -----------------------------
with executive_tab:
    st.subheader("1️⃣ Executive Dashboard 📊")

    st.markdown(
    "View a high-level summary of safety performance, including PPE compliance trends and key risk indicators," \
    "designed for supervisors and operational leaders.")
    st.image("data/executive.png", caption="")

# -----------------------------
# TAB 2: CCTV
# -----------------------------
with cctv_tab:
    st.subheader("2️⃣ **CCTV – PPE Monitoring 🎥**")

    st.markdown("""
    Review example CCTV scenarios that demonstrate:

    - Workers meeting PPE requirements  
    - Instances of PPE non-compliance  
    - Clear safety outcomes with confidence indicators  

    You may also upload a CCTV image to see how the system assesses PPE compliance.
    """)


    tab1, tab2, tab3 = st.tabs([
        "✅ Compliant",
        "❌ Non-Compliant",
        "🧪 Test Your Own"
    ])

    with tab1:
        st.markdown("### Example: PPE Compliant Worker ✅")

                
        col1, col2 = st.columns([3, 2])

        with col1:
            st.video("YOLO_Videoes/YOLO_videoes_sample_results/ppe_demo_output_h264.mp4")

        with col2:
            detections = [
                {"class": "helmet", "conf": 0.93},
                {"class": "vest", "conf": 0.89}
            ]

            # Evaluate compliance
            status, confidence, missing = evaluate_ppe_compliance(detections)

            # Display AI output
            st.markdown("#### Results Output:")

            st.success(f"Status: {status}")
            st.metric("Confidence Score", f"{confidence * 100:.1f}%")

            if missing:
                st.warning(f"Missing PPE: {', '.join(missing)}")
    
    with tab2:
        st.markdown("### Example: PPE Non-Compliant Worker ❌")

        col1, col2 = st.columns([3, 2])

        # --- CCTV Video ---
        with col1:
            st.video(
                "YOLO_Videoes/YOLO_videoes_sample_results/ppe_no_vest_demo_output_h264.mp4",
                format="video/mp4"
            )

        with col2:
            # Simulated YOLO detections (Vest missing)
            detections = [
                {"class": "helmet", "conf": 0.93}
            ]

            # Evaluate compliance
            status, confidence, missing = evaluate_ppe_compliance(detections)

            st.markdown("#### Results Output:")

            st.error(f"Status: {status}")
            st.metric("Confidence Score", f"{confidence * 100:.1f}%")


    with tab3:
        st.markdown("### Test Your Own CCTV Image/Video")

        uploaded = st.file_uploader(
            "Upload a CCTV image/video",
            type=["jpg", "png", "mp4"]
        )

        # if uploaded:
        #     st.image(uploaded, caption="Uploaded CCTV Frame")

        #     # Example: run YOLO model here
        #     detections = run_model_on_image(uploaded)

        #     status, confidence, missing = evaluate_ppe_compliance(detections)

        #     if status == "Compliant":
        #         st.success(f"Status: {status}")
        #     else:
        #         st.error(f"Status: {status}")

        #     st.metric("Confidence Score", f"{confidence*100:.1f}%")

        #     if missing:
        #         st.warning(f"Missing PPE: {', '.join(missing)}")

    
    
# -----------------------------
# TAB 3: GEOFENCE
# -----------------------------
with geo_fence_tab:
    st.subheader("3️⃣ GeoFence Map - Spatial Risk View 🗺️")
    st.subheader("Rio Tinto – Marandoo Mine Site OPF")
    st.markdown(
    "📍[View location on Google Maps](https://www.google.com/maps/place/Marandoo+Mine+Site/@-22.6363653,118.1185468,927m/data=!3m1!1e3!4m6!3m5!1s0x2bf292a3a2ffb34b:0x2e6673fa666c0dce!8m2!3d-22.6367416!4d118.1218566!16s%2Fm%2F0dsd4xb?entry=ttu&g_ep=EgoyMDI1MTIwOS4wIKXMDSoASAFQAw%3D%3D)"
    )

    st.markdown("""Explore a visual representation of the mine site showing operational zones and 
                restricted areas. This view demonstrates how worker location can increase or reduce 
                safety risk when combined with CCTV observations.
                """)

    st.markdown("""
    **Purpose**  
    This geospatial view represents a proof-of-concept geofencing risk layer for the Marandoo mine site. 
    It provides spatial context to support operational safety, hazard awareness, and workforce monitoring.

    **Note**  
    This static map is used to demonstrate spatial risk logic due to enterprise platform and 
    deployment limitations. In a production environment, this would be implemented as an 
    interactive, real-time geospatial dashboard.
    """)

    st.image("data/geospatial.png", caption="")
    

# -----------------------------
# TAB 4: FATIGUE PREDICTOR
# -----------------------------
with fatigue_tab:
    st.subheader("4️⃣ Fatigue Risk Predictor 🔮")
    st.markdown("""Use this section to assess potential worker fatigue risk based on reported or simulated 
                shift and work pattern information. The tool provides a **simple risk classification** 
                (e.g. low, medium, high) to support early intervention and workforce wellbeing planning.""")
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
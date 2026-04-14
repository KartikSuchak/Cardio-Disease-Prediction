import streamlit as st
import plotly.graph_objects as go
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/ui_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Hide Streamlit Header */
header {visibility: hidden;}

/* Main Background */
.main {
    background-color: #f4f7fb;
}

/* Container Padding */
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Title */
.title-main {
    font-size: 42px;
    font-weight: 800;
    color: #1f2c56;
    margin-bottom: 0px;
}

/* Subtitle */
.subtitle {
    color: #6c757d;
    font-size: 18px;
    margin-bottom: 10px;
}

/* Accent Line */
.accent-line {
    height:4px;
    width:120px;
    background:linear-gradient(to right,#3D5AFE,#00C9A7);
    border-radius:10px;
    margin-bottom:25px;
}

/* Premium Metric Cards */
.metric-card {
    background: linear-gradient(145deg, #ffffff, #f1f5ff);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    text-align: center;
    transition: 0.3s;
    border: 1px solid #eef2ff;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(61,90,254,0.12);
}

/* Panel Cards */
.panel-card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

/* Section Headers */
h3 {
    color: #1f2c56 !important;
    font-weight: 700 !important;
}

/* Prediction Box */
.prediction-box {
    padding: 18px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
}

/* Input Fields */
.stNumberInput, .stSelectbox, .stTextInput {
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title-main'>❤️ Healthcare Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time Cardiovascular Risk Monitoring System</div>", unsafe_allow_html=True)
st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)

# ---------------- TOP METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h4>📊 Model Accuracy</h4>
        <h2 style='color:#3D5AFE;'>72.81%</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h4>📁 Dataset Records</h4>
        <h2 style='color:#3D5AFE;'>68,706</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <h4>🧠 Features Used</h4>
        <h2 style='color:#3D5AFE;'>11</h2>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN DASHBOARD ----------------
left, center, right = st.columns([1.2, 2, 1.2])

# LEFT PANEL
with left:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)

    st.subheader("📋 Patient Inputs")

    age = st.number_input("Age", 1, 120, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)

    bp_input = st.text_input("Blood Pressure (120/80)", "120/80")

    try:
        ap_hi, ap_lo = map(int, bp_input.split("/"))
    except:
        st.error("Enter BP as 120/80")
        ap_hi, ap_lo = 120, 80

    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT PANEL
with right:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)

    st.subheader("🧪 Health Factors")

    chol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "High"])
    gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "High"])
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    alco = st.selectbox("Alcohol", ["No", "Yes"])
    active = st.selectbox("Active Lifestyle", ["Yes", "No"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREPROCESS INPUT ----------------
chol_val = ["Normal", "Above Normal", "High"].index(chol) + 1
gluc_val = ["Normal", "Above Normal", "High"].index(gluc) + 1
smoke_val = 1 if smoke == "Yes" else 0
alco_val = 1 if alco == "Yes" else 0
active_val = 1 if active == "Yes" else 0
gender_val = 2 if gender == "Male" else 1

# ---------------- MODEL PREDICTION ----------------
input_data = np.array([[
    age,
    gender_val,
    height,
    weight,
    ap_hi,
    ap_lo,
    chol_val,
    gluc_val,
    smoke_val,
    alco_val,
    active_val
]])

scaled_input = scaler.transform(input_data)

prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]

risk_percent = probability * 100

# CENTER PANEL
with center:

    st.write("")
    st.write("")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        title={'text': "Disease Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4F46E5"},
            'steps': [
                {'range': [0, 40], 'color': "#B9F6CA"},
                {'range': [40, 70], 'color': "#FFF59D"},
                {'range': [70, 100], 'color': "#FF8A80"}
            ]
        }
    ))

    fig.update_layout(height=420)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Prediction")

    if prediction == 1:
        st.markdown("""
        <div class='prediction-box' style='background:#ffe5e5;color:#c62828;'>
            🔴 HIGH RISK OF CARDIOVASCULAR DISEASE
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='prediction-box' style='background:#e8f5e9;color:#2e7d32;'>
            🟢 LOW RISK OF CARDIOVASCULAR DISEASE
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='metric-card'>
        <h4>📈 Risk Score</h4>
        <h2 style='color:#4F46E5;'>{risk_percent:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center style='color:gray;'>
Healthcare Analytics Dashboard • Powered by Apache Spark + Machine Learning
</center>
""", unsafe_allow_html=True)
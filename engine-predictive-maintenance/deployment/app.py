
import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# =========================================
# Load Model from Hugging Face Hub
# =========================================

model_path = hf_hub_download(
    repo_id="geniusut/engine-predictive-maintenance-model",
    filename="best_model.pkl"
)

model = joblib.load(model_path)

# =========================================
# Streamlit UI
# =========================================

st.title("Engine Predictive Maintenance")

st.write(
    "Enter engine operational parameters "
    "to predict maintenance requirements."
)

# =========================================
# User Inputs
# =========================================

engine_rpm = st.number_input(
    "Engine RPM",
    value=800.0
)

lub_oil_pressure = st.number_input(
    "Lub Oil Pressure",
    value=3.5
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    value=6.5
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    value=2.5
)

lub_oil_temp = st.number_input(
    "Lub Oil Temperature",
    value=75.0
)

coolant_temp = st.number_input(
    "Coolant Temperature",
    value=80.0
)

# =========================================
# Prepare Input Data
# =========================================

input_data = pd.DataFrame([{
    "Engine_RPM": engine_rpm,
    "Lub_Oil_Pressure": lub_oil_pressure,
    "Fuel_Pressure": fuel_pressure,
    "Coolant_Pressure": coolant_pressure,
    "Lub_Oil_Temperature": lub_oil_temp,
    "Coolant_Temperature": coolant_temp
}])

# =========================================
# Predict
# =========================================

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    # =========================================
    # Display Prediction Result
    # =========================================

    if prediction == 1:

        st.error(
            "🔴 Prediction Result: Maintenance Required"
        )

    else:

        st.success(
            "🟢 Prediction Result: Engine Operating Normally"
        )

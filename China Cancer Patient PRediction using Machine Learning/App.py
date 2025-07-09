import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load("China-Cancer-Patient-Prediction using Machine Learning Algorithms.pkl")

# App title and description
st.title("🧬 China Cancer Patient Survival Prediction")
st.markdown("This app predicts whether a cancer patient is likely to survive based on their health data using a machine learning model.")

# Input fields
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
age = st.slider("Age", 10, 100, 30)
tumor_type = st.selectbox("Tumor Type", ["Breast", "Cervical", "Colorectal", "Liver", "Lung", "Stomach"])
cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
tumor_size = st.slider("Tumor Size (cm)", 0.0, 10.0, 2.5)
treatment_type = st.selectbox("Treatment Type", ["Surgery", "Chemotherapy", "Radiation", "Combined"])
chemo_sessions = st.slider("Number of Chemotherapy Sessions", 0, 50, 10)
radiation_sessions = st.slider("Number of Radiation Sessions", 0, 50, 10)
follow_up_months = st.slider("Follow-up Months", 0, 120, 12)
smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

# Label encoding maps based on your training encoders
gender_map = {"Female": 0, "Male": 1, "Other": 2}
tumor_type_map = {
    "Breast": 0,
    "Cervical": 1,
    "Colorectal": 2,
    "Liver": 3,
    "Lung": 4,
    "Stomach": 5
}
cancer_stage_map = {"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3}
treatment_type_map = {"Surgery": 2, "Chemotherapy": 0, "Radiation": 3, "Combined": 1}
smoking_status_map = {"Never": 2, "Former": 0, "Current": 1}

# Convert input values to model format
input_data = np.array([[ 
    gender_map[gender],
    age,
    tumor_type_map[tumor_type],
    cancer_stage_map[cancer_stage],
    tumor_size,
    treatment_type_map[treatment_type],
    chemo_sessions,
    radiation_sessions,
    follow_up_months,
    smoking_status_map[smoking_status]
]])

# Prediction logic
if st.button("Predict Survival Status"):
    prediction = model.predict(input_data)

    # Interpret prediction
    if prediction[0] == 0:
        st.success("🟢 Prediction: **Likely to Survive**")
    else:
        st.error("🔴 Prediction: **Unlikely to Survive**")

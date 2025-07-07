import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load(r'C:\Users\HP\OneDrive\Desktop\Data Science\Breast Cancer Pediction Using Machine Learning.pkl')

st.set_page_config(page_title="Breast Cancer Prediction App", layout="centered")

st.title("🩺 Breast Cancer Prediction Using Machine Learning")
st.markdown("Enter the patient's medical data below to predict if the tumor is **Benign (0)** or **Malignant (1)**.")

# Feature Inputs
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Create input form
with st.form("input_form"):
    inputs = []
    for feature in features:
        value = st.number_input(f"Enter {feature}:", step=0.01, format="%.4f")
        inputs.append(value)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)
    print(prediction)
    prediction_proba = model.predict_proba(input_array)
    print(prediction_proba)

    st.subheader("🔍 Prediction Result")
    if prediction[0] == 0:
        st.success("The tumor is **Benign (0)** — non-cancerous.")
    else:
        st.error("The tumor is **Malignant (1)** — potentially cancerous.")

    st.subheader("📊 Prediction Probability")
    st.write(f"Benign: {prediction_proba[0][0]:.2f}, Malignant: {prediction_proba[0][1]:.2f}")

st.markdown("---")
st.caption("Developed using Streamlit and Scikit-learn | © 2025")

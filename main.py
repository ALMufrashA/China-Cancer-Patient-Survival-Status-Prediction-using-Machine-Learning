# Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!
# Hey mohan, can we get together to watch footbal game tomorrow?

import streamlit as st
import joblib

# Load pre-trained model and vectorizer
loaded_model = joblib.load('Spam Classification Model.pkl')
loaded_vectorizer = joblib.load('count_vectorizer.pkl')

# Page title
st.markdown("<h1 style='text-align: center; color: red;'>📧 Spam Mail Classification using Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("---")

# # Description

# st.markdown("""
# <div style='text-align: center;'>
#     This webapp uses a Machine Learning model to classify whether an email is 
#     <strong style='color:green;'>Not Spam</strong> or 
#     <strong style='color:red;'>Spam</strong>.
#     <br><br>
#     Just type or paste an email message below and click <strong>Predict</strong> to check.
# </div>
# """, unsafe_allow_html=True)

# Input text area
email_text = st.text_area("✉️ Enter the email content here:", height=200)

# Predict button
if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some text before prediction.")
    else:
        # Transform the input text
        transformed_text = loaded_vectorizer.transform([email_text])

        # Predict using the loaded model
        prediction = loaded_model.predict(transformed_text)
        print(prediction.ndim)

        # Optional: Print raw output for debug
        st.write(f"🔎 Raw prediction value: {prediction[0]}")

        # Display prediction result
        if prediction == 'spam':
            st.markdown("<h3 style='color: red;'>🚨 This email is classified as SPAM.</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>✅ This email is classified as NOT SPAM.</h3>", unsafe_allow_html=True)
        
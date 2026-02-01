# app.py - Streamlit version
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ============================================
# 1. PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# ============================================
# 2. LOAD MODEL (WITH CACHING)
# ============================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ============================================
# 3. CHEST PAIN MAPPING
# ============================================
cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

# ============================================
# 4. PREDICTION FUNCTION
# ============================================
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak):
    if model is None:
        return None, None
    
    try:
        features = np.array([[
            float(age),
            1 if sex == "Male" else 0,
            cp_mapping[cp],
            float(trestbps),
            float(chol),
            1 if fbs == "Yes" else 0,
            float(thalch),
            1 if exang == "Yes" else 0,
            float(oldpeak)
        ]])
        
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak']
        df = pd.DataFrame(features, columns=feature_names)
        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# ============================================
# 5. APP TITLE
# ============================================
st.title("❤️ Heart Disease Prediction System")
st.markdown("Predict the likelihood of heart disease using machine learning.")

# ============================================
# 6. INPUT FORM
# ============================================
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=100, value=55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        )
    
    with col2:
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=90, max_value=200, value=130)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    
    with col3:
        thalch = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression", min_value=-3.0, max_value=7.0, value=1.2, step=0.1)
    
    submitted = st.form_submit_button("🔍 Predict", type="primary")

# ============================================
# 7. PREDICTION RESULTS
# ============================================
if submitted and model is not None:
    prediction, probability = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak)
    
    if prediction is not None:
        st.divider()
        
        if prediction == 1:
            st.error(f"⚠️ **Heart Disease Detected** (Probability: {probability:.2f}%)")
            st.warning("""
            **Recommendations:**
            1. Consult a cardiologist immediately
            2. Maintain a heart-healthy diet
            3. Exercise regularly
            4. Monitor blood pressure and cholesterol
            """)
        else:
            st.success(f"✅ **No Heart Disease Detected** (Probability: {probability:.2f}%)")
            st.info("""
            **Recommendations:**
            1. Continue regular check-ups
            2. Maintain a balanced diet
            3. Exercise at least 30 minutes daily
            4. Stay hydrated
            """)

# ============================================
# 8. SIDEBAR INFO
# ============================================
with st.sidebar:
    st.title("ℹ️ Information")
    st.markdown("""
    **Model Details:**
    - Algorithm: Random Forest
    - Accuracy: 82.07%
    - Features: 9 parameters
    
    **Disclaimer:**
    This tool is for educational purposes only.
    Always consult healthcare professionals.
    """)

# ============================================
# 9. FOOTER
# ============================================
st.divider()
st.caption("Built with ❤️ using Streamlit and Machine Learning")
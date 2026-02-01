# app.py - Complete version with working prediction
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    try:
        # Try joblib first, then pickle
        model = joblib.load('heart_disease_model.pkl')
        return model
    except:
        try:
            with open('heart_disease_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except:
            st.error("❌ Could not load model file")
            return None

model = load_model()

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak):
    """Make prediction with error handling"""
    if model is None:
        return None, None
    
    try:
        # Chest pain mapping
        cp_mapping = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }
        
        # Prepare features
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
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0][1] * 100
        else:
            probability = None
            
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("ℹ️ Information")
    
    st.markdown("""
    ## Model Details:
    - **Algorithm:** Random Forest
    - **Accuracy:** 82.07%
    - **Features:** 9 parameters
    - **Status:** {'✅ Ready' if model else '❌ Not Loaded'}
    """)
    
    st.divider()
    
    st.markdown("""
    **⚠️ Disclaimer:**
    This tool is for educational purposes only.
    Always consult healthcare professionals.
    """)

# ============================================
# MAIN CONTENT
# ============================================
st.title("❤️ Heart Disease Prediction System")
st.markdown("Predict the likelihood of heart disease using machine learning.")

# Create input form
with st.form("prediction_form"):
    st.subheader("📋 Patient Information")
    
    # Three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=20,
            max_value=100,
            value=55,
            help="Patient's age in years"
        )
        
        sex = st.selectbox(
            "Sex",
            ["Male", "Female"],
            help="Patient's gender"
        )
        
        cp = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            help="Type of chest pain experienced"
        )
    
    with col2:
        trestbps = st.number_input(
            "Resting BP (mmHg)",
            min_value=90,
            max_value=200,
            value=130,
            help="Resting blood pressure"
        )
        
        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=250,
            help="Serum cholesterol level"
        )
        
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"],
            help="Fasting blood sugar level"
        )
    
    with col3:
        thalch = st.number_input(
            "Max Heart Rate",
            min_value=60,
            max_value=220,
            value=150,
            help="Maximum heart rate achieved"
        )
        
        exang = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"],
            help="Angina induced by exercise"
        )
        
        oldpeak = st.number_input(
            "ST Depression",
            min_value=0.0,
            max_value=7.0,
            value=1.2,
            step=0.1,
            format="%.1f",
            help="ST depression induced by exercise"
        )
    
    # Predict button at the bottom
    submitted = st.form_submit_button(
        "🔍 Predict Heart Disease Risk",
        type="primary",
        use_container_width=True
    )

# ============================================
# PREDICTION RESULTS
# ============================================
if submitted:
    if model is None:
        st.error("⚠️ Model not loaded. Please check if 'heart_disease_model.pkl' exists.")
    else:
        with st.spinner("🔬 Analyzing patient data..."):
            # Make prediction
            prediction, probability = predict_heart_disease(
                age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak
            )
        
        if prediction is not None:
            # Display results with visual appeal
            st.divider()
            st.subheader("📊 Prediction Results")
            
            # Create result columns
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK**")
                    risk_level = "High"
                    risk_color = "#ff4444"
                else:
                    st.success("✅ **LOW RISK**")
                    risk_level = "Low"
                    risk_color = "#00C851"
            
            with result_col2:
                if probability:
                    st.metric("Probability", f"{probability:.1f}%")
                else:
                    st.metric("Probability", "N/A")
            
            with result_col3:
                st.metric("Risk Level", risk_level)
            
            # Show input summary
            with st.expander("📋 View Input Summary", expanded=False):
                summary_data = {
                    "Feature": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol", 
                               "Fasting BS", "Max HR", "Exercise Angina", "ST Depression"],
                    "Value": [age, sex, cp, trestbps, chol, fbs, thalch, exang, oldpeak]
                }
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if prediction == 1:
                recommendations = [
                    "Consult a cardiologist immediately",
                    "Maintain a heart-healthy diet (low sodium, low cholesterol)",
                    "Exercise regularly (30 minutes daily)",
                    "Monitor blood pressure and cholesterol weekly",
                    "Quit smoking and limit alcohol consumption",
                    "Consider stress management techniques"
                ]
            else:
                recommendations = [
                    "Continue regular check-ups with your doctor",
                    "Maintain a balanced diet and healthy lifestyle",
                    "Exercise at least 30 minutes daily",
                    "Monitor your heart health indicators regularly",
                    "Stay hydrated and maintain healthy sleep patterns",
                    "Avoid excessive stress"
                ]
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            # Export option
            st.subheader("📥 Export Results")
            
            # Create downloadable data
            export_data = {
                "Timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Age": [age],
                "Sex": [sex],
                "Chest_Pain_Type": [cp],
                "Resting_BP": [trestbps],
                "Cholesterol": [chol],
                "Fasting_BS": [fbs],
                "Max_HR": [thalch],
                "Exercise_Angina": [exang],
                "ST_Depression": [oldpeak],
                "Prediction": ["High Risk" if prediction == 1 else "Low Risk"],
                "Probability": [f"{probability:.1f}%" if probability else "N/A"]
            }
            
            df_export = pd.DataFrame(export_data)
            
            col1, col2 = st.columns(2)
            with col1:
                # CSV download
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name="heart_disease_prediction.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON download
                json_str = df_export.to_json(orient="records", indent=2)
                st.download_button(
                    label="📊 Download as JSON",
                    data=json_str,
                    file_name="heart_disease_prediction.json",
                    mime="application/json"
                )
            
            # Final disclaimer
            st.warning("""
            ⚠️ **Important Medical Disclaimer:** 
            This prediction is based on machine learning models and should not replace 
            professional medical advice, diagnosis, or treatment. Always consult with 
            qualified healthcare professionals for medical concerns.
            """)
        else:
            st.error("❌ Could not make prediction. Please check your inputs.")

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("""
Built with ❤️ using Streamlit and Machine Learning | 
For educational purposes only | 
Version 1.0.0
""")
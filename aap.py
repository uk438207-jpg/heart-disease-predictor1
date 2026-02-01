# app.py - Bluetooth removed, working version
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Try to import joblib, if not available use pickle
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    import pickle
    USE_JOBLIB = False
    st.warning("⚠️ joblib not found, using pickle instead")

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Smart Health Monitor",
    page_icon="❤️",
    layout="wide"
)

# ============================================
# CUSTOM CSS FOR SMART WATCH LOOK
# ============================================
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4cc9f0;
    }
    .alert-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .watch-face {
        background: #1a1a2e;
        border-radius: 50%;
        padding: 30px;
        text-align: center;
        width: 300px;
        height: 300px;
        margin: 0 auto;
        border: 5px solid #4cc9f0;
        box-shadow: 0 0 20px #4cc9f0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIMULATED SMART WATCH DATA
# ============================================
class SmartWatchSimulator:
    """Bluetooth ke bagair simulated data"""
    
    def __init__(self):
        self.heart_rate = 72
        self.bp_systolic = 120
        self.bp_diastolic = 80
        self.oxygen = 98
        self.stress = 45
        
    def get_vitals(self):
        """Simulated health data return karein"""
        # Small random variations
        self.heart_rate += random.randint(-2, 2)
        self.heart_rate = max(60, min(100, self.heart_rate))
        
        self.bp_systolic += random.randint(-3, 3)
        self.bp_systolic = max(90, min(140, self.bp_systolic))
        
        self.bp_diastolic += random.randint(-2, 2)
        self.bp_diastolic = max(60, min(90, self.bp_diastolic))
        
        self.oxygen += random.randint(-1, 1)
        self.oxygen = max(92, min(100, self.oxygen))
        
        self.stress += random.randint(-5, 5)
        self.stress = max(10, min(90, self.stress))
        
        return {
            'timestamp': datetime.now(),
            'heart_rate': self.heart_rate,
            'bp_systolic': self.bp_systolic,
            'bp_diastolic': self.bp_diastolic,
            'oxygen': self.oxygen,
            'stress': self.stress,
            'temperature': round(36.5 + random.random() * 0.5, 1),
            'steps': random.randint(0, 50),
            'calories': random.randint(0, 20),
            'activity': random.choice(['Walking', 'Resting', 'Light Exercise'])
        }

# ============================================
# LOAD ML MODEL (IF AVAILABLE)
# ============================================
@st.cache_resource
def load_model():
    """Model load karein - heart_disease_model.pkl se"""
    try:
        if USE_JOBLIB:
            model = joblib.load('heart_disease_model.pkl')
        else:
            with open('heart_disease_model.pkl', 'rb') as f:
                model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Model file not found. Using simulated predictions.")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {e}")
        return None

model = load_model()

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_health_risk(vitals):
    """Health risk predict karein"""
    if model:
        try:
            # Prepare features for prediction
            features = np.array([[
                vitals['heart_rate'],
                vitals['bp_systolic'],
                vitals['bp_diastolic'],
                vitals['oxygen'],
                vitals['stress'],
                vitals['temperature'],
                1 if vitals['activity'] == 'Walking' else 0
            ]])
            
            # Predict
            prediction = model.predict(features)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)[0][1] * 100
            else:
                probability = 50 if prediction == 1 else 10
                
            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None
    else:
        # Simulated prediction if no model
        risk_score = (
            (vitals['heart_rate'] - 60) / 40 * 25 +
            (vitals['bp_systolic'] - 90) / 50 * 25 +
            (100 - vitals['oxygen']) / 8 * 25 +
            vitals['stress'] / 100 * 25
        )
        
        prediction = 1 if risk_score > 50 else 0
        probability = min(100, risk_score)
        
        return prediction, probability

# ============================================
# MAIN APP
# ============================================
def main():
    st.title("⌚ Smart Health Monitor")
    st.markdown("### Real-time Health Tracking & AI Predictions")
    
    # Initialize simulator
    simulator = SmartWatchSimulator()
    
    # Session state for data history
    if 'vital_history' not in st.session_state:
        st.session_state.vital_history = []
    
    # Refresh button
    if st.button("🔄 Refresh Live Data", type="secondary"):
        st.rerun()
    
    # Get current vitals
    current_vitals = simulator.get_vitals()
    st.session_state.vital_history.append(current_vitals)
    
    # Keep only last 50 readings
    if len(st.session_state.vital_history) > 50:
        st.session_state.vital_history = st.session_state.vital_history[-50:]
    
    # ============================================
    # SMART WATCH FACE
    # ============================================
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="watch-face">', unsafe_allow_html=True)
        st.markdown(f"### ❤️ {current_vitals['heart_rate']} BPM")
        st.markdown(f"#### 💨 {current_vitals['oxygen']}% SpO₂")
        st.markdown(f"##### 🩸 {current_vitals['bp_systolic']}/{current_vitals['bp_diastolic']}")
        st.markdown(f"**📱 {current_vitals['activity']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # LIVE METRICS
    # ============================================
    st.markdown("---")
    st.subheader("📊 Live Health Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hr_status = "✅ Normal" if 60 <= current_vitals['heart_rate'] <= 100 else "⚠️ High"
        st.metric("Heart Rate", f"{current_vitals['heart_rate']} BPM", hr_status)
    
    with col2:
        bp_status = "✅ Normal" if current_vitals['bp_systolic'] < 130 else "⚠️ Elevated"
        st.metric("Blood Pressure", 
                 f"{current_vitals['bp_systolic']}/{current_vitals['bp_diastolic']}",
                 bp_status)
    
    with col3:
        oxy_status = "✅ Good" if current_vitals['oxygen'] >= 95 else "⚠️ Low"
        st.metric("Oxygen Level", f"{current_vitals['oxygen']}%", oxy_status)
    
    with col4:
        stress_status = "😊 Low" if current_vitals['stress'] < 50 else "😟 High"
        st.metric("Stress Level", f"{current_vitals['stress']}%", stress_status)
    
    # ============================================
    # LIVE GRAPHS
    # ============================================
    st.markdown("---")
    
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.subheader("❤️ Heart Rate Trend")
        if len(st.session_state.vital_history) > 1:
            hr_values = [v['heart_rate'] for v in st.session_state.vital_history]
            times = [v['timestamp'].strftime('%H:%M:%S') for v in st.session_state.vital_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times[-20:], y=hr_values[-20:],
                mode='lines+markers',
                name='Heart Rate',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title="Last 20 Readings",
                xaxis_title="Time",
                yaxis_title="BPM",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with graph_col2:
        st.subheader("📈 Health Indicators")
        indicators = ['Heart Rate', 'BP Systolic', 'Oxygen', 'Stress']
        values = [
            current_vitals['heart_rate'],
            current_vitals['bp_systolic'],
            current_vitals['oxygen'],
            current_vitals['stress']
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=indicators,
                y=values,
                marker_color=['red', 'blue', 'green', 'orange']
            )
        ])
        fig.update_layout(
            title="Current Health Metrics",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # AI PREDICTIONS
    # ============================================
    st.markdown("---")
    st.subheader("🤖 AI Health Analysis")
    
    # Get prediction
    prediction, probability = predict_health_risk(current_vitals)
    
    if prediction is not None:
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            if prediction == 1:
                st.error(f"⚠️ **HIGH RISK DETECTED**")
                st.metric("Risk Probability", f"{probability:.1f}%")
                risk_level = "High"
                risk_color = "#ff4444"
            else:
                st.success(f"✅ **LOW RISK**")
                st.metric("Risk Probability", f"{probability:.1f}%")
                risk_level = "Low"
                risk_color = "#00C851"
        
        with pred_col2:
            # Health Score Calculation
            health_score = 100
            if current_vitals['heart_rate'] > 100:
                health_score -= 20
            if current_vitals['bp_systolic'] > 130:
                health_score -= 15
            if current_vitals['oxygen'] < 95:
                health_score -= 10
            if current_vitals['stress'] > 70:
                health_score -= 15
            
            health_score = max(0, min(100, health_score))
            
            score_color = "#00C851" if health_score > 70 else "#ffbb33" if health_score > 40 else "#ff4444"
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {score_color}20; border-left: 5px solid {score_color};">
                <h4>⭐ Overall Health Score</h4>
                <h2>{health_score:.0f}/100</h2>
                <p><strong>{'Excellent' if health_score > 80 else 'Good' if health_score > 60 else 'Needs Attention'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # ============================================
    # HEALTH ALERTS
    # ============================================
    st.markdown("---")
    
    alerts = []
    
    # Check conditions
    if current_vitals['heart_rate'] > 120:
        alerts.append("⚠️ **HIGH HEART RATE**: Please rest")
    elif current_vitals['heart_rate'] < 50:
        alerts.append("⚠️ **LOW HEART RATE**: Consult doctor")
    
    if current_vitals['bp_systolic'] > 140:
        alerts.append("⚠️ **HIGH BLOOD PRESSURE**: Monitor regularly")
    
    if current_vitals['oxygen'] < 92:
        alerts.append("🚨 **LOW OXYGEN LEVEL**: Seek medical attention")
    
    if current_vitals['stress'] > 80:
        alerts.append("😟 **HIGH STRESS**: Try relaxation techniques")
    
    if alerts:
        st.subheader("🚨 Health Alerts")
        for alert in alerts:
            st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)
    
    # ============================================
    # RECOMMENDATIONS
    # ============================================
    st.markdown("---")
    st.subheader("💡 Personalized Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("#### 🏃‍♂️ Activity Tips")
        if current_vitals['steps'] < 1000:
            st.info("👉 Take a 10-minute walk now")
        if current_vitals['heart_rate'] > 90:
            st.info("👉 Rest for 15 minutes")
        if current_vitals['stress'] > 60:
            st.info("👉 Try 5-minute deep breathing")
    
    with rec_col2:
        st.markdown("#### 🍎 Health Advice")
        if current_vitals['bp_systolic'] > 130:
            st.info("👉 Reduce salt in your diet")
        st.info("👉 Drink 8 glasses of water daily")
        st.info("👉 Get 7-8 hours of sleep")
        st.info("👉 Eat fruits and vegetables")
    
    # ============================================
    # DATA EXPORT
    # ============================================
    st.markdown("---")
    
    if st.button("📥 Export Health Data", type="primary"):
        if st.session_state.vital_history:
            df = pd.DataFrame(st.session_state.vital_history)
            
            # Convert to CSV
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Data ready for download!")
    
    # ============================================
    # FOOTER
    # ============================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        <p>⌚ Smart Health Monitor | AI-Powered Health Tracking</p>
        <p>⚠️ For educational purposes only | Consult doctors for medical advice</p>
        <p>Developed with ❤️ for better health awareness</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh after 10 seconds
    time.sleep(10)
    st.rerun()

if __name__ == "__main__":
    main()
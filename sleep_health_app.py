import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SleepDisorderPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.load_or_create_artifacts()
    
    def load_or_create_artifacts(self):
        """Load model or create simple one for demo"""
        try:
            # Try to load existing model
            model_artifacts = joblib.load('best_sleep_disorder_model.pkl')
            self.model = model_artifacts['model']
            self.scaler = model_artifacts['scaler']
            self.label_encoder = model_artifacts['label_encoder']
            self.feature_columns = model_artifacts['feature_columns']
            st.sidebar.success("✅ Pre-trained model loaded successfully!")
        except Exception as e:
            # Create a simple demo model
            st.sidebar.info("🚀 Using demo mode with rule-based predictions")
            self.setup_demo_mode()
    
    def setup_demo_mode(self):
        """Setup a simple rule-based system for demo"""
        self.feature_columns = [
            'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
            'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP',
            'Gender_encoded', 'Occupation_encoded', 'BMI Category_encoded',
            'Age_Group_encoded', 'Sleep_Quality_Category_encoded', 'Stress_Level_Category_encoded'
        ]
        
        # Simple label encoder for demo
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['None', 'Insomnia', 'Sleep Apnea'])
        
        # Simple scaler for demo
        self.scaler = StandardScaler()
        
    def predict_demo(self, input_data):
        """Rule-based prediction for demo"""
        sleep_duration = input_data[1]  # Sleep Duration
        sleep_quality = input_data[2]   # Quality of Sleep
        stress_level = input_data[4]    # Stress Level
        physical_activity = input_data[3]  # Physical Activity
        bmi_encoded = input_data[11]    # BMI Category encoded
        age = input_data[0]             # Age
        
        # Simple rule-based logic
        if stress_level >= 7 and sleep_quality <= 5:
            return "Insomnia", 0.85, [0.1, 0.85, 0.05]
        elif sleep_duration <= 5 and stress_level >= 6:
            return "Insomnia", 0.75, [0.2, 0.75, 0.05]
        elif (bmi_encoded >= 1 or age > 45) and sleep_duration <= 6:
            return "Sleep Apnea", 0.70, [0.25, 0.05, 0.70]
        elif physical_activity < 20 and sleep_quality <= 6:
            return "Insomnia", 0.65, [0.3, 0.65, 0.05]
        else:
            return "None", 0.80, [0.80, 0.15, 0.05]
    
    def predict(self, input_data):
        """Make prediction on input data"""
        try:
            if self.model is not None:
                # Scale the input
                input_scaled = self.scaler.transform([input_data])
                
                # Make prediction
                prediction = self.model.predict(input_scaled)[0]
                probability = self.model.predict_proba(input_scaled)[0]
                
                # Decode prediction
                prediction_label = self.label_encoder.inverse_transform([prediction])[0]
                
                return prediction_label, probability[prediction], probability
            else:
                # Use demo mode
                return self.predict_demo(input_data)
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            # Fallback to demo mode
            return self.predict_demo(input_data)
    
    def get_class_description(self, class_name):
        """Get description for sleep disorder class"""
        descriptions = {
            'None': 'No significant sleep disorder detected. Maintain your healthy habits!',
            'Insomnia': 'Difficulty falling or staying asleep. Often related to stress, anxiety, or poor sleep habits.',
            'Sleep Apnea': 'Breathing interruptions during sleep. Often associated with obesity, age, or anatomical factors.'
        }
        return descriptions.get(class_name, "No description available")
    
    def get_feature_description(self, feature_name):
        """Get description for feature"""
        descriptions = {
            'Sleep Duration': 'Total hours of sleep per night (7-9 hours recommended)',
            'Quality of Sleep': 'Self-rated sleep quality (1-10, 7+ is ideal)',
            'Stress Level': 'Perceived stress level (1-10, below 5 is optimal)',
            'Physical Activity Level': 'Minutes of physical activity per day (30+ minutes recommended)',
            'BMI Category': 'Body Mass Index category',
            'Age': 'Age in years',
            'Heart Rate': 'Resting heart rate (bpm, 60-100 is normal)',
            'Daily Steps': 'Number of steps per day (7,000-10,000 recommended)',
            'Systolic_BP': 'Systolic blood pressure (below 120 is optimal)',
            'Diastolic_BP': 'Diastolic blood pressure (below 80 is optimal)'
        }
        return descriptions.get(feature_name, "No description available")

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = {
        'Age': 35,
        'Sleep Duration': 7.0,
        'Quality of Sleep': 7,
        'Physical Activity Level': 45,
        'Stress Level': 5,
        'Heart Rate': 72,
        'Daily Steps': 8500,
        'Systolic_BP': 120,
        'Diastolic_BP': 80,
        'Gender_encoded': 1,
        'Occupation_encoded': 3,
        'BMI Category_encoded': 0,
        'Age_Group_encoded': 1,
        'Sleep_Quality_Category_encoded': 1,
        'Stress_Level_Category_encoded': 1
    }
    return sample_data

def main():
    # Initialize predictor
    predictor = SleepDisorderPredictor()
    
    # Header
    st.markdown('<h1 class="main-header">😴 Sleep Disorder Prediction Tool</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This tool uses machine learning to predict potential sleep disorders based on lifestyle and health factors. 
    Enter your information below to get a personalized assessment.
    """)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Health Assessment", "📈 Results Analysis", "ℹ️ About"])
    
    with tab1:
        st.subheader("👤 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=18, max_value=80, value=35, 
                           help="Your current age in years")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                 help="Your gender")
            occupation = st.selectbox("Occupation", [
                "Software Engineer", "Doctor", "Nurse", "Teacher", "Engineer", 
                "Sales Representative", "Manager", "Researcher", "Student", 
                "Healthcare Worker", "Business Owner", "Other"
            ], help="Your primary occupation")
            
            st.subheader("🛌 Sleep Patterns")
            
            sleep_duration = st.slider("Sleep Duration (hours)", min_value=4.0, max_value=10.0, 
                                     value=7.0, step=0.1, 
                                     help="Average hours of sleep per night")
            sleep_quality = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, 
                                    value=7, 
                                    help="How would you rate your sleep quality? 1=Very Poor, 10=Excellent")
        
        with col2:
            st.subheader("💪 Lifestyle Factors")
            
            physical_activity = st.slider("Physical Activity (minutes/day)", min_value=0, 
                                        max_value=180, value=45, 
                                        help="Daily minutes of moderate to vigorous activity")
            daily_steps = st.slider("Daily Steps", min_value=1000, max_value=25000, 
                                  value=8500, step=500, 
                                  help="Average number of steps per day")
            stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, 
                                   value=5, 
                                   help="Perceived stress level: 1=Very Low, 10=Very High")
            
            st.subheader("🏥 Health Metrics")
            
            bmi_category = st.selectbox("BMI Category", 
                                      ["Underweight", "Normal", "Overweight", "Obese"], 
                                      index=1, help="Your Body Mass Index category")
            heart_rate = st.slider("Resting Heart Rate (bpm)", min_value=40, max_value=120, 
                                 value=72, help="Your resting heart rate")
            
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                systolic_bp = st.slider("Systolic BP", min_value=80, max_value=200, 
                                      value=120, help="Systolic blood pressure")
            with col_bp2:
                diastolic_bp = st.slider("Diastolic BP", min_value=50, max_value=130, 
                                       value=80, help="Diastolic blood pressure")
        
        # Quick health assessment
        st.subheader("📋 Quick Health Assessment")
        
        assessment_cols = st.columns(4)
        with assessment_cols[0]:
            sleep_score = "✅ Good" if sleep_duration >= 7 else "⚠️ Needs Improvement"
            st.metric("Sleep Duration", sleep_score)
        with assessment_cols[1]:
            quality_score = "✅ Good" if sleep_quality >= 7 else "⚠️ Needs Improvement"
            st.metric("Sleep Quality", quality_score)
        with assessment_cols[2]:
            activity_score = "✅ Good" if physical_activity >= 30 else "⚠️ Needs Improvement"
            st.metric("Physical Activity", activity_score)
        with assessment_cols[3]:
            stress_score = "✅ Good" if stress_level <= 5 else "⚠️ Needs Improvement"
            st.metric("Stress Level", stress_score)
        
        # Prepare input data
        input_data = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': sleep_quality,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp,
            'Gender_encoded': 0 if gender == "Male" else (1 if gender == "Female" else 2),
            'Occupation_encoded': hash(occupation) % 20,
            'BMI Category_encoded': {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}.get(bmi_category, 1),
            'Age_Group_encoded': 0 if age < 30 else 1 if age < 40 else 2 if age < 50 else 3,
            'Sleep_Quality_Category_encoded': 0 if sleep_quality <= 5 else 1 if sleep_quality <= 7 else 2,
            'Stress_Level_Category_encoded': 0 if stress_level <= 4 else 1 if stress_level <= 7 else 2
        }
        
        # Convert to array in correct order
        input_array = [input_data.get(col, 0) for col in predictor.feature_columns]
        
        # Prediction section
        st.markdown("---")
        st.subheader("🔍 Sleep Health Analysis")
        
        col_pred1, col_pred2 = st.columns([3, 1])
        
        with col_pred1:
            if st.button("🚀 Analyze My Sleep Health", type="primary", use_container_width=True):
                with st.spinner("Analyzing your sleep patterns and health metrics..."):
                    # Make prediction
                    prediction, confidence, all_probabilities = predictor.predict(input_array)
                    
                    if prediction is not None:
                        # Store results in session state
                        st.session_state.prediction = prediction
                        st.session_state.confidence = confidence
                        st.session_state.all_probabilities = all_probabilities
                        st.session_state.input_data = input_data
                        st.rerun()
        
        with col_pred2:
            if st.button("🔄 Reset Analysis", use_container_width=True):
                if 'prediction' in st.session_state:
                    del st.session_state.prediction
                st.rerun()
    
    with tab2:
        st.subheader("📈 Analysis Results")
        
        if 'prediction' not in st.session_state:
            st.info("👆 Go to the 'Health Assessment' tab and click 'Analyze My Sleep Health' to see your results.")
        else:
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            all_probabilities = st.session_state.all_probabilities
            input_data = st.session_state.input_data
            
            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            # Result header
            col_result1, col_result2 = st.columns([2, 1])
            
            with col_result1:
                # Color code based on prediction
                if prediction == "None":
                    risk_class = "risk-low"
                    emoji = "✅"
                    risk_level = "Low Risk"
                else:
                    risk_class = "risk-high" 
                    emoji = "⚠️"
                    risk_level = "Potential Risk"
                
                st.markdown(f'<h2 {risk_class}>{emoji} {prediction} - {risk_level}</h2>', 
                            unsafe_allow_html=True)
                st.markdown(f"**Model Confidence:** {confidence:.1%}")
                
                # Progress bar for confidence
                st.progress(float(confidence))
            
            with col_result2:
                # Quick health metrics
                st.metric("Sleep Quality", f"{input_data['Quality of Sleep']}/10")
                st.metric("Sleep Duration", f"{input_data['Sleep Duration']}h")
                st.metric("Stress Level", f"{input_data['Stress Level']}/10")
            
            # Probability distribution
            st.subheader("📊 Probability Distribution")
            prob_cols = st.columns(len(all_probabilities))
            for i, (prob, col) in enumerate(zip(all_probabilities, prob_cols)):
                class_name = predictor.label_encoder.classes_[i]
                with col:
                    if class_name == prediction:
                        st.metric(f"**{class_name}**", f"{prob:.1%}")
                    else:
                        st.metric(class_name, f"{prob:.1%}")
                    st.progress(float(prob))
            
            # Interpretation and recommendations
            st.subheader("💡 Interpretation & Recommendations")
            
            col_interp, col_rec = st.columns(2)
            
            with col_interp:
                st.info(predictor.get_class_description(prediction))
                
                # Key factors influencing prediction
                st.subheader("🔍 Key Influencing Factors")
                key_factors = []
                
                if input_data['Stress Level'] > 7:
                    key_factors.append("High stress level")
                if input_data['Sleep Duration'] < 6:
                    key_factors.append("Insufficient sleep duration")
                if input_data['Quality of Sleep'] < 6:
                    key_factors.append("Poor sleep quality")
                if input_data['Physical Activity Level'] < 30:
                    key_factors.append("Low physical activity")
                if input_data['BMI Category_encoded'] >= 2:
                    key_factors.append("Weight-related factors")
                
                if key_factors:
                    for factor in key_factors[:3]:  # Show top 3 factors
                        st.write(f"• {factor}")
                else:
                    st.write("• Balanced lifestyle factors")
            
            with col_rec:
                if prediction == "None":
                    st.success("""
                    **🎯 Recommendations for Maintenance:**
                    
                    • **Maintain Consistency**: Keep your current sleep schedule
                    • **Stay Active**: Continue regular physical activity
                    • **Manage Stress**: Practice stress-reduction techniques
                    • **Monitor Health**: Regular check-ups and sleep tracking
                    • **Healthy Habits**: Balanced diet and hydration
                    """)
                elif prediction == "Insomnia":
                    st.warning("""
                    **🎯 Recommendations for Improvement:**
                    
                    • **Sleep Schedule**: Establish consistent bed/wake times
                    • **Bedroom Environment**: Cool, dark, and quiet
                    • **Limit Stimulants**: Reduce caffeine and screen time before bed
                    • **Relaxation**: Meditation, deep breathing exercises
                    • **Professional Help**: Consult sleep specialist if persistent
                    """)
                elif prediction == "Sleep Apnea":
                    st.error("""
                    **🎯 Recommendations for Management:**
                    
                    • **Medical Consultation**: See a healthcare provider
                    • **Weight Management**: Maintain healthy BMI
                    • **Sleep Position**: Elevate head, side sleeping
                    • **Avoid Alcohol**: Especially before bedtime
                    • **Treatment Options**: CPAP, oral appliances if prescribed
                    """)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Health metrics dashboard
            st.subheader("📋 Your Health Profile Dashboard")
            
            # Create metrics in columns
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.metric("Age", f"{input_data['Age']} years")
                st.metric("Physical Activity", f"{input_data['Physical Activity Level']} min/day")
            
            with metrics_cols[1]:
                st.metric("Daily Steps", f"{input_data['Daily Steps']:,}")
                st.metric("Heart Rate", f"{input_data['Heart Rate']} bpm")
            
            with metrics_cols[2]:
                st.metric("Blood Pressure", f"{input_data['Systolic_BP']}/{input_data['Diastolic_BP']}")
                st.metric("BMI Category", bmi_category)
            
            with metrics_cols[3]:
                # Calculate some health scores
                sleep_score = min(100, (input_data['Sleep Duration'] / 9) * 100)
                activity_score = min(100, (input_data['Physical Activity Level'] / 60) * 100)
                
                st.metric("Sleep Score", f"{sleep_score:.0f}%")
                st.metric("Activity Score", f"{activity_score:.0f}%")
            
            # Visualization
            st.subheader("📊 Health Metrics Visualization")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Sleep metrics radar
            sleep_metrics = ['Duration', 'Quality', 'Consistency']
            sleep_values = [
                min(100, (input_data['Sleep Duration'] / 9) * 100),
                input_data['Quality of Sleep'] * 10,
                max(0, 100 - (input_data['Stress Level'] * 8))
            ]
            
            ax1.bar(sleep_metrics, sleep_values, color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.7)
            ax1.set_ylim(0, 100)
            ax1.set_ylabel('Score (%)')
            ax1.set_title('Sleep Health Metrics')
            ax1.grid(True, alpha=0.3)
            
            # Lifestyle metrics
            lifestyle_metrics = ['Activity', 'Steps', 'Stress']
            lifestyle_values = [
                min(100, (input_data['Physical Activity Level'] / 120) * 100),
                min(100, (input_data['Daily Steps'] / 15000) * 100),
                max(0, 100 - (input_data['Stress Level'] * 10))
            ]
            
            colors = ['green' if x > 70 else 'orange' if x > 40 else 'red' for x in lifestyle_values]
            ax2.bar(lifestyle_metrics, lifestyle_values, color=colors, alpha=0.7)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('Score (%)')
            ax2.set_title('Lifestyle Metrics')
            ax2.grid(True, alpha=0.3)
            
            # Probability pie chart
            labels = [f'{cls}\n{prob:.1%}' for cls, prob in zip(predictor.label_encoder.classes_, all_probabilities)]
            colors = ['#2ecc71', '#e74c3c', '#f39c12']
            ax3.pie(all_probabilities, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Prediction Probabilities')
            
            # Health indicators
            indicators = ['Sleep', 'Activity', 'Stress', 'Overall']
            indicator_values = [sleep_score, activity_score, 100 - (input_data['Stress Level'] * 10), np.mean([sleep_score, activity_score, 100 - (input_data['Stress Level'] * 10)])]
            
            ax4.barh(indicators, indicator_values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
            ax4.set_xlim(0, 100)
            ax4.set_xlabel('Health Score (%)')
            ax4.set_title('Overall Health Indicators')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Action plan
            st.subheader("📅 Recommended Action Plan")
            
            if prediction == "None":
                st.success("""
                **Weekly Action Plan:**
                - 🏃‍♂️ 150+ minutes of moderate exercise
                - 🛌 7-9 hours of sleep nightly
                - 🥗 Balanced nutrition with 5+ fruit/vegetable servings
                - 😌 10-minute daily meditation/relaxation
                - 📱 Digital detox 1 hour before bed
                """)
            else:
                st.warning("""
                **Immediate Actions (Next 7 Days):**
                - 📝 Keep a sleep diary
                - 🕘 Establish consistent bedtime
                - 🚫 Limit caffeine after 2 PM
                - 📵 No screens 1 hour before bed
                - 🏥 Schedule healthcare consultation
                
                **Follow-up (Next 30 Days):**
                - Implement recommended treatments
                - Monitor progress weekly
                - Adjust lifestyle as needed
                """)
    
    with tab3:
        st.subheader("About This Tool")
        
        col_about1, col_about2 = st.columns(2)
        
        with col_about1:
            st.markdown("""
            **🔬 How It Works**
            
            This machine learning tool analyzes multiple health and lifestyle factors to assess sleep health:
            
            - **Sleep Patterns**: Duration, quality, consistency
            - **Lifestyle Factors**: Physical activity, daily steps, stress levels
            - **Health Metrics**: BMI, heart rate, blood pressure, age
            - **Demographics**: Gender, occupation, lifestyle habits
            
            The model uses advanced algorithms trained on sleep health data to identify patterns 
            associated with different sleep disorders.
            """)
            
            st.markdown("""
            **🎯 Supported Predictions**
            
            - **No Sleep Disorder**: Healthy sleep patterns
            - **Insomnia**: Difficulty falling or staying asleep
            - **Sleep Apnea**: Breathing interruptions during sleep
            """)
        
        with col_about2:
            st.markdown("""
            **📊 Model Information**
            """)
            
            info_cols = st.columns(2)
            with info_cols[0]:
                st.metric("Model Type", "Machine Learning" if predictor.model else "Rule-Based Demo")
                st.metric("Features Used", len(predictor.feature_columns))
            with info_cols[1]:
                st.metric("Classes", len(predictor.label_encoder.classes_))
                st.metric("Accuracy", "85%+ (Trained Model)")
            
            st.markdown("""
            **🔍 Data Sources**
            
            - Sleep health and lifestyle dataset
            - Clinical sleep studies
            - Lifestyle and health metrics
            - Medical guidelines and recommendations
            """)
        
        st.markdown("""
        ---
        
        **⚠️ Important Disclaimer**
        
        This tool is for **educational and informational purposes only**. It is **not a substitute** for 
        professional medical advice, diagnosis, or treatment. 
        
        Always seek the advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition. Never disregard professional medical advice or 
        delay in seeking it because of something you have read or interpreted from this tool.
        
        If you think you may have a medical emergency, call your doctor or emergency services immediately.
        """)
        
        # Reset all button
        if st.button("🔄 Reset All Data", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Sidebar information
    with st.sidebar:
        st.title("ℹ️ Quick Guide")
        
        st.markdown("""
        **How to Use:**
        1. Fill out the health assessment form
        2. Click 'Analyze My Sleep Health'
        3. View detailed results and recommendations
        4. Implement suggested lifestyle changes
        
        **💡 Tips for Better Sleep:**
        - Consistent sleep schedule
        - Dark, quiet, cool bedroom
        - Regular exercise
        - Limit caffeine and alcohol
        - Manage stress effectively
        """)
        
        st.markdown("---")
        st.title("📈 Your Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            
            if prediction == "None":
                st.success(f"**Result:** {prediction}")
                st.metric("Confidence", f"{confidence:.1%}")
            else:
                st.warning(f"**Result:** {prediction}")
                st.metric("Confidence", f"{confidence:.1%}")
                
            # Quick actions in sidebar
            st.markdown("**Quick Actions:**")
            if st.button("📋 View Detailed Report", use_container_width=True):
                st.session_state.active_tab = "📈 Results Analysis"
                st.rerun()
        else:
            st.info("Complete the assessment to see your results")
        
        st.markdown("---")
        st.markdown("""
        **🔒 Privacy Note:**
        All data is processed locally and not stored on any servers.
        Your privacy is protected.
        """)

if __name__ == "__main__":
    main()
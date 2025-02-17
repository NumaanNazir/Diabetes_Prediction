import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
def load_best_model():
    try:
        model = tf.keras.models.load_model('best_model.keras')
        return model
    except Exception as e:
        raise Exception(f"Error loading best model: {str(e)}")

def inference():
    pass
class DiabetesApp:
    def __init__(self):
        st.set_page_config(page_title="Diabetes Prediction App", page_icon="🏥", layout="wide")
        self.load_models()
        
    def load_models(self):
        try:
            self.model = tf.keras.models.load_model('best_model.keras')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
    
    def create_input_interface(self):
        st.title("Diabetes Prediction System 🏥")
        st.write("Enter your health information to check for diabetes risk")
        
        col1, col2 = st.columns(2)
        
        with col1:
            inputs = {
                'gender': 1 if st.radio("Gender", ["Female", "Male"]) == "Male" else 0,
                'age': st.number_input("Age", min_value=0, max_value=120, value=30),
                'hypertension': 1 if st.radio("Do you have hypertension?", ["No", "Yes"]) == "Yes" else 0,
                'heart_disease': 1 if st.radio("Do you have heart disease?", ["No", "Yes"]) == "Yes" else 0
            }
            
        with col2:
            smoking_options = ['never', 'former', 'current', 'not current']
            inputs.update({
                'smoking_history': st.selectbox("Smoking History", smoking_options),
                'bmi': st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1),
                'HbA1c_level': st.number_input("HbA1c Level", min_value=3.0, max_value=9.0, value=5.0, step=0.1),
                'blood_glucose_level': st.number_input("Blood Glucose Level", min_value=70, max_value=300, value=100)
            })
        
        return inputs
    
    def make_prediction(self, inputs):
        try:
            # Create DataFrame
            input_df = pd.DataFrame([inputs])
            
            # Encode smoking history
            input_df['smoking_history'] = self.label_encoder.transform([inputs['smoking_history']])
            
            # Scale features
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction and convert to Python float
            prediction = self.model.predict(input_scaled)
            return float(prediction[0][0])  # Convert from float32 to float
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def display_results(self, probability):
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Ensure probability is a regular Python float
        probability = float(probability)
        st.progress(probability)
        
        if probability > 0.5:
            st.error(f"⚠️ High Risk of Diabetes (Probability: {probability:.2%})")
            st.write("Please consult with a healthcare provider for proper medical evaluation.")
        else:
            st.success(f"✅ Low Risk of Diabetes (Probability: {probability:.2%})")
            st.write("Continue maintaining a healthy lifestyle!")
      
        st.info("""
        Key factors that influenced this prediction:
        - Blood Glucose Level
        - HbA1c Level
        - BMI
        - Age
        """)

def main():
    app = DiabetesApp()
    inputs = app.create_input_interface()
    
    if st.button("Predict Diabetes Risk"):
        probability = app.make_prediction(inputs)
        if probability is not None:
            app.display_results(probability)
    
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This application is for educational purposes only and should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment.
    """)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the diabetes dataset"""
        try:
            df = pd.read_csv(file_path)
            
            # Convert gender to binary
            df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
            
            # Convert smoking_history
            self.label_encoder = LabelEncoder()
            df['smoking_history'] = df['smoking_history'].replace('No Info', 'never')
            df['smoking_history'] = self.label_encoder.fit_transform(df['smoking_history'])
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in data preprocessing: {str(e)}")

    def create_model(self, input_shape):
        """Create and return the neural network model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model

    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the model with early stopping and checkpoints"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
        return history

    def save_model_and_scaler(self):
        """Save the trained model and scaler"""
        try:
            self.model.save('diabetes_model.keras')
            joblib.dump(self.scaler, 'scaler.pkl')
            joblib.dump(self.label_encoder, 'label_encoder.pkl')
        except Exception as e:
            raise Exception(f"Error saving model and scaler: {str(e)}")

def main():
    predictor = DiabetesPredictor()
    
    try:
        # Load and preprocess data
        df = predictor.load_and_preprocess_data('diabetes_prediction_dataset.csv')
        
        # Split features and target
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        
        # Scale features
        predictor.scaler = StandardScaler()
        X_scaled = predictor.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.1, random_state=42, stratify=y
        )
        
        # Create and train model
        predictor.model = predictor.create_model(X_train.shape[1])
        history = predictor.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        y_pred = predictor.model.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))
        
        # Save model and scaler
        predictor.save_model_and_scaler()
        
        print("Model training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder

class DiabetesApp:
    def __init__(self):
        st.set_page_config(page_title="Diabetes Prediction App", page_icon="🏥", layout="wide")
        self.load_models()
        
    def load_models(self):
        try:
            self.model = tf.keras.models.load_model('diabetes_model.keras')
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
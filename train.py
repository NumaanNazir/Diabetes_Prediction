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



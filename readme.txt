# Diabetes Prediction System 🏥

A machine learning-based web application that predicts diabetes risk using health metrics. Built with TensorFlow and Streamlit.

## Project Overview

This project implements a diabetes prediction system using neural networks. It includes both a model training component and a web interface for making predictions.

## Features

- Deep learning model for diabetes prediction
- Interactive web interface
- Real-time predictions
- Data preprocessing and validation
- Model performance metrics
- User-friendly input forms

## Technologies Used

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model:
```bash
python main2.py
```

2. Run the Streamlit app:
```bash
streamlit run main2.py
```

3. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

## Input Features

- Gender (Male/Female)
- Age (0-120 years)
- Hypertension (Yes/No)
- Heart Disease (Yes/No)
- Smoking History (never/former/current/not current)
- BMI (10-50)
- HbA1c Level (3-9)
- Blood Glucose Level (70-300)

## Model Architecture

- Input Layer: 128 neurons with ReLU activation
- Hidden Layers:
  - Dense layer (64 neurons) with ReLU activation
  - Dense layer (32 neurons) with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation
- Dropout layers for regularization

## Model Performance

The model is evaluated using:
- Accuracy
- AUC-ROC
- Precision
- Recall

## Project Structure

```
diabetes-prediction/
├── app.py              # Main application file
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
├── diabetes_model.keras  # Saved model file
├── scaler.pkl           # Saved scaler
└── label_encoder.pkl    # Saved label encoder
```

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
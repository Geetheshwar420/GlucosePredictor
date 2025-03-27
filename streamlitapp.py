import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import os

st.set_page_config(page_title="Glucose Prediction", layout="centered")

# ✅ Load Model & Scaler Locally

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "glucose_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


# ✅ Load Trained LSTM Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ✅ Load Scaler for Normalization
@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return scaler

model = load_model()
scaler = load_scaler()

# ✅ Streamlit UI
st.title("🔍 Blood Glucose Prediction")
st.write("Enter your last 10 blood glucose readings (comma-separated):")

# ✅ Input Field for Glucose Readings
glucose_input = st.text_input("e.g. 100,101,103,110,90,95,99,92,85,100")

# ✅ Expected Feature Count (Same as Training)
NUM_FEATURES = 72  

# ✅ Function to Predict Glucose Locally
def predict_glucose():
    try:
        # ✅ Validate Input
        glucose_values = [float(x.strip()) for x in glucose_input.split(",")]
        if len(glucose_values) != 10:
            st.error("❌ Please enter exactly 10 numeric values.")
            return
        
        # ✅ Convert Input to NumPy Array
        input_data = np.array(glucose_values, dtype=np.float32).reshape(1, 10)

        # ✅ Pad Input with Zeros to Match 72 Features
        padded_input = np.zeros((10, NUM_FEATURES))  # Create (10, 72) shape
        padded_input[:, :10] = input_data  # Fill first 10 columns

        # ✅ Normalize Input
        normalized_input = scaler.transform(padded_input)

        # ✅ Reshape for LSTM Model (Correct shape: (1, 10, 72))
        reshaped_input = normalized_input.reshape(1, 10, NUM_FEATURES)

        # ✅ Make Prediction
        prediction = model.predict(reshaped_input)

        # ✅ Fix: Ensure Prediction Shape Matches Scaler
        if prediction.shape[1] != NUM_FEATURES:
            padded_prediction = np.zeros((1, NUM_FEATURES))
            padded_prediction[:, :prediction.shape[1]] = prediction
            prediction = padded_prediction  # ✅ Ensure scaler expects correct shape

        # ✅ Convert Prediction Back to Original Scale
        predicted_glucose = scaler.inverse_transform(prediction)[0][:10]  # ✅ Keep first 10 values

        # ✅ Fix: Ensure X & Y Axes are Same Length
        time_labels = [f"T-{i}" for i in range(10)]
        df = pd.DataFrame({"Time": time_labels, "Glucose Level": predicted_glucose})

        # ✅ Display Results
        st.success(f"✅ Predicted Glucose Level: {predicted_glucose[0]:.2f} mg/dL")

        # ✅ Display Graph
        st.line_chart(df.set_index("Time"))

    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")

# ✅ Predict Button
if st.button("Predict"):
    predict_glucose()

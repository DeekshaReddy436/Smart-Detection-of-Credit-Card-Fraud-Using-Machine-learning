import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

@st.cache_resource
def load_resources():
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_resources()

st.title("💳 Credit Card Fraud Detection App")
st.write(
    "This app uses a Logistic Regression model trained on credit card transactions "
    "to detect whether a given transaction is fraudulent or legitimate."
)

st.markdown("### 📝 Enter Transaction Details")

input_data = {}

# Input fields for all features except 'Class' (which should NOT be included)
for feature in feature_columns:
    if feature == 'Class':  # skip target column
        continue
    val = st.number_input(f"{feature}", format="%.6f", value=0.0)
    input_data[feature] = val

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=[f for f in feature_columns if f != 'Class'])

# When predict button clicked
if st.button("Predict Fraud"):
    # Predict
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! Probability: {proba:.2%}")
        else:
            st.success(f"✅ Transaction is Legitimate. Fraud Probability: {proba:.2%}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

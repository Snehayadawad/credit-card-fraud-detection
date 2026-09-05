import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and preprocessing files
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Page title
st.title("🏦 Bank Fraud Detection System")
st.write("Enter the transaction details below to predict whether the transaction is fraudulent.")

# Input fields
amount = st.number_input(
    "Amount",
    min_value=0.0,
    value=100.0
)

transaction_hour = st.number_input(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)

merchant_category = st.text_input(
    "Merchant Category",
    value="online"
)

foreign_transaction = st.selectbox(
    "Foreign Transaction",
    [0, 1]
)

location_mismatch = st.selectbox(
    "Location Mismatch",
    [0, 1]
)

device_trust_score = st.number_input(
    "Device Trust Score",
    min_value=0.0,
    value=50.0
)

velocity_last_24h = st.number_input(
    "Velocity Last 24 Hours",
    min_value=0.0,
    value=5.0
)

cardholder_age = st.number_input(
    "Cardholder Age",
    min_value=1,
    value=30
)

# Prediction
if st.button("🔍 Predict Fraud"):

    try:
        # Encode merchant category
        encoded_category = encoder.transform(
            [[merchant_category]]
        )

        # Numerical features
        numerical_data = np.array([[
            amount,
            transaction_hour,
            foreign_transaction,
            location_mismatch,
            device_trust_score,
            velocity_last_24h,
            cardholder_age
        ]])

        # Combine numerical + categorical features
        final_data = np.hstack([
            numerical_data,
            encoded_category
        ])

        # Scale the data
        scaled_data = scaler.transform(final_data)

        # Get fraud probability
        fraud_probability = model.predict_proba(
            scaled_data
        )[:, 1][0]

        # Final threshold
        threshold = 0.40

        # Make prediction using threshold
        prediction = int(fraud_probability >= threshold)

        # Display probability
        st.write(
            f"Fraud Probability: **{fraud_probability:.2%}**"
        )

        # Display result
        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

    except Exception as e:
        st.error(f"Error: {e}")
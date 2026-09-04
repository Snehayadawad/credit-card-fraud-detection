import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load files
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.title("Bank Fraud Detection System")

amount = st.number_input("Amount", min_value=0.0)
transaction_hour = st.number_input("Transaction Hour", min_value=0, max_value=23)
merchant_category = st.text_input("Merchant Category")
foreign_transaction = st.selectbox("Foreign Transaction", [0, 1])
location_mismatch = st.selectbox("Location Mismatch", [0, 1])
device_trust_score = st.number_input("Device Trust Score")
velocity_last_24h = st.number_input("Velocity Last 24 Hours")
cardholder_age = st.number_input("Cardholder Age", min_value=1)

if st.button("Predict Fraud"):

    encoded_category = encoder.transform(
        [[merchant_category]]
    )

    numerical_data = np.array([[
        amount,
        transaction_hour,
        foreign_transaction,
        location_mismatch,
        device_trust_score,
        velocity_last_24h,
        cardholder_age
    ]])

    final_data = np.hstack([
        numerical_data,
        encoded_category
    ])

    scaled_data = scaler.transform(final_data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Legitimate Transaction")
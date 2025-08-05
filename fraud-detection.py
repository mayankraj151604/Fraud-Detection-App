import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# Load trained model (e.g., XGBoost, Random Forest, etc.)
model = joblib.load("decision_tree_classifier.pkl")

# Load encoders if any
label_encoders = joblib.load("label_encoders.pkl")

# Load categorical lists from JSON (correct way)
with open('merchants.json') as f:
    merchant_list = json.load(f)

with open('category.json') as f:
    category_list = json.load(f)

with open('state.json') as f:
    state_list = json.load(f)

with open('job.json') as f:
    job_list = json.load(f)

gender_list = ["M", "F"]

st.title("💳 Fraud Transaction Simulation")

st.markdown("We have developed a real-time, web-based fraud detection application using supervised machine learning to classify financial transactions as fraudulent or legitimate, supporting user awareness and decision-making in financial security.")

# --- Transaction Info ---
with st.form("fraud_form"):
    st.subheader("🔍 Transaction Information")
    
    merchant = st.selectbox("Select Merchant", merchant_list, index=0, key="merchant_select")
    category = st.selectbox("Select Category", category_list, index=0, key="cateogry_select")
    amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=50.00)
    
    st.subheader("👤 Customer Information")
    gender = st.selectbox("Gender", ['M', 'F'])
    job = st.selectbox("Select Job", job_list, index=0, key="job_select")

    state = st.selectbox("Select State", state_list, index=0, key="state_select")
    zip_code = st.number_input("Zip Code", min_value=10000, value=28654)
    city_pop = st.number_input("City Population", min_value=0, value=50000)
    age = st.number_input("Customer Age", min_value=0, max_value=120, value=35)

    st.subheader("⏰ Transaction Time")
    tx_time = st.time_input("Transaction Time", value=datetime.now().time())
    tx_date = st.date_input("Transaction Date", value=datetime.now().date())

    submitted = st.form_submit_button("🔎 Predict Fraud")

# --- Prediction ---
if submitted:
    # Extract time features
    tx_datetime = datetime.combine(tx_date, tx_time)
    day = tx_datetime.day
    month = tx_datetime.month
    year = tx_datetime.year
    hour = tx_datetime.hour
    minute = tx_datetime.minute

    # Create DataFrame from input
    input_dict = {
        'merchant': [merchant],
        'category': [category],
        'amt': [amount],
        'gender': [gender],
        'state': [state],
        'zip': [zip_code],
        'city_pop': [city_pop],
        'job': [job],
        'day': [day],
        'month': [month],
        'year': [year],
        'hour': [hour],
        'minute': [minute],
        'age': [age]
    }

    input_df = pd.DataFrame(input_dict)

    # Apply preprocessing if needed (encode, scale, etc.)
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Output
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Transaction is **Fraudulent** with {prob*100:.2f}% confidence.")
    else:
        st.success(f"✅ Transaction is **Legitimate** with {100 - prob*100:.2f}% confidence.")

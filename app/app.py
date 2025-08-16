import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the Saved Model Pipeline ---
# We load the pipeline object we created and saved in the previous step.
# This pipeline includes scaling, PCA, and the final tuned Logistic Regression model.
pipeline = joblib.load('. ./credit_risk_pipeline.joblib')

# --- 2. Set Up the Streamlit Page ---
st.set_page_config(page_title="CreditGuard AI", page_icon="🛡️", layout="centered")

# Add a title and a short description
st.title("🛡️ CreditGuard AI")
st.write("""
This app predicts the risk of a loan applicant defaulting on their credit line.
Enter the applicant's financial details below to get a real-time risk assessment.
""")

# --- 3. Create Input Fields for User Data ---
st.header("Applicant's Financial Details")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    ext_risk_estimate = st.number_input(
        'External Risk Estimate',
        min_value=0, max_value=100, value=65,
        help="A consolidated risk score from external credit bureaus (e.g., FICO)."
    )
    msince_oldest_trade = st.number_input(
        'Months Since Oldest Trade Open',
        min_value=0, max_value=1000, value=240,
        help="How many months ago the applicant's oldest credit account was opened."
    )
    avg_min_file = st.number_input(
        'Average Months in File',
        min_value=0, max_value=1000, value=84,
        help="The average age in months of all credit accounts."
    )
    num_satisfactory_trades = st.number_input(
        'Number of Satisfactory Trades',
        min_value=0, max_value=100, value=20
    )
    percent_trades_never_delq = st.number_input(
        'Percent of Trades Never Delinquent',
        min_value=0, max_value=100, value=95
    )

with col2:
    msince_recent_trade = st.number_input(
        'Months Since Most Recent Trade',
        min_value=0, max_value=500, value=6
    )
    max_delq_2yrs = st.number_input(
        'Max Delinquency in Last 2 Years',
        min_value=0, max_value=10, value=1,
        help="The highest level of delinquency (e.g., 30, 60, 90 days past due)."
    )
    num_total_trades = st.number_input(
        'Total Number of Trades',
        min_value=0, max_value=200, value=30
    )
    num_trades_open_in_12m = st.number_input(
        'Number of Trades Opened in Last 12 Months',
        min_value=0, max_value=50, value=2
    )
    net_fraction_revolving_balance = st.number_input(
        'Net Fraction of Revolving Balance',
        min_value=0, max_value=500, value=30,
        help="The percentage of available revolving credit that is being used."
    )

# --- 4. Prediction Logic ---
# This block runs when the user clicks the "Predict" button.
if st.button('**Predict Risk**', type="primary"):

    # Create a DataFrame from the user's input
    # The column names MUST match the names used during model training.
    # We use the top 10 features from your feature importance plot.
    input_data = pd.DataFrame({
        'ExternalRiskEstimate': [ext_risk_estimate],
        'MSinceOldestTradeOpen': [msince_oldest_trade],
        'AverageMInFile': [avg_min_file],
        'NumSatisfactoryTrades': [num_satisfactory_trades],
        'PercentTradesNeverDelq': [percent_trades_never_delq],
        'MSinceMostRecentTradeOpen': [msince_recent_trade],
        'MaxDelq2PublicRecLast12M': [max_delq_2yrs],
        'NumTotalTrades': [num_total_trades],
        'NumTradesOpeninLast12M': [num_trades_open_in_12m],
        'NetFractionRevolvingBurden': [net_fraction_revolving_balance]
        # NOTE: Add the rest of the columns your model expects,
        # even if they are not in the top 10. For now, we'll fill with zeros.
    })

    # Get the full list of feature names the model was trained on
    # (excluding the one-hot encoded 'Job' columns for simplicity here)
    expected_features = [
        'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
        'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec',
        'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
        'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M',
        'PercentInstallmentTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M',
        'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallmentBurden',
        'NumRevolvingTradesWBalance', 'NumInstallmentTradesWBalance',
        'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance'
    ]

    # Add any missing columns and fill them with 0
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    # Ensure the order of columns is the same as during training
    input_data = input_data[expected_features]

    # Use the loaded pipeline to make a prediction
    prediction = pipeline.predict(input_data)
    prediction_proba = pipeline.predict_proba(input_data)

    # --- 5. Display the Result ---
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error('**High Risk: Loan Default Predicted**', icon="🚨")
        st.write(f"The model predicts a **{prediction_proba[0][1]*100:.2f}%** probability of default.")
    else:
        st.success('**Low Risk: Loan Repayment Predicted**', icon="✅")
        st.write(f"The model predicts a **{prediction_proba[0][0]*100:.2f}%** probability of successful repayment.")


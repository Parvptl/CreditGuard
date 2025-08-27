import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the Saved Model Pipeline ---
# CORRECTED: Use a simpler relative path.
# This tells the script to look for the model in the main project root directory.
pipeline = joblib.load('.\credit_risk_pipeline.joblib')

# Add a title and a short description
st.title("🛡️ CreditGuard AI")
st.write("""
This app predicts the risk of a loan applicant defaulting on their credit line.
Enter the applicant's financial details below to get a real-time risk assessment.
""")

# --- Helper function for the risk meter ---
def create_risk_meter(probability):
    """Creates a visually appealing SVG gauge for the risk probability."""
    # Determine color based on probability
    if probability < 50:
        color = "#28a745"  # Green
    elif 50 <= probability < 70:
        color = "#ffc107"  # Yellow
    else:
        color = "#dc3545"  # Red

    # SVG for the gauge
    gauge_html = f"""
    <div style="text-align: center;">
        <svg width="250" height="150">
            <defs>
                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#28a745;stop-opacity:1" />
                    <stop offset="50%" style="stop-color:#ffc107;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#dc3545;stop-opacity:1" />
                </linearGradient>
            </defs>
            <path d="M 50 120 A 70 70 0 0 1 200 120" stroke="url(#gaugeGradient)" stroke-width="20" fill="none" />
            <text x="125" y="110" font-family="sans-serif" font-size="30" text-anchor="middle" fill="#333">{probability:.1f}%</text>
            <text x="125" y="135" font-family="sans-serif" font-size="14" text-anchor="middle" fill="#666">Default Probability</text>
            <line x1="125" y1="120" x2="{125 + 70 * np.cos((1 - probability/100) * np.pi - np.pi)}" y2="{120 - 70 * np.sin((1 - probability/100) * np.pi)}"
                  stroke="#333" stroke-width="3" />
        </svg>
    </div>
    """
    return gauge_html

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
    # CORRECTED FEATURE NAMES BELOW
    expected_features = [
        'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
        'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec',
        'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
        'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M',
        'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M',
        'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
        'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
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
    default_probability = prediction_proba[0][1] * 100

    # --- 5. Display the Result ---
    st.header("Prediction Result")
    
    # Display the risk meter
    st.markdown(create_risk_meter(default_probability), unsafe_allow_html=True)

    # Display a more detailed message
    if prediction[0] == 1:
        st.error('**Conclusion: High Risk**', icon="🚨")
        st.write(f"The model has identified this applicant as having a high risk of default. The confidence in this prediction is moderate, as indicated by the probability score.")
    else:
        st.success('**Conclusion: Low Risk**', icon="✅")
        st.write(f"The model has identified this applicant as having a low risk of default. The probability of successful repayment is **{prediction_proba[0][0]*100:.2f}%**.")

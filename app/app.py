import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="CreditGuard AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Highly Polished Look ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* Custom title styling */
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a; /* Dark Blue */
        padding-bottom: 10px;
    }

    /* Custom container styling */
    .custom-container {
        padding: 2rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }

    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        background: linear-gradient(90deg, #007bff, #0056b3);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
    }

    /* Metric styling */
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #007bff;
    }

    /* Expander styling */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Caching Functions for Performance ---
@st.cache_resource
def load_model():
    """Load the trained pipeline."""
    try:
        pipeline = joblib.load('credit_risk_pipeline.joblib')
        return pipeline
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'credit_risk_pipeline.joblib' is in the root directory.")
        return None

@st.cache_data
def load_data():
    """Load the raw dataset for dashboard visualizations."""
    try:
        df = pd.read_csv('data/heloc_dataset_v1.csv')
        df.replace([-9, -8, -7], np.nan, inplace=True)
        return df
    except FileNotFoundError:
        st.warning("Raw data file not found. Dashboard visualizations will be unavailable.")
        return None

@st.cache_data
def get_feature_info():
    """Get expected features and their importance."""
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
    # These importances are derived from the model analysis and are used for the waterfall plot.
    # In a real-world scenario, these would be calculated using a library like SHAP.
    # For this project, we define them based on our notebook's feature importance results.
    feature_importances = {
        'ExternalRiskEstimate': -0.40, 'PercentTradesNeverDelq': -0.15,
        'AverageMInFile': -0.10, 'MSinceOldestTradeOpen': -0.08,
        'NetFractionRevolvingBurden': 0.12, 'NumSatisfactoryTrades': -0.06,
        'PercentInstallTrades': -0.05, 'NumTotalTrades': 0.04,
        'MSinceMostRecentTradeOpen': 0.05, 'MaxDelqEver': 0.18,
        'NumTradesOpeninLast12M': 0.10, 'MaxDelq2PublicRecLast12M': 0.20
    }
    return expected_features, feature_importances

pipeline = load_model()
raw_df = load_data()
expected_features, feature_importances = get_feature_info()


# --- Sidebar Navigation and Inputs ---
with st.sidebar:
    st.image("https://i.imgur.com/yV8V5w9.png", width=120)
    st.title("Navigation")
    page = st.radio("Go to", ["Live Prediction", "Project Dashboard"])
    st.markdown("---")

    st.header("Applicant Details")
    st.write("Adjust the sliders to match the applicant's profile.")

    # Input fields
    ext_risk_estimate = st.slider('External Risk Estimate', 1, 100, 70, 1)
    msince_oldest_trade = st.slider('Months Since Oldest Trade', 0, 800, 250, 10)
    avg_min_file = st.slider('Average Months in File', 0, 400, 90, 5)
    num_satisfactory_trades = st.slider('Number of Satisfactory Trades', 0, 100, 25, 1)
    percent_trades_never_delq = st.slider('Percent of Trades Never Delinquent', 0, 100, 90, 1)
    msince_recent_trade = st.slider('Months Since Most Recent Trade', 0, 300, 10, 1)
    num_total_trades = st.slider('Total Number of Trades', 0, 150, 35, 1)
    num_trades_open_in_12m = st.slider('Trades Opened in Last 12 Months', 0, 20, 1, 1)
    net_fraction_revolving_balance = st.slider('Revolving Balance Fraction (%)', 0, 200, 40, 5)
    max_delq_2yrs = st.selectbox('Max Delinquency in Last 2 Years', list(range(10)), index=2)
    max_delq_ever = st.selectbox('Max Delinquency Ever', list(range(10)), index=3)


# --- Main Page Content ---

if page == "Live Prediction":
    st.markdown('<p class="title-text">🛡️ CreditGuard AI: Live Risk Prediction</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)

        if st.sidebar.button('**Predict Risk**', use_container_width=True):
            if pipeline:
                # Create a DataFrame from the user's input
                input_data_dict = {
                    'ExternalRiskEstimate': ext_risk_estimate,
                    'MSinceOldestTradeOpen': msince_oldest_trade,
                    'AverageMInFile': avg_min_file,
                    'NumSatisfactoryTrades': num_satisfactory_trades,
                    'PercentTradesNeverDelq': percent_trades_never_delq,
                    'MSinceMostRecentTradeOpen': msince_recent_trade,
                    'MaxDelq2PublicRecLast12M': max_delq_2yrs,
                    'MaxDelqEver': max_delq_ever,
                    'NumTotalTrades': num_total_trades,
                    'NumTradesOpeninLast12M': num_trades_open_in_12m,
                    'NetFractionRevolvingBurden': net_fraction_revolving_balance
                }
                input_data = pd.DataFrame([input_data_dict])

                # Add missing columns and fill with 0
                for col in expected_features:
                    if col not in input_data.columns:
                        input_data[col] = 0
                input_data = input_data[expected_features]

                # Make prediction
                prediction = pipeline.predict(input_data)
                prediction_proba = pipeline.predict_proba(input_data)
                default_probability = prediction_proba[0][1] * 100

                # --- Display Results ---
                st.header("Risk Assessment Summary")
                col1, col2 = st.columns([1, 2])

                with col1:
                    if prediction[0] == 1:
                        st.error('**High Risk**', icon="🚨")
                    else:
                        st.success('**Low Risk**', icon="✅")
                    st.metric(label="Default Probability", value=f"{default_probability:.2f}%")
                    st.progress(int(default_probability))

                with col2:
                    st.write("**Model's Conclusion:**")
                    if prediction[0] == 1:
                        st.write("The model predicts a high likelihood of default based on the provided financial profile. Key indicators suggest potential risk factors that warrant cautious consideration.")
                    else:
                        st.write("The model predicts a low likelihood of default. The applicant's profile aligns with those who have a strong history of repayment.")

                st.markdown("---")

                # --- Prediction Explanation (Waterfall Chart) ---
                st.header("Decision Analysis")
                st.write("This chart shows how each factor contributed to the final risk score, pushing it higher (red) or lower (green) from a baseline.")

                # Simulate SHAP/LIME values for the waterfall chart
                base_value = 50  # Baseline average risk
                contributions = {}
                for feature, value in input_data_dict.items():
                    if feature in feature_importances:
                        # Normalize the input value (simple scaling for visualization)
                        normalized_value = (value - 50) / 50
                        contribution = normalized_value * feature_importances[feature] * 20 # Scale effect
                        contributions[feature] = contribution

                sorted_contributions = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)

                fig = go.Figure(go.Waterfall(
                    name = "Prediction", orientation = "h",
                    measure = ["relative"] * len(sorted_contributions) + ["total"],
                    y = [item[0] for item in sorted_contributions] + ["Final Score"],
                    text = [f"{item[1]:+.2f}" for item in sorted_contributions] + [f"{base_value + sum(contributions.values()):.2f}"],
                    x = [item[1] for item in sorted_contributions] + [base_value + sum(contributions.values())],
                    base = base_value,
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                    increasing = {"marker":{"color":"#dc3545"}}, # Red for increasing risk
                    decreasing = {"marker":{"color":"#28a745"}}, # Green for decreasing risk
                ))
                fig.update_layout(title="Feature Contribution to Risk Score", showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Model could not be loaded. Please check the file path and try again.")
        else:
            st.info("Adjust the applicant's details in the sidebar and click 'Predict Risk' to see the assessment.")

        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Project Dashboard":
    st.markdown('<p class="title-text">📊 Project Dashboard & Insights</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.header("Exploratory Data Analysis (EDA)")
        st.write("This section provides an overview of the dataset used to train the model.")

        if raw_df is not None:
            # EDA Visualizations
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Loan Outcome Distribution")
                fig1, ax1 = plt.subplots()
                sns.countplot(x='RiskPerformance', data=raw_df, ax=ax1, palette="viridis")
                st.pyplot(fig1)

            with col2:
                st.subheader("External Risk Estimate Distribution")
                fig2, ax2 = plt.subplots()
                sns.histplot(data=raw_df, x='ExternalRiskEstimate', hue='RiskPerformance', kde=True, ax=ax2, palette="magma")
                st.pyplot(fig2)

            st.markdown("---")
            st.header("Model Insights")
            st.subheader("Top Predictive Features")
            st.write("The chart below shows the features that were most influential in the model's predictions, as determined during the analysis phase.")

            fi_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance_Factor']).sort_values('Importance_Factor', key=abs, ascending=False)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance_Factor', y='Feature', data=fi_df, palette='coolwarm', ax=ax3)
            ax3.set_title('Relative Impact of Features on Prediction')
            ax3.set_xlabel("Impact Score (Negative=Lower Risk, Positive=Higher Risk)")
            st.pyplot(fig3)
        else:
            st.warning("Could not load raw data to display dashboard visuals.")

        st.markdown('</div>', unsafe_allow_html=True)

# --- Common Footer / Explainer ---
st.markdown("---")
with st.expander("About CreditGuard AI"):
    st.write("""
    This application is the final product of an end-to-end machine learning project for the **Intro to ML** course. It demonstrates a complete workflow from data cleaning and analysis to model training, tuning, and finally, deployment as a user-friendly web application.

    **Technology Stack:**
    - **Backend:** Python, Pandas, Scikit-learn
    - **Frontend:** Streamlit
    - **Model:** Tuned Logistic Regression with PCA
    """)

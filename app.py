# SaaS Revenue Intelligence Platform (Business Analytics & Churn Prediction System)

import streamlit as st
import pandas as pd
import src.data_loader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.predict import ChurnPredictor
from src.analysis_utils import DataAnalyzer

# Page configuration
st.set_page_config(
    page_title="SaaS Revenue Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/cleaned_data.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_predictor():
    return ChurnPredictor()

# Initialize session
if 'df' not in st.session_state:
    st.session_state.df = load_data()

if 'predictor' not in st.session_state:
    st.session_state.predictor = load_predictor()

if 'analyzer' not in st.session_state:
    if st.session_state.df is not None:
        st.session_state.analyzer = DataAnalyzer(st.session_state.df)

# Header
st.markdown('<p class="main-title">SaaS Revenue Intelligence Platform</p>', 
           unsafe_allow_html=True)
st.markdown('<p class="subtitle">Business Analytics • Churn Prediction • Revenue Optimization</p>', 
           unsafe_allow_html=True)

# Check data
if st.session_state.df is None:
    st.error("Data file not found! Please ensure 'data/cleaned_data.csv' exists.")
    st.stop()

df = st.session_state.df

# Business columns for analysis
BUSINESS_COLUMNS = [
    "company_size", "industry", "discount_pct", "regime_state",
    "usage_growth", "ticket_growth", "payment_delay_flag"
]

# Quick Stats Row
st.markdown("### Business Performance Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Accounts", f"{df['account_id'].nunique():,}")
with col2:
    st.metric("Total MRR", f"${df['current_mrr'].sum():,.0f}")
with col3:
    st.metric("Avg MRR/Account", f"${df['current_mrr'].mean():,.0f}")
with col4:
    churn_rate = (df['next_month_mrr'] < df['current_mrr']).mean()
    st.metric("Churn Rate", f"{churn_rate:.1%}")
with col5:
    delayed_pct = df['payment_delay_flag'].mean()
    st.metric("Payment Delays", f"{delayed_pct:.1%}")

# Business Metrics Cards
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Revenue Health")
    st.markdown("*Revenue distribution, churn impact, financial risk*")
    st.markdown("Analyze revenue patterns and identify financial risks")
    st.page_link("pages/1_Revenue_Health.py", label="→ Revenue Health Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Business Drivers")
    st.markdown("*Feature relationships, business impact, key metrics*")
    st.markdown("Discover what drives revenue and customer success")
    st.page_link("pages/2_Business_Drivers.py", label="→ Business Drivers Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Risk Patterns")
    st.markdown("*Correlation patterns, risk indicators, churn signals*")
    st.markdown("Identify patterns that signal customer risk")
    st.page_link("pages/3_Risk_Patterns.py", label="→ Risk Patterns Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Cards
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Churn Forecasting")
    st.markdown("*Predict future churn using ML models on your data*")
    st.markdown("Run predictions on existing accounts to identify at-risk customers")
    st.page_link("pages/4_Churn_Forecasting.py", label="→ Run Churn Forecast")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Account Risk Profiler")
    st.markdown("*Interactive risk assessment for individual accounts*")
    st.markdown("Enter account details to get personalized churn risk score")
    st.page_link("pages/5_Account_Risk_Profiler.py", label="→ Profile Account Risk")
    st.markdown('</div>', unsafe_allow_html=True)

# Quick Business Insights
st.markdown("---")
st.markdown('<p class="section-header">Business Insights at a Glance</p>', unsafe_allow_html=True)

insight_cols = st.columns(4)

with insight_cols[0]:
    enterprise_rev = df[df['company_size'] == 'Enterprise']['current_mrr'].mean()
    avg_rev = df['current_mrr'].mean()
    st.metric("Enterprise vs Avg MRR", f"{enterprise_rev/avg_rev:.1f}x")
    st.caption("Enterprise accounts revenue multiplier")

with insight_cols[1]:
    high_adoption = df[df['feature_adoption_rate'] > 0.7]['current_mrr'].mean()
    low_adoption = df[df['feature_adoption_rate'] < 0.3]['current_mrr'].mean()
    st.metric("High vs Low Adoption MRR", f"{high_adoption/low_adoption:.1f}x" if low_adoption > 0 else "N/A")
    st.caption("Revenue impact of feature adoption")

with insight_cols[2]:
    delayed_churn = df[df['payment_delay_flag'] == 1]['next_month_mrr'].lt(df[df['payment_delay_flag'] == 1]['current_mrr']).mean()
    ontime_churn = df[df['payment_delay_flag'] == 0]['next_month_mrr'].lt(df[df['payment_delay_flag'] == 0]['current_mrr']).mean()
    st.metric("Delayed vs On-Time Churn", f"{delayed_churn/ontime_churn:.1f}x" if ontime_churn > 0 else "N/A")
    st.caption("Churn rate for delayed payment accounts")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>SaaS Revenue Intelligence Platform</strong> • AI-Powered Business Analytics</p>
    <p style="font-size: 0.9rem;">Business Metrics: company_size • industry • discount_pct • regime_state • usage_growth • ticket_growth • payment_delay_flag</p>
</div>
""", unsafe_allow_html=True)
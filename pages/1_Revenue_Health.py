# Revenue distribution, churn impact, and financial risk analysis

import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Revenue Health", layout="wide")

if 'df' not in st.session_state:
    st.error("Please go to the main page first")
    st.stop()

df = st.session_state.df
analyzer = st.session_state.analyzer

BUSINESS_COLUMNS = [
    "company_size", "industry", "discount_pct", "regime_state",
    "usage_growth", "ticket_growth", "payment_delay_flag"
]

st.title("Revenue Health Analysis")
st.markdown("Analyze how business metrics impact revenue and identify financial risks")

# Sidebar column selector
st.sidebar.markdown("### Select Business Metric")
selected_col = st.sidebar.selectbox("Choose metric to analyze:", BUSINESS_COLUMNS)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Selected:** {selected_col}")
st.sidebar.markdown(f"**Type:** {'Categorical' if selected_col in analyzer.categorical_cols else 'Numeric'}")

# Analysis
fig, insights = analyzer.analyze_business_distribution(selected_col)
st.subheader(f"{selected_col} - Revenue Health Impact")
st.plotly_chart(fig, use_container_width=True)

# Business Insights
st.subheader("Business Insights")
for insight in insights:
    st.markdown(f"<div class='highlight-box'>{insight}</div>", unsafe_allow_html=True)

# Revenue summary metrics
st.markdown("---")
st.subheader("Revenue Overview by Segment")
col1, col2 = st.columns(2)

with col1:
    # Revenue by company size
    rev_by_size = df.groupby('company_size')['current_mrr'].sum().reset_index()
    fig = px.pie(rev_by_size, values='current_mrr', names='company_size',
                 title="Revenue Share by Company Size")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Revenue by industry
    rev_by_industry = df.groupby('industry')['current_mrr'].sum().nlargest(10).reset_index()
    fig = px.bar(rev_by_industry, x='industry', y='current_mrr',
                 title="Top 10 Industries by Revenue",
                 color='current_mrr', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# Churn impact
st.subheader("Churn Risk Analysis")
df['churn_flag'] = (df['next_month_mrr'] < df['current_mrr']).astype(int)
col1, col2, col3 = st.columns(3)

with col1:
    overall_churn = df['churn_flag'].mean()
    st.metric("Overall Churn Rate", f"{overall_churn:.1%}")
with col2:
    revenue_at_risk = df[df['churn_flag'] == 1]['current_mrr'].sum()
    st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")
with col3:
    accounts_at_risk = df['churn_flag'].sum()
    st.metric("Accounts at Risk", f"{accounts_at_risk:,}")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/2_Business_Drivers.py", label="→ Business Drivers Analysis")
with col2:
    st.page_link("pages/4_Churn_Forecasting.py", label="→ Run Churn Forecast")
# Feature relationships and business impact analysis

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Business Drivers", layout="wide")

if 'df' not in st.session_state:
    st.error("Please go to the main page first")
    st.stop()

df = st.session_state.df
analyzer = st.session_state.analyzer

BUSINESS_COLUMNS = [
    "company_size", "industry", "discount_pct", "regime_state",
    "usage_growth", "ticket_growth", "payment_delay_flag"
]

st.title("Business Drivers Analysis")
st.markdown("Discover relationships between business metrics and their impact on performance")

# Sidebar
st.sidebar.markdown("### Select Business Metrics")
col1 = st.sidebar.selectbox("Metric X:", BUSINESS_COLUMNS, index=0)
col2 = st.sidebar.selectbox("Metric Y:", BUSINESS_COLUMNS, index=1)

# Analysis
fig, insights = analyzer.analyze_business_relationship(col1, col2)
st.subheader(f"{col1} vs {col2}")
st.plotly_chart(fig, use_container_width=True)

# Business Insights
st.subheader("Relationship Insights")
for insight in insights:
    st.markdown(f"<div class='highlight-box'>{insight}</div>", unsafe_allow_html=True)

# Key business drivers summary
st.markdown("---")
st.subheader("Key Business Drivers Impact on Revenue")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Discount Impact**")
    discount_corr = df['discount_pct'].corr(df['current_mrr'])
    st.metric("Correlation with Revenue", f"{discount_corr:.3f}")
    
    high_disc = df[df['discount_pct'] > 0.3]['current_mrr'].mean()
    low_disc = df[df['discount_pct'] <= 0.1]['current_mrr'].mean()
    st.metric("High vs Low Discount MRR", f"{high_disc/low_disc:.1f}x" if low_disc > 0 else "N/A")

with col2:
    st.markdown("**Usage Growth Impact**")
    usage_corr = df['usage_growth'].corr(df['current_mrr'])
    st.metric("Correlation with Revenue", f"{usage_corr:.3f}")
    
    pos_growth = df[df['usage_growth'] > 0]['current_mrr'].mean()
    neg_growth = df[df['usage_growth'] <= 0]['current_mrr'].mean()
    st.metric("Growth vs Decline MRR", f"{pos_growth/neg_growth:.1f}x" if neg_growth > 0 else "N/A")

with col3:
    st.markdown("**Support Ticket Impact**")
    ticket_corr = df['ticket_growth'].corr(df['current_mrr'])
    st.metric("Correlation with Revenue", f"{ticket_corr:.3f}")
    
    high_tickets = df[df['ticket_growth'] > 0.2]['current_mrr'].mean()
    low_tickets = df[df['ticket_growth'] <= 0]['current_mrr'].mean()
    st.metric("High vs Low Ticket MRR", f"{high_tickets/low_tickets:.1f}x" if low_tickets > 0 else "N/A")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Revenue_Health.py", label="← Revenue Health")
with col2:
    st.page_link("pages/3_Risk_Patterns.py", label="→ Risk Patterns")
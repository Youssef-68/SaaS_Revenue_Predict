# Correlation patterns, risk indicators, and churn signals

import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Risk Patterns", page_icon="🔍", layout="wide")

if 'df' not in st.session_state:
    st.error("Please go to the main page first")
    st.stop()

df = st.session_state.df
analyzer = st.session_state.analyzer

BUSINESS_COLUMNS = [
    "company_size", "industry", "discount_pct", "regime_state",
    "usage_growth", "ticket_growth", "payment_delay_flag"
]

st.title("Risk Pattern Analysis")
st.markdown("Identify patterns and signals that indicate customer churn risk")

# Create churn flag
df['churn_flag'] = (df['next_month_mrr'] < df['current_mrr']).astype(int)

# Correlation with churn
st.subheader("Business Metrics Correlation with Churn")
corr_fig, corr_insights = analyzer.business_correlation_matrix()
st.plotly_chart(corr_fig, use_container_width=True)

for insight in corr_insights:
    st.markdown(f"<div class='highlight-box'>{insight}</div>", unsafe_allow_html=True)

# Risk indicators by business metric
st.markdown("---")
st.subheader("Risk Indicators by Business Metric")
selected_metric = st.selectbox("Select metric for risk analysis:", BUSINESS_COLUMNS)


if selected_metric in analyzer.categorical_cols:
    
    # Churn rate by category
    churn_by_metric = df.groupby(selected_metric).agg({
        'churn_flag': 'mean',
        'current_mrr': ['sum', 'count']
    }).round(3)
    churn_by_metric.columns = ['Churn Rate', 'Total MRR', 'Account Count']
    churn_by_metric = churn_by_metric.sort_values('Churn Rate', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            churn_by_metric.reset_index(), x=selected_metric, y='Churn Rate',
            title=f"Churn Rate by {selected_metric}",
            color='Churn Rate', color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            churn_by_metric.style
            .background_gradient(subset=['Churn Rate'], cmap='Reds')
            .format({'Churn Rate': '{:.1%}', 'Total MRR': '${:,.0f}'}),
            use_container_width=True
        )
    
    # Highest risk category
    highest_risk_cat = churn_by_metric.index[0]
    highest_risk_rate = churn_by_metric['Churn Rate'].iloc[0]
    st.warning(f"**{highest_risk_cat}** shows highest churn risk at **{highest_risk_rate:.1%}**")

else:
    # Churn vs numeric metric
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"{selected_metric} Distribution by Churn",
        f"Churn Rate by {selected_metric} Quartiles"
    ))
    
    # Box plot
    fig.add_trace(
        go.Box(x=df['churn_flag'].map({0: 'Retained', 1: 'Churned'}), 
               y=df[selected_metric], name="Distribution",
               marker_color='coral'),
        row=1, col=1
    )
    
    # Quartile analysis
    df['quartile'] = pd.qcut(df[selected_metric], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    churn_by_quartile = df.groupby('quartile')['churn_flag'].mean()
    
    fig.add_trace(
        go.Bar(x=churn_by_quartile.index, y=churn_by_quartile.values,
               name="Churn Rate", marker_color='red'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation
    corr = df[selected_metric].corr(df['churn_flag'])
    st.info(f"Correlation with churn: **{corr:.3f}** - {'Higher values = More risk' if corr > 0 else 'Lower values = More risk'}")

# Payment delay risk
st.markdown("---")
st.subheader("Payment Delay - Risk Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    delayed_churn = df[df['payment_delay_flag'] == 1]['churn_flag'].mean()
    ontime_churn = df[df['payment_delay_flag'] == 0]['churn_flag'].mean()
    st.metric("Delayed Payment Churn", f"{delayed_churn:.1%}")
    st.caption(f"vs {ontime_churn:.1%} for on-time payers")

with col2:
    delayed_mrr = df[df['payment_delay_flag'] == 1]['current_mrr'].sum()
    st.metric("Revenue with Delays", f"${delayed_mrr:,.0f}")

with col3:
    delay_rate = df['payment_delay_flag'].mean
    
# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/2_Business_Drivers.py", label="← Business Drivers")
with col2:
    st.page_link("pages/4_Churn_Forecasting.py", label="→ Churn Forecasting")
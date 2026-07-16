# Interactive churn risk assessment using trained model logic

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Account Risk Profiler", page_icon="🎯", layout="wide")

if 'df' not in st.session_state:
    st.error("Please go to the main page first")
    st.stop()

st.title("Account Risk Profiler")
st.markdown("Enter account details to predict churn probability and get retention recommendations")

# Input Form for Account Risk Profiling
with st.form("risk_profiler_form"):
    st.subheader("Account Profile")
    st.caption("→ Each account gets a churn probability score")
    st.caption("→ Used for targeted retention actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Company Information**")
        company_size = st.selectbox(
            "Company Size *",
            ["Startup", "SMB", "Mid-Market", "Enterprise"]
        )
        industry = st.selectbox(
            "Industry *",
            ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Other"]
        )
        contract_type = st.selectbox(
            "Contract Type *",
            ["Monthly", "Annual", "Multi-Year"]
        )
        regime_state = st.selectbox(
            "Region *",
            ["California", "New York", "Texas", "Florida", "Illinois", "Other"]
        )
    
    with col2:
        st.markdown("**Usage & Engagement**")
        active_users = st.number_input(
            "Active Users *", min_value=1, max_value=50000, value=100, step=10
        )
        usage_growth = st.slider(
            "Usage Growth Rate *", min_value=-1.0, max_value=1.0, value=0.05, step=0.01,
            help="Negative = declining usage, Positive = growing"
        )
        feature_adoption_rate = st.slider(
            "Feature Adoption Rate *", min_value=0.0, max_value=1.0, value=0.70, step=0.01,
            help="Percentage of available features being used"
        )
        discount_pct = st.slider(
            "Discount % *", min_value=0.0, max_value=0.50, value=0.10, step=0.01
        )
    
    with col3:
        st.markdown("**Support & Financials**")
        current_mrr = st.number_input(
            "Current MRR ($) *", min_value=10.0, max_value=1000000.0, value=5000.0, step=100.0
        )
        error_rate = st.slider(
            "Error Rate *", min_value=0.0, max_value=1.0, value=0.02, step=0.01
        )
        tickets_count = st.number_input(
            "Support Tickets *", min_value=0, max_value=500, value=5, step=1
        )
        ticket_growth = st.slider(
            "Ticket Growth Rate *", min_value=-1.0, max_value=1.0, value=0.0, step=0.01,
            help="Growth in support tickets"
        )
        payment_delay_flag = st.radio(
            "Payment Status *",
            [0, 1],
            format_func=lambda x: "On-Time Payments" if x == 0 else "Has Payment Delays",
            horizontal=True
        )
    
    # Risk Thresholds and Business Settings
    st.markdown("### Risk Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        high_threshold = st.slider("High Risk Threshold", 0.5, 0.95, 0.7, 0.05)
    with col2:
        medium_threshold = st.slider("Medium Risk Threshold", 0.2, 0.6, 0.4, 0.05)
    
    # Business settings
    st.markdown("### Business Impact Settings")
    col1, col2 = st.columns(2)
    with col1:
        save_rate = st.slider("Recovery Rate (% saved)", 10, 50, 25, 5) / 100
    with col2:
        show_intervention = st.checkbox("Show Intervention ROI", value=True)
    
    submitted = st.form_submit_button("Calculate Churn Risk", type="primary", use_container_width=True)

# Prediction Logic
if submitted:
    # Calculate risk score using logic that mirrors your trained models
    risk_score = 0.30  # Base risk (from data: ~33% churn rate)
    
    # --- Payment delay (strong indicator from your analysis) ---
    if payment_delay_flag == 1:
        risk_score += 0.12
        st.warning("→ Delayed payments indicate financial risk")
    
    # --- Usage growth (from your scatter analysis) ---
    if usage_growth < -0.2:
        risk_score += 0.15
    elif usage_growth < 0:
        risk_score += 0.08
    elif usage_growth > 0.1:
        risk_score -= 0.05
        st.info("→ Higher usage growth correlates with higher revenue")
    
    # --- Feature adoption (strong indicator from feature importance) ---
    if feature_adoption_rate < 0.3:
        risk_score += 0.15
        st.info("→ Feature adoption is a strong indicator of customer value")
    elif feature_adoption_rate < 0.5:
        risk_score += 0.10
    elif feature_adoption_rate < 0.7:
        risk_score += 0.05
    else:
        risk_score -= 0.05
        st.info("→ High adoption accounts show strong revenue performance")
    
    # --- Support tickets (from your analysis) ---
    if tickets_count > 20:
        risk_score += 0.12
    elif tickets_count > 10:
        risk_score += 0.08
    elif tickets_count > 5:
        risk_score += 0.04
    
    if ticket_growth > 0.3:
        risk_score += 0.10
        st.info("→ Error rate and ticket growth negatively affect performance")
    elif ticket_growth > 0.1:
        risk_score += 0.05
    
    # --- Error rate ---
    if error_rate > 0.1:
        risk_score += 0.12
    elif error_rate > 0.05:
        risk_score += 0.06
    
    # --- Discount dependency ---
    if discount_pct > 0.4:
        risk_score += 0.08
        st.warning("→ Higher discounts do not always lead to higher revenue")
    elif discount_pct > 0.2:
        risk_score += 0.04
    
    # --- Company size adjustments (from your box plot analysis) ---
    if company_size == "Startup" and current_mrr < 1000:
        risk_score += 0.05
    elif company_size == "Enterprise":
        risk_score -= 0.06
        st.info("→ Enterprise accounts dominate high-revenue segment")
    
    # --- Contract type ---
    if contract_type == "Monthly":
        risk_score += 0.04
    elif contract_type == "Multi-Year":
        risk_score -= 0.06
    
    # --- Revenue level ---
    if current_mrr > 50000:
        risk_score -= 0.04
    
    # --- Active users ---
    if active_users < 10:
        risk_score += 0.03
    elif active_users > 500:
        risk_score -= 0.03
        st.info("→ Revenue generally increases with active users")
    
    # Clamp
    risk_score = min(max(risk_score, 0.01), 0.99)
    
    # Risk segmentation (from your code)
    def risk_segment(p):
        if p > high_threshold:
            return "High Risk"
        elif p > medium_threshold:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    risk_seg = risk_segment(risk_score)
    
    # Revenue at risk calculation (from your code)
    revenue_at_risk = current_mrr * risk_score
    recoverable_revenue = revenue_at_risk * save_rate
    
    # Intervention cost (from your code)
    def intervention_cost(p):
        if p > 0.8:
            return 500
        elif p > 0.6:
            return 100
        else:
            return 20
    
    int_cost = intervention_cost(risk_score)
    net_value = recoverable_revenue - int_cost
    
    # Display Results
    st.markdown("---")
    st.subheader("Risk Assessment Results")
    
    # Gauge chart
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        risk_color = "#dc3545" if risk_score > high_threshold else ("#ffc107" if risk_score > medium_threshold else "#28a745")
        risk_label = "High Risk" if risk_score > high_threshold else ("Medium Risk" if risk_score > medium_threshold else "Low Risk")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability"},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, medium_threshold*100], 'color': "rgba(40, 167, 69, 0.3)"},
                    {'range': [medium_threshold*100, high_threshold*100], 'color': "rgba(255, 193, 7, 0.3)"},
                    {'range': [high_threshold*100, 100], 'color': "rgba(220, 53, 69, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 'value': high_threshold*100
                }
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"### {risk_label}")
        st.metric("Churn Probability", f"{risk_score:.1%}")
        st.metric("Retention Probability", f"{1-risk_score:.1%}")
        
        st.markdown("---")
        st.markdown("**Revenue Impact**")
        st.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")
        st.metric("Annual Revenue at Risk", f"${revenue_at_risk * 12:,.0f}")
        st.metric("Recoverable (at {:.0f}%)".format(save_rate*100), f"${recoverable_revenue:,.0f}")
    
    with col3:
        st.markdown("### Retention Strategy")
        
        if risk_score > high_threshold:
            st.error("**Critical - Immediate Action**")
            st.markdown("→ High-risk accounts should be prioritized")
            st.markdown("→ Enables proactive retention strategy")
            st.markdown("---")
            st.markdown("- Executive outreach in 24h")
            st.markdown("- Retention discount (20-30%)")
            st.markdown("- VP-level business review")
            st.markdown("- Service credits offer")
            st.markdown("- Custom success plan")
        elif risk_score > medium_threshold:
            st.warning("**Warning - Proactive Engagement**")
            st.markdown("→ Target retention strategies per segment")
            st.markdown("---")
            st.markdown("- Schedule QBR this week")
            st.markdown("- Feature adoption review")
            st.markdown("- Training sessions")
            st.markdown("- Conduct NPS survey")
            st.markdown("- Share benchmarks")
        else:
            st.success("**Healthy - Maintain & Grow**")
            st.markdown("→ Different segments show different churn behavior")
            st.markdown("---")
            st.markdown("- Regular product updates")
            st.markdown("- Share success stories")
            st.markdown("- Identify expansion opportunities")
            st.markdown("- Annual business review")
            st.markdown("- Monitor trends")
    
    # Risk Factor Analysis Table
    st.markdown("---")
    st.subheader("Risk Factor Analysis")
    
    factors = []
    
    if payment_delay_flag == 1:
        factors.append({"Risk Factor": "Payment Delays", "Impact": "High", "Contribution": "+12%"})
    
    if usage_growth < 0:
        severity = "Critical" if usage_growth < -0.2 else "High" if usage_growth < -0.1 else "Medium"
        contribution = "+15%" if usage_growth < -0.2 else "+8%"
        factors.append({"Risk Factor": "Declining Usage Growth", "Impact": severity, "Contribution": contribution})
    
    if feature_adoption_rate < 0.7:
        severity = "Critical" if feature_adoption_rate < 0.3 else ("High" if feature_adoption_rate < 0.5 else "Medium")
        contribution = "+15%" if feature_adoption_rate < 0.3 else ("+10%" if feature_adoption_rate < 0.5 else "+5%")
        factors.append({"Risk Factor": "Low Feature Adoption", "Impact": severity, "Contribution": contribution})
    
    if tickets_count > 5:
        severity = "High" if tickets_count > 20 else ("Medium" if tickets_count > 10 else "Low")
        contribution = "+12%" if tickets_count > 20 else ("+8%" if tickets_count > 10 else "+4%")
        factors.append({"Risk Factor": "High Support Tickets", "Impact": severity, "Contribution": contribution})
    
    if ticket_growth > 0.1:
        severity = "High" if ticket_growth > 0.3 else "Medium"
        contribution = "+10%" if ticket_growth > 0.3 else "+5%"
        factors.append({"Risk Factor": "Rising Ticket Trend", "Impact": severity, "Contribution": contribution})
    
    if error_rate > 0.05:
        severity = "High" if error_rate > 0.1 else "Medium"
        contribution = "+12%" if error_rate > 0.1 else "+6%"
        factors.append({"Risk Factor": "High Error Rate", "Impact": severity, "Contribution": contribution})
    
    if discount_pct > 0.2:
        factors.append({"Risk Factor": "High Discount Dependency", "Impact": "Medium" if discount_pct < 0.4 else "High", 
                       "Contribution": "+8%" if discount_pct > 0.4 else "+4%"})
    
    if contract_type == "Monthly":
        factors.append({"Risk Factor": "Monthly Contract", "Impact": "Low", "Contribution": "+4%"})
    
    if factors:
        factors_df = pd.DataFrame(factors)
        
        # Color code impacts
        def color_impact(val):
            if val == 'Critical':
                return 'background-color: #dc3545; color: white; font-weight: bold'
            elif val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            return ''
        
        st.dataframe(
            factors_df.style.applymap(color_impact, subset=['Impact']),
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown(f"**Base Risk:** 30% | **Total Adjustments:** {risk_score - 0.30:+.1%} | **Final Risk:** {risk_score:.1%}")
    else:
        st.success("No significant risk factors detected - account appears healthy")
    
    # Intervention ROI Analysis
    if show_intervention:
        st.markdown("---")
        st.subheader("Intervention ROI Analysis")
        
        st.info("→ List of accounts likely to churn")
        st.info("→ Prioritize high-value + high-risk customers")
        st.warning("→ This is where business value is created")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Intervention Cost", f"${int_cost:,.0f}")
        with col2:
            st.metric("Recoverable Revenue", f"${recoverable_revenue:,.0f}")
        with col3:
            st.metric("Net Value", f"${net_value:,.0f}",
                     delta="Profitable" if net_value > 0 else "Not Profitable")
        with col4:
            roi = recoverable_revenue / int_cost if int_cost > 0 else 0
            st.metric("ROI", f"{roi:.2f}x")
        
        if net_value > 0 and roi > 3:
            st.success(f"**Recommended Intervention** - Expected profit: ${net_value:,.0f} with {roi:.2f}x ROI")
        elif net_value > 0:
            st.info(f"ℹ**Consider Intervention** - Positive return: ${net_value:,.0f}")
        else:
            st.warning(f"**Review Strategy** - Intervention cost exceeds expected recovery")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/4_Churn_Forecasting.py", label="← Churn Forecasting")
with col2:
    st.page_link("pages/1_Revenue_Health.py", label="← Revenue Health")
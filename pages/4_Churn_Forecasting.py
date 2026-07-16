# Predict future churn using ML models (Random Forest, XGBoost, LightGBM)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Churn Forecasting", layout="wide")

if 'df' not in st.session_state:
    st.error("Please go to the main page first")
    st.stop()

st.title("Churn Forecasting")
st.markdown("Predict which accounts will experience revenue decline using machine learning models")
df = st.session_state.df.copy()

# Model Selection and Configuration
st.sidebar.markdown("### Model Configuration")

model_choice = st.sidebar.selectbox(
    "Choose Model:",
    ["Random Forest (Baseline)", "XGBoost (Optimized)", "LightGBM (Best Performance)"],
    help="LightGBM achieves AUC 0.969 - Best for catching churn cases"
)

# Churn definition
st.sidebar.markdown("### Churn Definition")
churn_type = st.sidebar.radio(
    "Churn Definition:",
    ["Revenue Decline (Any)", "Strict Churn (50%+ Drop)"],
    help="Revenue Decline: next_month_mrr < current_mrr\nStrict Churn: next_month_mrr < current_mrr × 0.5"
)

# Risk thresholds (from your code)
st.sidebar.markdown("### 🎯 Risk Thresholds")
high_threshold = st.sidebar.slider("High Risk Threshold", 0.5, 0.95, 0.7, 0.05,
                                    help="Accounts above this = High Risk")
medium_threshold = st.sidebar.slider("Medium Risk Threshold", 0.2, 0.6, 0.4, 0.05,
                                      help="Accounts above this = Medium Risk")

# Business impact settings
st.sidebar.markdown("### Business Impact Settings")
save_rate = st.sidebar.slider("Recovery Rate (% of revenue saved)", 10, 50, 25, 5) / 100
intervention_budget = st.sidebar.number_input("Intervention Budget ($)", 10000, 2000000, 500000, 50000)

# Data Preparation and Feature Selection
st.markdown("### Data Preparation")

# Create churn flag based on definition
if churn_type == "Revenue Decline (Any)":
    df["churn_flag"] = (df["next_month_mrr"] < df["current_mrr"]).astype(int)
    st.info("→ Churn = Revenue decline next month (includes both full churn and contraction)")
else:
    df["churn_strict"] = (df["next_month_mrr"] < df["current_mrr"] * 0.5).astype(int)
    df["churn_flag"] = df["churn_strict"]
    st.info("→ Strict Churn = Revenue drops by 50%+ next month")

# Show churn distribution
col1, col2 = st.columns(2)
with col1:
    churn_dist = df["churn_flag"].value_counts(normalize=True)
    st.metric("Churn Rate", f"{churn_dist.get(1, 0):.1%}")
    st.caption(f"Non-Churn: {churn_dist.get(0, 0):.1%} | Churn: {churn_dist.get(1, 0):.1%}")
with col2:
    st.metric("Total Accounts", f"{len(df):,}")
    st.metric("Features Used", "13 (categorical + numeric)")

# Features (exactly from your code)
features = [
    "company_size", "industry", "contract_type", "regime_state",
    "discount_pct", "active_users", "usage_growth",
    "feature_adoption_rate", "error_rate", "tickets_count",
    "ticket_growth", "payment_delay_flag", "current_mrr"
]

cat_cols = ["company_size", "industry", "contract_type", "regime_state"]
num_cols = [col for col in features if col not in cat_cols]

# Prepare data
X = df[features]
y = df["churn_flag"]

# Train Model and Predict Churn
if st.button("Train Model & Predict Churn", type="primary", use_container_width=True):
    with st.spinner("Training model and making predictions..."):
        
        # Preprocessor (from your code)
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Select and train model
        if model_choice == "Random Forest (Baseline)":
            model = Pipeline([
                ("prep", preprocessor),
                ("clf", RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                ))
            ])
            model_type = "Random Forest"
            
        elif model_choice == "XGBoost (Optimized)":
            scale = (y_train == 0).sum() / (y_train == 1).sum()
            model = Pipeline([
                ("prep", preprocessor),
                ("clf", XGBClassifier(
                    scale_pos_weight=scale,
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=42
                ))
            ])
            model_type = "XGBoost"
            
        else:  # LightGBM
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model = Pipeline([
                ("prep", preprocessor),
                ("clf", LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, num_leaves=31,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42, verbose=-1
                ))
            ])
            model_type = "LightGBM"
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict on full dataset (from your code: model.predict_proba(X)[:,1])
        df["churn_probability"] = model.predict_proba(X)[:, 1]
        
        # Risk segmentation (from your code)
        def risk_segment(p):
            if p > high_threshold:
                return "High Risk"
            elif p > medium_threshold:
                return "Medium Risk"
            else:
                return "Low Risk"
        
        df["risk_segment"] = df["churn_probability"].apply(risk_segment)
        
        # Revenue at risk (from your code)
        df["revenue_at_risk"] = df["current_mrr"] * df["churn_probability"]
        
        # Store in session
        st.session_state.forecast_df = df
        st.session_state.model = model
        st.session_state.model_type = model_type
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_train = y_train
        
        st.success(f"{model_type} model trained and predictions complete!")

# Display Results if available
if 'forecast_df' in st.session_state:
    df = st.session_state.forecast_df
    
    st.markdown("---")
    
    # Model Performance Metrics
    st.subheader("Model Performance")
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Classification report
        st.markdown("**Classification Report**")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.style.format("{:.3f}").background_gradient(cmap='Blues', axis=0),
            use_container_width=True
        )
    
    with col2:
        # Confusion Matrix (from your code)
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=['No Churn (0)', 'Churn (1)'],
            y=['No Churn (0)', 'Churn (1)'],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix - {st.session_state.model_type}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.markdown("**ROC Curve & AUC Score**")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={roc_auc:.4f})',
                                line=dict(color='darkorange', width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random',
                                line=dict(dash='dash', color='gray')))
        fig.update_layout(title=f"ROC Curve - AUC: {roc_auc:.4f}",
                         xaxis_title="False Positive Rate",
                         yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("AUC-ROC Score", f"{roc_auc:.4f}",
                 delta="Strong" if roc_auc > 0.9 else "Moderate" if roc_auc > 0.7 else "Weak")
        st.caption("→ AUC measures model ability to separate churn vs non-churn")
        st.caption("→ Closer to 1 = strong model")
        st.caption("→ Near 0.5 = random guessing")
    
    # Risk Segmentation & Revenue at Risk
    st.markdown("---")
    st.subheader("Risk Segmentation Results")
    
    st.info("→ Customers grouped into actionable segments")
    st.warning("→ High-risk accounts need immediate attention")
    
    # Risk distribution
    col1, col2, col3 = st.columns(3)
    risk_counts = df["risk_segment"].value_counts()
    
    with col1:
        high_count = risk_counts.get("High Risk", 0)
        st.metric("High Risk", f"{high_count:,}",
                 delta=f"{high_count/len(df)*100:.1f}%")
    with col2:
        med_count = risk_counts.get("Medium Risk", 0)
        st.metric("Medium Risk", f"{med_count:,}",
                 delta=f"{med_count/len(df)*100:.1f}%")
    with col3:
        low_count = risk_counts.get("Low Risk", 0)
        st.metric("Low Risk", f"{low_count:,}",
                 delta=f"{low_count/len(df)*100:.1f}%")
    
    # Risk pie chart
    fig = px.pie(
        values=risk_counts.values, names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            'High Risk': '#dc3545',
            'Medium Risk': '#ffc107',
            'Low Risk': '#28a745'
        },
        title="Risk Segment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue at Risk
    st.markdown("---")
    st.subheader("Revenue at Risk Analysis")
    
    # Summary by risk segment (from your code)
    summary = df.groupby("risk_segment").agg({
        "churn_probability": "mean",
        "current_mrr": "sum",
        "revenue_at_risk": "sum"
    }).sort_values(by="revenue_at_risk", ascending=False)
    
    total_risk = summary["revenue_at_risk"].sum()
    summary["risk_share"] = summary["revenue_at_risk"] / total_risk
    
    st.dataframe(
        summary.style.format({
            'churn_probability': '{:.1%}',
            'current_mrr': '${:,.0f}',
            'revenue_at_risk': '${:,.0f}',
            'risk_share': '{:.1%}'
        }).background_gradient(subset=['revenue_at_risk'], cmap='Reds'),
        use_container_width=True
    )
    
    # Revenue at risk metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue at Risk", f"${total_risk:,.0f}")
    with col2:
        recoverable = total_risk * save_rate
        st.metric("Recoverable Revenue", f"${recoverable:,.0f}",
                 delta=f"At {save_rate:.0%} recovery rate")
    with col3:
        st.metric("Accounts at Risk", f"{high_count:,}")
    
    # High Risk Accounts Table
    st.markdown("---")
    st.subheader("🚨 High Risk Accounts - Priority List")
    
    st.info("→ List of accounts likely to churn")
    st.info("→ Prioritize high-value + high-risk customers")
    st.warning("→ This is where business value is created")
    
    high_risk_df = df[df["risk_segment"] == "High Risk"].nlargest(
        20, "revenue_at_risk"
    )[[
        "account_id", "company_size", "industry",
        "current_mrr", "churn_probability", "revenue_at_risk"
    ]]
    
    st.dataframe(
        high_risk_df.style
        .background_gradient(subset=['churn_probability', 'revenue_at_risk'], cmap='Reds')
        .format({
            'current_mrr': '${:,.0f}',
            'churn_probability': '{:.1%}',
            'revenue_at_risk': '${:,.0f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Intervention ROI Calculator
    st.markdown("---")
    st.subheader("Intervention ROI Calculator")
    
    # Intervention cost model (from your code)
    def intervention_cost(row):
        if row["churn_probability"] > 0.8:
            return 500   # expensive sales call
        elif row["churn_probability"] > 0.6:
            return 100   # discount / incentive
        else:
            return 20    # email / light touch
    
    high_risk = df[df["risk_segment"] == "High Risk"].copy()
    
    if len(high_risk) > 0:
        high_risk["intervention_cost"] = high_risk.apply(intervention_cost, axis=1)
        high_risk["recoverable_revenue"] = high_risk["revenue_at_risk"] * save_rate
        high_risk["net_value"] = high_risk["recoverable_revenue"] - high_risk["intervention_cost"]
        
        # Filter profitable targets (from your code)
        profitable_targets = high_risk[
            (high_risk["net_value"] > 0)
        ].sort_values("net_value", ascending=False)
        
        # Apply budget constraint
        final_targets = profitable_targets[
            profitable_targets["intervention_cost"].cumsum() <= intervention_budget
        ]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Selected Customers", len(final_targets))
        with col2:
            st.metric("Total Cost", f"${final_targets['intervention_cost'].sum():,.0f}")
        with col3:
            st.metric("Expected Profit", f"${final_targets['net_value'].sum():,.0f}")
        with col4:
            roi = final_targets["net_value"].sum() / final_targets["intervention_cost"].sum() if final_targets["intervention_cost"].sum() > 0 else 0
            st.metric("ROI", f"{roi:.2f}x")
        
        if roi > 3:
            st.success(f"Excellent ROI: {roi:.2f}x - Proceed with intervention plan")
        elif roi > 1:
            st.info(f"ℹPositive ROI: {roi:.2f}x - Consider targeted intervention")
        else:
            st.warning(f"Low ROI: {roi:.2f}x - Review strategy")
    
    # Risk by Segment (Strategic Insight)
    st.markdown("---")
    st.subheader("Strategic Risk Insights")
    
    st.info("→ Different segments show different churn behavior")
    st.warning("→ Target retention strategies per segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn risk by company size
        risk_by_size = df.groupby("company_size")["churn_probability"].mean().sort_values(ascending=False)
        fig = px.bar(
            x=risk_by_size.index, y=risk_by_size.values,
            title="Churn Risk by Company Size",
            color=risk_by_size.values, color_continuous_scale='Reds',
            labels={'x': 'Company Size', 'y': 'Avg Churn Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue at risk by industry
        industry_risk = df.groupby("industry")["revenue_at_risk"].sum().nlargest(10)
        fig = px.bar(
            x=industry_risk.index, y=industry_risk.values,
            title="Top 10 Industries - Revenue at Risk",
            color=industry_risk.values, color_continuous_scale='Reds',
            labels={'x': 'Industry', 'y': 'Revenue at Risk ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export buttons for full predictions and high-risk accounts
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Export Full Predictions (CSV)",
            df.to_csv(index=False),
            "churn_forecast_all_accounts.csv",
            "text/csv"
        )
    with col2:
        st.download_button(
            "Export High Risk Accounts (CSV)",
            high_risk_df.to_csv(index=False),
            "churn_forecast_high_risk.csv",
            "text/csv"
        )

else:
    st.info("Click 'Train Model & Predict Churn' to run the prediction pipeline")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/3_Risk_Patterns.py", label="← Risk Patterns", icon="🔍")
with col2:
    st.page_link("pages/5_Account_Risk_Profiler.py", label="→ Account Risk Profiler", icon="🎯")
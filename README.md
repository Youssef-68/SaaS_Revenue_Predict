# SaaS Revenue Intelligence Platform

An end-to-end "Data Science & Analytics System" that transforms SaaS data into: revenue forecasts, churn predictions, and actionable business insights.
Built as a multi-page "Streamlit Application", this project needs to target real-world SaaS analytics platform used by product, finance, and growth teams.

-----------------------------------------------------------

## Key Capabilities

### Revenue Forecasting
Predict Next Month MRR using customer usage, behavior, and financial signals.

### Churn Risk Prediction
Identify high-risk accounts using behavioral and operational indicators.

### Business Drivers Analysis
Understand what drives:
- Revenue growth
- Revenue contraction
- Customer engagement

### Account Risk Profiling
Analyze individual accounts and classify them into:
- Safe
- High Risk - Low Value 
- High Risk - High Value 

### Interactive Dashboards
Includes multiple analytical views:
- Revenue Health
- Business Drivers
- Risk Patterns
- Churn Forecasting
- Account Risk Profiler

-----------------------------------------------------------

## Dataset Overview

- Identifiers: account_id, month
- Semi-static: company_size, industry, contract_type, regime_state
- Behavior: active_users, usage_growth, feature_adoption_rate
- Issues: error_rate, tickets_count, ticket_growth
- Revenue: discount_pct, current_mrr, next_month_mrr, payment_delay_flag


### Feature Categories

- Usage Metrics: active_users, usage_growth  
- Engagement Signals: feature_adoption_rate  
- Operational Health: error_rate, tickets_count  
- Financial Data: current_mrr, discounts  
- Behavioral Indicators: payment delays  

-----------------------------------------------------------

## Modeling Approach

- Supervised regression model to predict (next_month_mrr)
- Feature-driven analysis to identify (growth & risk drivers)
- Business logic layer to classify (account risk segments)

-----------------------------------------------------------

## Tech Stack

- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn & LightGBM
- Matplotlib, Seaborn & Plotly

-----------------------------------------------------------

## Deploy

Notebooks:
01_cleaning.ipynb: Data preprocessing
02_analysis.ipynb: Exploratory data analysis
03_prediction.ipynb: Model experimentation
04_modeling.ipynb: Final modeling pipeline

Future Improvements:
Save trained model (model.pkl) instead of training on runtime
Add real-time prediction API
Enhance feature engineering
Add cohort & retention analysis

Author:
Youssef Wael - Data Science & Analytics

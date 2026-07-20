# SaaS Revenue Intelligence Platform
This project transforms raw product usage, support, and billing data into actionable insights, helping companies identify at-risk accounts and maximize retention ROI.

---

## Key Features

- Churn Prediction using ML models (Random Forest, XGBoost, LightGBM)
- Revenue at Risk Estimation
- Customer Targeting Strategy: based on profitability
- Interactive Streamlit Dashboard
- Scenario Simulation: (budget & intervention planning)
- Account Risk Profiler: for real-time predictions

---

## Problem Definition

A customer is considered **churned** if: next_month_mrr < current_mrr

---

## The goal is to:
- Predict churn probability
- Estimate financial impact
- Optimize which customers to retain

---

## Features Used

- Company Profile: "company_size", "industry", "contract_type"
- Usage Metrics: "active_users", "usage_growth", `feature_adoption_rate"
- Support Signals: "tickets_count", "ticket_growth", "error_rate"
- Financial Signals: "current_mrr", "discount_pct", "payment_delay_flag"

---

## Dashboard Modules

### 1. Revenue Health
- Revenue at risk
- Recoverable revenue
- ROI metrics

### 2. Business Drivers
- Key factors affecting churn
- Feature importance insights

### 3. Risk Patterns
- Customer segmentation
- High-risk behavior patterns

### 4. Churn Forecasting
- Model predictions
- Risk distribution

### 5. Account Risk Profiler
- Input new customer data
- Get instant churn prediction

---

## Business Impact

This system enables:
* Reduction in churn rate
* Better allocation of retention budget
* Targeting high-value customers
* Data-driven decision making

---

## Future Improvements

* Real-time data integration
* Automated campaign triggering
* Deep learning models
* Customer lifetime value (CLV) modeling

---

## Author

Youssef Wael
Data Scientist & Analytics

# Analysis Utilities Module = Business-focused data analysis and visualization

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats
import streamlit as st

# Business-focused data analysis utilities
class DataAnalyzer:
    
    
    def __init__(self, df):
        self.df = df
        
        # Business analysis columns (limited set)
        self.business_columns = [
            "company_size", "industry", "discount_pct", "regime_state",
            "usage_growth", "ticket_growth", "payment_delay_flag"
        ]
        
        self.categorical_cols = [
            "company_size", "industry", "regime_state", "payment_delay_flag"
        ]
        
        self.numeric_cols = [
            "discount_pct", "usage_growth", "ticket_growth"
        ]
        
        
    # Get business column metadata
    def get_column_info(self):
        
        info = []
        for col in self.business_columns:
            col_info = {
                "column": col,
                "dtype": str(self.df[col].dtype),
                "missing": self.df[col].isnull().sum(),
                "missing_pct": (self.df[col].isnull().sum() / len(self.df)) * 100,
                "unique": self.df[col].nunique(),
                "type": "Categorical" if col in self.categorical_cols else "Numeric"
            }
            info.append(col_info)
        return pd.DataFrame(info)
    
    
    # Analyze distribution of business metrics
    def analyze_business_distribution(self, col):
        
        if col in self.categorical_cols:
            return self._categorical_analysis(col)
        else:
            return self._numeric_analysis(col)
        
        
    # Categorical business metric analysis
    def _categorical_analysis(self, col):
        
        counts = self.df[col].value_counts()
        percentages = self.df[col].value_counts(normalize=True) * 100
        
        # Revenue impact by category
        revenue_by_cat = self.df.groupby(col)['current_mrr'].agg(['mean', 'sum']).round(2)
        
        # Churn impact by category
        self.df['churn_flag'] = (self.df['next_month_mrr'] < self.df['current_mrr']).astype(int)
        churn_by_cat = self.df.groupby(col)['churn_flag'].mean() * 100
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"{col} Distribution",
                f"Revenue by {col}",
                f"Churn Rate by {col}",
                f"{col} Composition"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )
        
        # Distribution
        fig.add_trace(
            go.Bar(x=counts.index.astype(str), y=counts.values, name="Count",
                   marker_color='steelblue'),
            row=1, col=1
        )
        
        # Revenue impact
        fig.add_trace(
            go.Bar(x=revenue_by_cat.index.astype(str), y=revenue_by_cat['mean'].values, 
                   name="Avg Revenue", marker_color='lightseagreen'),
            row=1, col=2
        )
        
        # Churn rate
        fig.add_trace(
            go.Bar(x=churn_by_cat.index.astype(str), y=churn_by_cat.values,
                   name="Churn Rate %", marker_color='coral'),
            row=2, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=counts.index.astype(str), values=percentages.values,
                   name="Share", hole=0.4),
            row=2, col=2
        )
        
        fig.update_layout(height=700, showlegend=False)
        
        # Business insights
        insights = []
        
        # Revenue insight
        top_rev_cat = revenue_by_cat['mean'].idxmax()
        low_rev_cat = revenue_by_cat['mean'].idxmin()
        insights.append(f"**{top_rev_cat}** has highest avg MRR (${revenue_by_cat['mean'].max():,.0f})")
        insights.append(f"**{low_rev_cat}** has lowest avg MRR (${revenue_by_cat['mean'].min():,.0f})")
        
        # Churn insight
        top_churn_cat = churn_by_cat.idxmax()
        insights.append(f"**{top_churn_cat}** has highest churn rate ({churn_by_cat.max():.1f}%)")
        
        # Dominance insight
        dominant = counts.index[0]
        dominant_pct = percentages.iloc[0]
        if dominant_pct > 50:
            insights.append(f"**{dominant}** dominates with {dominant_pct:.1f}% share")
        
        return fig, insights
    
    
    # Numeric business metric analysis
    def _numeric_analysis(self, col):
        
        data = self.df[col].dropna()
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"{col} Distribution",
                f"{col} vs Revenue",
                f"{col} by Company Size",
                f"{col} Trend Analysis"
            )
        )
        
        # Distribution
        fig.add_trace(
            go.Histogram(x=data, nbinsx=40, name="Distribution",
                        marker_color='steelblue'),
            row=1, col=1
        )
        
        # vs Revenue scatter
        sample = self.df.sample(min(5000, len(self.df)))
        fig.add_trace(
            go.Scatter(x=sample[col], y=sample['current_mrr'], 
                      mode='markers', name="vs Revenue",
                      marker=dict(color='lightseagreen', opacity=0.6)),
            row=1, col=2
        )
        
        # By company size
        fig.add_trace(
            go.Box(x=self.df['company_size'], y=self.df[col],
                   name="By Company Size", marker_color='coral'),
            row=2, col=1
        )
        
        # Percentile curve
        sorted_vals = np.sort(data)
        percentiles = np.linspace(0, 100, len(sorted_vals))
        fig.add_trace(
            go.Scatter(x=percentiles, y=sorted_vals, mode='lines',
                      name="Percentile Curve", line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        fig.update_layout(height=700, showlegend=False)
        
        # Business insights
        insights = []
        mean_val = data.mean()
        median_val = data.median()
        
        insights.append(f"Average {col}: **{mean_val:.3f}** (Median: {median_val:.3f})")
        
        # Revenue correlation
        corr = self.df[col].corr(self.df['current_mrr'])
        if abs(corr) > 0.3:
            direction = "positive" if corr > 0 else "negative"
            insights.append(f"{direction.capitalize()} correlation with revenue (r={corr:.3f})")
        
        # Outliers
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        if len(outliers) > 0:
            insights.append(f"{len(outliers)} outliers detected ({len(outliers)/len(data)*100:.1f}%)")
        
        return fig, insights
    
    
    # Analyze relationship between two business metrics
    def analyze_business_relationship(self, col1, col2):        
        
        type1 = "Categorical" if col1 in self.categorical_cols else "Numeric"
        type2 = "Categorical" if col2 in self.categorical_cols else "Numeric"
        
        if type1 == "Numeric" and type2 == "Numeric":
            return self._numeric_relationship(col1, col2)
        elif type1 == "Categorical" and type2 == "Numeric":
            return self._cat_num_relationship(col1, col2)
        elif type1 == "Numeric" and type2 == "Categorical":
            return self._cat_num_relationship(col2, col1, swap=True)
        else:
            return self._cat_cat_relationship(col1, col2)
    
    
    # Numeric vs Numeric relationship
    def _numeric_relationship(self, col1, col2):
        
        sample = self.df.sample(min(5000, len(self.df)))
        
        fig = px.scatter(
            sample, x=col1, y=col2,
            color='company_size',
            trendline="ols",
            title=f"{col1} vs {col2}",
            opacity=0.7
        )
        
        corr = self.df[col1].corr(self.df[col2])
        
        insights = []
        if abs(corr) > 0.5:
            insights.append(f"Strong relationship detected (r={corr:.3f})")
        elif abs(corr) > 0.3:
            insights.append(f"Moderate relationship detected (r={corr:.3f})")
        else:
            insights.append(f"Weak relationship (r={corr:.3f})")
        
        return fig, insights
    
    
    # Categorical vs Numeric relationship
    def _cat_num_relationship(self, cat_col, num_col, swap=False):
        
        fig = px.box(
            self.df, x=cat_col, y=num_col,
            color=cat_col,
            title=f"{num_col} by {cat_col}"
        )
        
        # Business impact
        group_stats = self.df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
        
        insights = []
        insights.append(f"Highest: **{group_stats.index[0]}** ({group_stats.values[0]:.3f})")
        insights.append(f"Lowest: **{group_stats.index[-1]}** ({group_stats.values[-1]:.3f})")
        
        return fig, insights
    
    
    # Categorical vs Categorical relationship
    def _cat_cat_relationship(self, col1, col2):
        
        cross_tab = pd.crosstab(self.df[col1], self.df[col2])
        
        fig = px.imshow(
            cross_tab, text_auto=True,
            aspect="auto",
            title=f"{col1} vs {col2}",
            color_continuous_scale="Blues"
        )
        
        insights = []
        insights.append(f"Cross-tabulation of {col1} and {col2}")
        
        return fig, insights
    
    
    # Business metrics correlation matrix
    def business_correlation_matrix(self):
        
        numeric_df = self.df[self.numeric_cols + ['current_mrr']]
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Business Metrics Correlation Matrix",
            zmin=-1, zmax=1
        )
        
        # Business insights from correlations
        insights = []
        
        for col in self.numeric_cols:
            corr_with_revenue = corr_matrix.loc[col, 'current_mrr']
            if abs(corr_with_revenue) > 0.3:
                direction = "positively" if corr_with_revenue > 0 else "negatively"
                insights.append(f"**{col}** is {direction} correlated with revenue (r={corr_with_revenue:.3f})")
        
        return fig, insights
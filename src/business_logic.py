# Revenue at risk calculations, intervention strategies, and ROI analysis

import pandas as pd
import numpy as np


# Analyzes business impact of churn predictions
class BusinessImpactAnalyzer:
    
    
    def __init__(self, df, churn_prob_col="churn_probability", mrr_col="current_mrr"):
        self.df = df.copy()
        self.churn_prob_col = churn_prob_col
        self.mrr_col = mrr_col
        
        
    # Segment customers by risk level
    def risk_segmentation(self, high_threshold=0.7, medium_threshold=0.4):
        
        if self.churn_prob_col not in self.df.columns:
            self.df[self.churn_prob_col] = 0
        
        self.df["risk_segment"] = self.df[self.churn_prob_col].apply(
            lambda p: "High Risk" if p > high_threshold 
            else ("Medium Risk" if p > medium_threshold else "Low Risk")
        )
        
        return self.df
    
    # Calculate revenue at risk metrics
    def calculate_revenue_at_risk(self):
        
        if "risk_segment" not in self.df.columns:
            self.risk_segmentation()
        
        self.df["revenue_at_risk"] = (
            self.df[self.mrr_col] * self.df[self.churn_prob_col]
        )
        
        summary = self.df.groupby("risk_segment").agg({
            self.churn_prob_col: "mean",
            self.mrr_col: ["sum", "count"],
            "revenue_at_risk": "sum"
        }).round(2)
        
        summary.columns = ["Avg_Churn_Prob", "Total_MRR", "Account_Count", "Revenue_at_Risk"]
        summary["Risk_Share"] = (
            summary["Revenue_at_Risk"] / summary["Revenue_at_Risk"].sum()
        )
        
        return summary
    
    
    # Calculate ROI for intervention strategies
    def calculate_intervention_roi(self, budget=500000, save_rate=0.25):
        
        if "revenue_at_risk" not in self.df.columns:
            self.calculate_revenue_at_risk()
        
        high_risk = self.df[self.df["risk_segment"] == "High Risk"].copy()
        
        if len(high_risk) == 0:
            return None, {"error": "No high risk accounts found"}
        
        # Cost model
        def intervention_cost(row):
            if row[self.churn_prob_col] > 0.8:
                return 500
            elif row[self.churn_prob_col] > 0.6:
                return 100
            else:
                return 20
        
        high_risk["intervention_cost"] = high_risk.apply(intervention_cost, axis=1)
        high_risk["recoverable_revenue"] = high_risk["revenue_at_risk"] * save_rate
        high_risk["net_value"] = high_risk["recoverable_revenue"] - high_risk["intervention_cost"]
        
        # Filter profitable targets
        profitable = high_risk[high_risk["net_value"] > 0].sort_values(
            "net_value", ascending=False
        )
        
        # Select within budget
        profitable["cumulative_cost"] = profitable["intervention_cost"].cumsum()
        targets = profitable[profitable["cumulative_cost"] <= budget]
        
        if len(targets) > 0:
            total_cost = targets["intervention_cost"].sum()
            total_profit = targets["net_value"].sum()
            roi = total_profit / total_cost if total_cost > 0 else 0
            
            metrics = {
                "selected_customers": len(targets),
                "total_cost": total_cost,
                "expected_profit": total_profit,
                "roi": roi,
                "total_revenue_at_risk": high_risk["revenue_at_risk"].sum(),
                "total_recoverable": high_risk["recoverable_revenue"].sum()
            }
            
            return targets, metrics
        else:
            return None, {"error": "No profitable targets found"}
        
        
        
    # Get top priority accounts for intervention
    def get_priority_accounts(self, top_n=20):
        
        if "revenue_at_risk" not in self.df.columns:
            self.calculate_revenue_at_risk()
        
        priority = (
            self.df[self.df["risk_segment"] == "High Risk"]
            .nlargest(top_n, "revenue_at_risk")
        )
        
        display_cols = [
            "account_id", "company_size", "industry", 
            self.mrr_col, self.churn_prob_col, 
            "revenue_at_risk", "risk_segment"
        ]
        
        available = [c for c in display_cols if c in priority.columns]        
        return priority[available]
 
 
    # Detailed analysis by business segments    
    def segment_analysis(self):
       
        analysis = {}       
        segment_cols = ["company_size", "industry", "contract_type"]
        
        for col in segment_cols:
            if col in self.df.columns:
                analysis[col] = self.df.groupby(col).agg({
                    self.churn_prob_col: "mean",
                    self.mrr_col: ["sum", "mean", "count"],
                    "revenue_at_risk": "sum" if "revenue_at_risk" in self.df.columns else None
                }).round(2)
        
        return analysis
    
    
    # Generate executive summary
    def generate_summary(self):
        
        if "revenue_at_risk" not in self.df.columns:
            self.calculate_revenue_at_risk()
        
        total_mrr = self.df[self.mrr_col].sum()
        total_risk = self.df["revenue_at_risk"].sum()       
        high_risk = self.df[self.df["risk_segment"] == "High Risk"]
        
        return {
            "total_accounts": len(self.df),
            "total_mrr": total_mrr,
            "total_revenue_at_risk": total_risk,
            "risk_percentage": (total_risk / total_mrr * 100) if total_mrr > 0 else 0,
            "high_risk_accounts": len(high_risk),
            "high_risk_mrr": high_risk[self.mrr_col].sum(),
            "avg_churn_probability": self.df[self.churn_prob_col].mean(),
            "risk_distribution": self.df["risk_segment"].value_counts().to_dict()
        }



# Optimizes retention strategies based on risk segments
class RetentionStrategyOptimizer:    
    
    
    # Get recommended retention strategy
    @staticmethod
    def get_strategy(row):
        
        prob = row.get("churn_probability", 0)
        mrr = row.get("current_mrr", 0)
        segment = row.get("company_size", "Unknown")
        
        if prob > 0.8:
            if mrr > 10000:
                return {
                    "strategy": "Executive Outreach",
                    "action": "VP call + custom solution within 24h",
                    "budget": "$1,000-$5,000",
                    "priority": "Critical"
                }
            else:
                return {
                    "strategy": "Urgent Intervention",
                    "action": "Account manager call + service credits",
                    "budget": "$500-$1,000",
                    "priority": "Critical"
                }
        elif prob > 0.6:
            return {
                "strategy": "Proactive Engagement",
                "action": "QBR + feature adoption plan",
                "budget": "$100-$500",
                "priority": "High"
            }
        elif prob > 0.4:
            return {
                "strategy": "Value Reinforcement",
                "action": "Success stories + training",
                "budget": "$20-$100",
                "priority": "Medium"
            }
        else:
            return {
                "strategy": "Maintenance",
                "action": "Newsletter + product updates",
                "budget": "Standard",
                "priority": "Low"
            }
            
            
    # Generate action plan for all accounts
    @staticmethod
    
    def generate_action_plan(df):
        
        if "churn_probability" not in df.columns:
            df["churn_probability"] = 0
        
        plans = []
        for _, row in df.iterrows():
            strategy = RetentionStrategyOptimizer.get_strategy(row)
            plan = {
                "account_id": row.get("account_id", "N/A"),
                "churn_probability": row["churn_probability"],
                "strategy": strategy["strategy"],
                "action": strategy["action"],
                "priority": strategy["priority"]
            }
            plans.append(plan)
        
        return pd.DataFrame(plans)


if __name__ == "__main__":
    # Test business logic
    df = pd.read_csv("data/cleaned_data.csv")
    df["churn_probability"] = np.random.beta(2, 5, len(df))
    
    analyzer = BusinessImpactAnalyzer(df)
    analyzer.risk_segmentation()
    
    summary = analyzer.calculate_revenue_at_risk()
    print("\nRevenue at Risk Summary:")
    print(summary)
    
    targets, metrics = analyzer.calculate_intervention_roi(budget=500000)
    if targets is not None:
        print(f"\nIntervention ROI: {metrics['roi']:.2f}x")
        print(f"Target accounts: {len(targets)}")
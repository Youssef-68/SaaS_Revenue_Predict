# Handles making predictions with trained models

import pandas as pd
import numpy as np
import joblib
import os



# Handles churn predictions using trained model
class ChurnPredictor:
    
    
    def __init__(self, model_path="model/churn_model.pkl"):
        self.model_path = model_path
        self.model_data = None
        self.load_model()
    
    
    # Load the trained model
    def load_model(self):
        
        if os.path.exists(self.model_path):
            self.model_data = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Model not found at {self.model_path}!")
            print("Please train the model first: python src/train.py")
            self.model_data = None
            
            
    # Check if model is loaded
    def is_model_loaded(self):
        
        return self.model_data is not None
    
    
    # Predict churn probability for accounts
    def predict_churn_probability(self, df):
        
        if not self.is_model_loaded():
            return None
        
        models = self.model_data["models"]
        features = self.model_data["metadata"]["features"]        
        X = df[features]
        probabilities = models["xgb_model"].predict_proba(X)[:, 1]
        
        return probabilities
    
    
    # Make batch predictions with risk segmentation
    def predict_batch(self, df):
        
        if not self.is_model_loaded():
            df["churn_probability"] = 0.0
            df["risk_segment"] = "Unknown"
            df["churn_prediction"] = 0
            return df
        
        probabilities = self.predict_churn_probability(df)       
        results = df.copy()
        results["churn_probability"] = probabilities
        
        # Binary prediction
        results["churn_prediction"] = (probabilities > 0.5).astype(int)
        
        # Risk segmentation
        results["risk_segment"] = results["churn_probability"].apply(
            lambda p: "High Risk" if p > 0.7 
            else ("Medium Risk" if p > 0.4 else "Low Risk")
        )
        
        # Revenue at risk
        results["revenue_at_risk"] = (
            results["current_mrr"] * results["churn_probability"]
        )
        
        return results

    
    # Predict for a single account
    def predict_single(self, account_data):
        
        if not self.is_model_loaded():
            return {
                "churn_probability": 0.0,
                "risk_level": "Unknown",
                "prediction": "Unknown"
            }
        
        df = pd.DataFrame([account_data])        
        features = self.model_data["metadata"]["features"]
        X = df[features]        
        prob = self.model_data["models"]["xgb_model"].predict_proba(X)[0, 1]
        
        result = {
            "churn_probability": float(prob),
            "risk_level": "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low"),
            "prediction": "Churn" if prob > 0.5 else "Retain"
        }
        
        return result
    
    
    # Predict next month MRR
    def predict_mrr(self, df):
        
        if not self.is_model_loaded():
            return None
        
        mrr_features = self.model_data["metadata"]["mrr_features"]
        X = df[mrr_features]
        
        predictions = self.model_data["models"]["mrr_model"].predict(X)
        return predictions


if __name__ == "__main__":
   
    # Example usage
    predictor = ChurnPredictor()
    
    if predictor.is_model_loaded():
        # Load test data
        df = pd.read_csv("data/cleaned_data.csv")
        
        # Make predictions
        results = predictor.predict_batch(df.head(100))
        print("\nSample Predictions:")
        print(results[["account_id", "current_mrr", "churn_probability", "risk_segment"]].head(10))
        
        # Single prediction
        sample = df.iloc[0].to_dict()
        result = predictor.predict_single(sample)
        print(f"\nSingle Prediction: {result}")
    else:
        print("Please train the model first.")
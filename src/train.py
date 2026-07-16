# Trains and saves churn prediction models with preprocessing

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import classification_report, roc_auc_score

import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# Handles training of all churn prediction models
class ChurnModelTrainer:
      
    def __init__(self, data_path="data/cleaned_data.csv"):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.encoders = {}
    
    
    # Load and prepare the dataset    
    def load_data(self):
                
        print("="*60)
        print("Loading Data")
        print("="*60)        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df):,} records")
        
        # Create basic features
        self.df["churn_flag"] = (
            self.df["next_month_mrr"] < self.df["current_mrr"]
        ).astype(int)
        
        # Sort for time-based features
        self.df = self.df.sort_values(["account_id", "month"])
        
        # Create advanced features
        self._create_advanced_features()
        
        # Create encoders for categorical variables
        cat_cols = ["company_size", "industry", "contract_type", "regime_state"]
        for col in cat_cols:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(self.df[col].astype(str))
        
        print(f"Churn rate: {self.df['churn_flag'].mean():.2%}")
        return self.df
    
    # Create lag and derived features
    def _create_advanced_features(self):
        
        print("\nCreating advanced features...")
        
        # Future MRR
        self.df["next_mrr"] = self.df.groupby("account_id")["current_mrr"].shift(-1)
        
        # Strict churn (50%+ revenue drop)
        self.df["churn_strict"] = (
            self.df["next_mrr"] < self.df["current_mrr"] * 0.5
        ).astype(int)
        
        # Lag features
        base_cols = ["current_mrr", "active_users", "usage_growth", 
                    "tickets_count", "error_rate"]
        
        for col in base_cols:
            self.df[f"{col}_lag1"] = self.df.groupby("account_id")[col].shift(1)
            self.df[f"{col}_lag2"] = self.df.groupby("account_id")[col].shift(2)
        
        # Growth and derived features
        self.df["mrr_growth"] = self.df["current_mrr"] - self.df["current_mrr_lag1"]
        self.df["usage_growth_change"] = self.df["usage_growth"] - self.df["usage_growth_lag1"]
        self.df["customer_age"] = (
            self.df["month"] - self.df.groupby("account_id")["month"].transform("min")
        )
        self.df["high_error"] = (self.df["error_rate"] > 0.1).astype(int)
        self.df["high_tickets"] = (self.df["tickets_count"] > 5).astype(int)
        self.df["support_pressure"] = self.df["tickets_count"] / (self.df["active_users"] + 1)        
        print(f"Created {len([c for c in self.df.columns if 'lag' in c or 'growth' in c or 'age' in c])} advanced features")
    
    
    # Train all models and return them
    def train_all_models(self):
        
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Basic features
        features = [
            "company_size", "industry", "contract_type", "regime_state",
            "discount_pct", "active_users", "usage_growth",
            "feature_adoption_rate", "error_rate", "tickets_count",
            "ticket_growth", "payment_delay_flag", "current_mrr"
        ]
        
        cat_cols = ["company_size", "industry", "contract_type", "regime_state"]
        num_cols = [col for col in features if col not in cat_cols]
        
        # Preprocessor
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])
        
        X = self.df[features]
        y = self.df["churn_flag"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train):,} | Test set: {len(X_test):,}")
        
        # 1. Random Forest
        print("\n[1/4] Training Random Forest...")
        rf_model = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ))
        ])
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
        print(f"RF AUC: {rf_auc:.4f}")
        
        # 2. XGBoost
        print("\n[2/4] Training XGBoost...")
        scale = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_model = Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                scale_pos_weight=scale,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42
            ))
        ])
        xgb_model.fit(X_train, y_train)
        xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
        print(f"XGBoost AUC: {xgb_auc:.4f}")
        
        # 3. LightGBM (advanced features)
        print("\n[3/4] Training LightGBM...")
        df_encoded = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
        adv_features = [
            col for col in df_encoded.columns
            if col not in [
                "account_id", "month", "churn_flag", "churn_strict",
                "next_mrr", "next_month_mrr"
            ]
        ]
        adv_features = [f for f in adv_features if not f.endswith('_lag2')]
        
        X_adv = df_encoded[adv_features].fillna(0)
        y_adv = self.df["churn_strict"]
        
        X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(
            X_adv, y_adv, test_size=0.2, random_state=42
        )
        
        scale_pos = (y_train_adv == 0).sum() / (y_train_adv == 1).sum()
        
        lgb_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_adv, y_train_adv)
        lgb_auc = roc_auc_score(y_test_adv, lgb_model.predict_proba(X_test_adv)[:, 1])
        print(f"LightGBM AUC: {lgb_auc:.4f}")
        
        # 4. MRR Prediction Model
        print("\n[4/4] Training MRR Predictor...")
        mrr_features = [
            "active_users", "usage_growth", "feature_adoption_rate",
            "error_rate", "tickets_count", "ticket_growth",
            "discount_pct", "current_mrr"
        ]
        
        X_mrr = self.df[mrr_features].dropna()
        y_mrr = self.df.loc[X_mrr.index, "next_month_mrr"].dropna()
        X_mrr = X_mrr.loc[y_mrr.index]
        
        X_train_mrr, X_test_mrr, y_train_mrr, y_test_mrr = train_test_split(
            X_mrr, y_mrr, test_size=0.2, random_state=42
        )
        
        mrr_model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        mrr_model.fit(X_train_mrr, y_train_mrr)
        mrr_score = mrr_model.score(X_test_mrr, y_test_mrr)
        print(f"MRR Model R²: {mrr_score:.4f}")
        
        # Store models
        self.models = {
            "rf_model": rf_model,
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "mrr_model": mrr_model,
            "preprocessor": preprocessor,
            "encoders": self.encoders,
            "features": features,
            "adv_features": adv_features,
            "mrr_features": mrr_features,
            "cat_cols": cat_cols,
            "num_cols": num_cols
        }
        
        return self.models
    
    def save_model(self, filepath="model/churn_model.pkl"):
        """Save all trained models and metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            "models": self.models,
            "metadata": {
                "features": self.models["features"],
                "adv_features": self.models["adv_features"],
                "mrr_features": self.models["mrr_features"],
                "cat_cols": self.models["cat_cols"],
                "num_cols": self.models["num_cols"],
            }
        }
        
        joblib.dump(save_data, filepath)
        print(f"\n✓ Model saved to {filepath}")
        print(f"  Size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    
    
    # Execute full training pipeline
    def run(self):        
        self.load_data()
        self.train_all_models()
        self.save_model()
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        return self.models


if __name__ == "__main__":
    trainer = ChurnModelTrainer()
    models = trainer.run()
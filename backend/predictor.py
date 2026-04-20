# backend/predictor.py
# ============================================================
# Loads artifacts once at startup, exposes predict() function
# Uses the IDENTICAL pipeline saved during training
# ============================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Add project root to path so utils/ is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.config import (
    PIPELINE_PATH,
    SHAP_EXPLAINER_PATH,
    FEATURE_ORDER_FILE,
    FEATURE_IMPORTANCE_PATH,
    MODEL_METADATA_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TOP_SHAP_FEATURES,
)


class ChurnPredictor:
    """
    Singleton-style predictor.
    Loads all artifacts once when the FastAPI app starts.
    """

    def __init__(self):
        print("🔄 Loading artifacts...")

        # Resolve paths relative to project root
        pipeline_path    = os.path.join(ROOT, PIPELINE_PATH)
        explainer_path   = os.path.join(ROOT, SHAP_EXPLAINER_PATH)
        feature_ord_path = os.path.join(ROOT, FEATURE_ORDER_FILE)
        feat_imp_path    = os.path.join(ROOT, FEATURE_IMPORTANCE_PATH)
        metadata_path    = os.path.join(ROOT, MODEL_METADATA_PATH)

        # Load pipeline (preprocessor + XGBoost model)
        self.pipeline = joblib.load(pipeline_path)
        print(f"   ✅ Pipeline loaded from: {pipeline_path}")

        # Load SHAP explainer
        self.explainer = joblib.load(explainer_path)
        print(f"   ✅ SHAP explainer loaded")

        # Load exact feature order (post-OHE)
        with open(feature_ord_path) as f:
            self.feature_order = json.load(f)
        print(f"   ✅ Feature order loaded: {len(self.feature_order)} features")

        # Load global feature importance
        with open(feat_imp_path) as f:
            self.feature_importance = json.load(f)

        # Load model metadata
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        print(f"   ✅ Model: {self.metadata['best_model']} | "
              f"Test AUC: {self.metadata['test_auc']:.4f}")
        print("✅ All artifacts loaded. Predictor ready.\n")

    def _build_input_df(self, customer_data: dict) -> pd.DataFrame:
        """
        Converts raw customer dict into a DataFrame with
        EXACTLY the same column names and order as training input.
        """
        all_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        row = {col: customer_data.get(col, None) for col in all_cols}
        return pd.DataFrame([row], columns=all_cols)

    def _get_risk_level(self, probability: float) -> str:
        if probability >= 0.70:
            return "High"
        elif probability >= 0.40:
            return "Medium"
        else:
            return "Low"

    def predict(self, customer_data: dict) -> dict:
        """
        Full prediction pipeline:
        1. Build input DataFrame
        2. Run through sklearn pipeline (impute → scale → OHE → XGBoost)
        3. Compute SHAP values for this customer
        4. Return prediction + probability + top SHAP features
        """

        # Step 1: Build input DataFrame
        input_df = self._build_input_df(customer_data)

        # Step 2: Predict
        churn_pred = int(self.pipeline.predict(input_df)[0])
        churn_prob = float(self.pipeline.predict_proba(input_df)[0][1])

        # Step 3: Transform input for SHAP (use fitted preprocessor)
        preprocessor  = self.pipeline.named_steps["preprocessor"]
        input_transformed = preprocessor.transform(input_df)
        input_shap_df  = pd.DataFrame(
            input_transformed,
            columns=self.feature_order
        )

        # Step 4: Compute SHAP values for this single customer
        shap_values = self.explainer.shap_values(input_shap_df)

        # Handle RF (list) vs XGBoost (2D array)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # class 1, first (only) row
        else:
            sv = shap_values[0]      # first (only) row

        # Step 5: Build top SHAP features list
        shap_dict = dict(zip(self.feature_order, sv.tolist()))
        top_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:TOP_SHAP_FEATURES]

        top_shap_features = [
            {
                "feature":    feat,
                "shap_value": round(val, 4),
                "direction":  "increases churn risk" if val > 0 else "decreases churn risk",
                "customer_value": str(input_shap_df[feat].values[0])   # scaled (for display)
            }
            for feat, val in top_features
        ]

        return {
            "churn_prediction":  churn_pred,
            "churn_label":       "Will Churn" if churn_pred == 1 else "Will Not Churn",
            "churn_probability": round(churn_prob, 4),
            "risk_level":        self._get_risk_level(churn_prob),
            "top_shap_features": top_shap_features,
        }


# Single instance — loaded once when module is imported
predictor = ChurnPredictor()
# backend/predictor.py
import os, sys, json
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.config import (
    PIPELINE_PATH, SHAP_EXPLAINER_PATH, FEATURE_ORDER_FILE,
    FEATURE_IMPORTANCE_PATH, MODEL_METADATA_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TOP_SHAP_FEATURES,
)


class ChurnPredictor:

    def __init__(self):
        print("🔄 Loading artifacts...")
        self.pipeline   = joblib.load(os.path.join(ROOT, PIPELINE_PATH))
        self.explainer  = joblib.load(os.path.join(ROOT, SHAP_EXPLAINER_PATH))

        with open(os.path.join(ROOT, FEATURE_ORDER_FILE))     as f:
            self.feature_order = json.load(f)
        with open(os.path.join(ROOT, FEATURE_IMPORTANCE_PATH)) as f:
            self.feature_importance = json.load(f)
        with open(os.path.join(ROOT, MODEL_METADATA_PATH))    as f:
            self.metadata = json.load(f)

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier   = self.pipeline.named_steps["classifier"]
        print(f"   ✅ Model: {self.metadata['best_model']} | "
              f"Test AUC: {self.metadata['test_auc']:.4f}")
        print("✅ Predictor ready.\n")

    # ── internal helpers ──────────────────────────────────────────────────────
    def _build_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a clean DataFrame from a list of customer dicts."""
        all_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        records  = [{col: r.get(col, None) for col in all_cols} for r in rows]
        return pd.DataFrame(records, columns=all_cols)

    def _risk(self, p: float) -> str:
        return "High" if p >= 0.70 else "Medium" if p >= 0.40 else "Low"

    def _timeline(self, p: float) -> str:
        return ("1 Month"  if p >= 0.75 else
                "3 Months" if p >= 0.45 else
                "6 Months" if p >= 0.20 else "Stable")

    def _shap_for_matrix(self, transformed_df: pd.DataFrame):
        """Run SHAP once on entire matrix, return 2-D numpy array."""
        sv = self.explainer.shap_values(transformed_df)
        if isinstance(sv, list):   # RandomForest → [class0, class1]
            return sv[1]
        return sv                  # XGBoost → 2-D array directly

    def _top_shap(self, shap_row: np.ndarray) -> list:
        pairs = sorted(
            zip(self.feature_order, shap_row.tolist()),
            key=lambda x: abs(x[1]), reverse=True
        )[:TOP_SHAP_FEATURES]
        return [
            {
                "feature":    feat,
                "shap_value": round(val, 4),
                "direction":  "increases churn risk" if val > 0
                              else "decreases churn risk",
            }
            for feat, val in pairs
        ]

    # ── PUBLIC: single prediction (used by /predict endpoint) ─────────────────
    def predict(self, customer_data: dict) -> dict:
        input_df          = self._build_df([customer_data])
        pred              = int(self.pipeline.predict(input_df)[0])
        prob              = float(self.pipeline.predict_proba(input_df)[0][1])
        transformed       = self.preprocessor.transform(input_df)
        transformed_df    = pd.DataFrame(transformed, columns=self.feature_order)
        sv_matrix         = self._shap_for_matrix(transformed_df)
        top_shap          = self._top_shap(sv_matrix[0])

        # add customer_value for single predict (used by recommender)
        for i, item in enumerate(top_shap):
            item["customer_value"] = str(transformed_df[item["feature"]].values[0])

        return {
            "churn_prediction":  pred,
            "churn_label":       "Will Churn" if pred == 1 else "Will Not Churn",
            "churn_probability": round(prob, 4),
            "risk_level":        self._risk(prob),
            "top_shap_features": top_shap,
        }

    # ── PUBLIC: batch prediction (used by /predict/batch-analyze) ─────────────
    def predict_batch(self, rows: list[dict]) -> list[dict]:
        """
        Process ALL customers in one vectorised pass.
        1 preprocessor transform + 1 model predict + 1 SHAP call for entire dataset.
        """
        print(f"   ⚡ Batch predicting {len(rows)} customers...")

        input_df       = self._build_df(rows)

        # Step 1: predict probabilities for all rows at once
        probs          = self.pipeline.predict_proba(input_df)[:, 1]
        preds          = (probs >= 0.5).astype(int)

        # Step 2: transform entire dataset once
        transformed    = self.preprocessor.transform(input_df)
        transformed_df = pd.DataFrame(transformed, columns=self.feature_order)

        # Step 3: SHAP for entire dataset in ONE call
        print(f"   ⚡ Computing SHAP for {len(rows)} customers (batch)...")
        sv_matrix      = self._shap_for_matrix(transformed_df)
        print(f"   ✅ SHAP done.")

        # Step 4: assemble results
        results = []
        for i, row in enumerate(rows):
            prob     = float(probs[i])
            pred     = int(preds[i])
            top_shap = self._top_shap(sv_matrix[i])

            results.append({
                "churn_prediction":  pred,
                "churn_label":       "Will Churn" if pred == 1 else "Will Not Churn",
                "churn_probability": round(prob, 4),
                "risk_level":        self._risk(prob),
                "churn_timeline":    self._timeline(prob),
                "top_shap_features": top_shap,
            })

        return results


predictor = ChurnPredictor()
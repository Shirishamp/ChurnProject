# backend/main.py
# ============================================================
# FastAPI — all endpoints including batch analytics
# ============================================================

import os, sys, json, io
from typing import List

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.schemas     import CustomerInput, PredictionResponse, HealthResponse
from backend.predictor   import predictor
from backend.recommender import generate_recommendations
from utils.config        import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL, ID_COL

app = FastAPI(
    title       = "Customer Churn Prediction API",
    description = "XGBoost + SHAP — churn prediction & analytics",
    version     = "2.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _assign_churn_timeline(prob: float) -> str:
    """Bucket churn probability into timeline risk."""
    if prob >= 0.75:
        return "1 Month"
    elif prob >= 0.45:
        return "3 Months"
    elif prob >= 0.20:
        return "6 Months"
    else:
        return "Stable"


def _predict_row(row: dict) -> dict:
    """Run predictor on one row dict, return enriched result."""
    result = predictor.predict(row)
    recs   = generate_recommendations(
        top_shap_features = result["top_shap_features"],
        customer_data     = row,
        churn_probability = result["churn_probability"],
        n=3,
    )
    result["recommendations"]  = recs
    result["churn_timeline"]   = _assign_churn_timeline(result["churn_probability"])
    return result


# ── ENDPOINT 1: Health ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    meta = predictor.metadata
    return {
        "status":     "ok",
        "model":      meta["best_model"],
        "test_auc":   meta["test_auc"],
        "cv_auc":     meta["cv_results"]["XGBoost"]["mean_auc"],
        "n_features": meta["n_features_after_ohe"],
    }


# ── ENDPOINT 2: Single predict ────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    try:
        customer_dict = customer.model_dump()
        result        = predictor.predict(customer_dict)
        recs          = generate_recommendations(
            top_shap_features = result["top_shap_features"],
            customer_data     = customer_dict,
            churn_probability = result["churn_probability"],
            n=3,
        )
        return {
            "customer_input":    customer_dict,
            "churn_prediction":  result["churn_prediction"],
            "churn_label":       result["churn_label"],
            "churn_probability": result["churn_probability"],
            "risk_level":        result["risk_level"],
            "top_shap_features": result["top_shap_features"],
            "recommendations":   recs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 3: Batch file upload + full analytics ────────────────────────────
@app.post("/predict/batch-analyze", tags=["Analytics"])
async def batch_analyze(file: UploadFile = File(...)):
    """
    Upload CSV or XLSX file with customer data.
    Returns full analytics payload for the dashboard.
    """
    try:
        contents = await file.read()

        # Parse file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            # Read all sheets — pick "E Comm" if present, else use first sheet
            xl      = pd.ExcelFile(io.BytesIO(contents))
            sheet   = "E Comm" if "E Comm" in xl.sheet_names else xl.sheet_names[0]
            df      = xl.parse(sheet)
        else:
            raise HTTPException(status_code=400, detail="Only CSV or XLSX files supported.")

        # Accept files with or without CustomerID
        has_id = ID_COL in df.columns
        if not has_id:
            df[ID_COL] = [f"CUST_{i+1:05d}" for i in range(len(df))]

        # Drop target if present (we're predicting)
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        required_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        if len(df) > 10000:
            raise HTTPException(status_code=400, detail="Max 10,000 rows per upload.")

        # ── Run predictions (full batch — single SHAP call) ────────────────
        customer_ids = df[ID_COL].astype(str).tolist()
        feat_cols    = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        rows_list    = []
        profiles     = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict.pop(ID_COL, None)
            # Build clean profile (replace NaN with None)
            profile  = {
                col: (None if pd.isna(v) else (
                    int(v) if isinstance(v, float) and v == int(v) else v
                ))
                for col, v in row_dict.items()
                if col in feat_cols
            }
            rows_list.append(profile)
            profiles.append(profile)

        # Single vectorised call — transforms + predicts + SHAP in one pass
        batch_results = predictor.predict_batch(rows_list)

        customers = []
        for i, (cust_id, profile, result) in enumerate(
                zip(customer_ids, profiles, batch_results)):
            recs = generate_recommendations(
                top_shap_features = result["top_shap_features"],
                customer_data     = profile,
                churn_probability = result["churn_probability"],
                n=3,
            )
            customers.append({
                "customer_id":       cust_id,
                "profile":           profile,
                "churn_prediction":  result["churn_prediction"],
                "churn_probability": result["churn_probability"],
                "churn_label":       result["churn_label"],
                "risk_level":        result["risk_level"],
                "churn_timeline":    result["churn_timeline"],
                "top_shap_features": result["top_shap_features"],
                "recommendations":   recs,
            })

        # ── Aggregated analytics ───────────────────────────────────────────
        total      = len(customers)
        churners   = [c for c in customers if c["churn_prediction"] == 1]
        n_churn    = len(churners)
        churn_rate = round(n_churn / total * 100, 1)

        # Timeline buckets
        timeline_counts = {"1 Month": 0, "3 Months": 0, "6 Months": 0, "Stable": 0}
        for c in customers:
            timeline_counts[c["churn_timeline"]] += 1

        # Risk distribution
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        for c in customers:
            risk_counts[c["risk_level"]] += 1

        # Top 3 global churn reasons (most frequent top-SHAP feature among churners)
        feat_freq: dict = {}
        for c in churners:
            for sf in c["top_shap_features"]:
                if sf["shap_value"] > 0:
                    f = sf["feature"]
                    feat_freq[f] = feat_freq.get(f, 0) + 1
        top_reasons = sorted(feat_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        top_reasons_out = [
            {
                "rank":    i + 1,
                "feature": feat,
                "count":   cnt,
                "pct":     round(cnt / n_churn * 100, 1) if n_churn else 0,
            }
            for i, (feat, cnt) in enumerate(top_reasons)
        ]

        # Segment breakdowns (for bar charts)
        def seg_churn(col):
            out = {}
            for c in customers:
                val = str(c["profile"].get(col, "Unknown") or "Unknown")
                if val not in out:
                    out[val] = {"total": 0, "churn": 0}
                out[val]["total"] += 1
                if c["churn_prediction"] == 1:
                    out[val]["churn"] += 1
            return [
                {"label": k, "total": v["total"],
                 "churn": v["churn"],
                 "churn_rate": round(v["churn"] / v["total"] * 100, 1)}
                for k, v in sorted(out.items())
            ]

        # Satisfaction score distribution among churners vs retained
        def sat_dist():
            dist = {}
            for c in customers:
                score = str(c["profile"].get("SatisfactionScore", "?") or "?")
                if score not in dist:
                    dist[score] = {"churned": 0, "retained": 0}
                key = "churned" if c["churn_prediction"] == 1 else "retained"
                dist[score][key] += 1
            return [{"score": k, **v} for k, v in sorted(dist.items())]

        # Probability histogram buckets
        prob_buckets = [0] * 10
        for c in customers:
            idx = min(int(c["churn_probability"] * 10), 9)
            prob_buckets[idx] += 1

        # Action recommendation: customers to target to prevent churn next month
        at_risk_next_month = sorted(
            [c for c in customers if c["churn_timeline"] == "1 Month"],
            key=lambda x: x["churn_probability"], reverse=True
        )
        action_pct = round(len(at_risk_next_month) / total * 100, 1)

        # Collect all unique recommendations for at-risk group
        action_rec_freq: dict = {}
        for c in at_risk_next_month:
            for r in c["recommendations"]:
                txt = r["recommendation"]
                action_rec_freq[txt] = action_rec_freq.get(txt, 0) + 1
        top_actions = sorted(action_rec_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "meta": {
                "total_customers":  total,
                "total_churners":   n_churn,
                "churn_rate_pct":   churn_rate,
                "action_pct":       action_pct,
                "at_risk_count":    len(at_risk_next_month),
            },
            "timeline_buckets":  timeline_counts,
            "risk_distribution": risk_counts,
            "top_reasons":       top_reasons_out,
            "segments": {
                "by_gender":        seg_churn("Gender"),
                "by_city_tier":     seg_churn("CityTier"),
                "by_marital":       seg_churn("MaritalStatus"),
                "by_login_device":  seg_churn("PreferredLoginDevice"),
                "by_payment":       seg_churn("PreferredPaymentMode"),
                "by_order_cat":     seg_churn("PreferedOrderCat"),
            },
            "satisfaction_dist":  sat_dist(),
            "prob_histogram":     [
                {"bucket": f"{i*10}-{i*10+10}%", "count": prob_buckets[i]}
                for i in range(10)
            ],
            "top_actions":        [
                {"recommendation": t, "affected_customers": c}
                for t, c in top_actions
            ],
            "customers":          customers,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 4: Feature importance ────────────────────────────────────────────
@app.get("/features/importance", tags=["Explainability"])
def get_feature_importance(top_n: int = 15):
    importance = predictor.feature_importance
    top = list(importance.items())[:top_n]
    return {
        "model": predictor.metadata["best_model"],
        "features": [
            {"rank": i+1, "feature": k, "mean_shap": round(v, 4)}
            for i, (k, v) in enumerate(top)
        ]
    }


# ── ENDPOINT 5: Model info ────────────────────────────────────────────────────
@app.get("/model/info", tags=["System"])
def get_model_info():
    return predictor.metadata
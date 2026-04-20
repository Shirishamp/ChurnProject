# backend/main.py
# ============================================================
# FastAPI application — all endpoints
# Run with: uvicorn backend.main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.schemas     import CustomerInput, PredictionResponse, HealthResponse
from backend.predictor   import predictor
from backend.recommender import generate_recommendations

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Customer Churn Prediction API",
    description = "Predicts e-commerce customer churn using XGBoost + SHAP explanations + personalised recommendations.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allows the HTML frontend to call this API locally ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to specific domain in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── ENDPOINT 1: Health Check ──────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Returns model status, AUC scores and feature count.
    Use this to confirm the API is running and model is loaded.
    """
    meta = predictor.metadata
    return {
        "status":     "ok",
        "model":      meta["best_model"],
        "test_auc":   meta["test_auc"],
        "cv_auc":     meta["cv_results"]["XGBoost"]["mean_auc"],
        "n_features": meta["n_features_after_ohe"],
    }


# ── ENDPOINT 2: Single Customer Prediction ────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Accepts a single customer's data and returns:
    - Churn prediction (0 or 1)
    - Churn probability (0.0 – 1.0)
    - Risk level (Low / Medium / High)
    - Top SHAP features driving the prediction
    - Personalised retention recommendations
    """
    try:
        customer_dict = customer.model_dump()

        # Run prediction + SHAP
        result = predictor.predict(customer_dict)

        # Generate recommendations from SHAP features
        # Generate recommendations using RAW customer values (not scaled)
        recommendations = generate_recommendations(
            top_shap_features = result["top_shap_features"],
            customer_data     = customer_dict,        # raw original values
            churn_probability = result["churn_probability"],
            n                 = 3,
        )

        return {
            "customer_input":    customer_dict,
            "churn_prediction":  result["churn_prediction"],
            "churn_label":       result["churn_label"],
            "churn_probability": result["churn_probability"],
            "risk_level":        result["risk_level"],
            "top_shap_features": result["top_shap_features"],
            "recommendations":   recommendations,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 3: Batch Prediction ──────────────────────────────────────────────
@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerInput]):
    """
    Accepts a list of customers and returns predictions for all.
    Maximum 500 customers per request.
    """
    if len(customers) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds limit of 500 customers per request."
        )
    try:
        results = []
        for customer in customers:
            customer_dict = customer.model_dump()
            result = predictor.predict(customer_dict)
            recommendations = generate_recommendations(
                top_shap_features = result["top_shap_features"],
                customer_data     = customer_dict,
                churn_probability = result["churn_probability"],
                n                 = 3,
            )
            results.append({
                "churn_prediction":  result["churn_prediction"],
                "churn_label":       result["churn_label"],
                "churn_probability": result["churn_probability"],
                "risk_level":        result["risk_level"],
                "top_shap_features": result["top_shap_features"],
                "recommendations":   recommendations,
            })
        return {"total": len(results), "predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 4: Feature Importance ────────────────────────────────────────────
@app.get("/features/importance", tags=["Explainability"])
def get_feature_importance(top_n: int = 15):
    """
    Returns global feature importance (mean |SHAP|) from training.
    top_n: how many top features to return (default 15).
    """
    importance = predictor.feature_importance
    top = list(importance.items())[:top_n]
    return {
        "model":    predictor.metadata["best_model"],
        "top_n":    top_n,
        "features": [
            {"rank": i+1, "feature": k, "mean_shap": round(v, 4)}
            for i, (k, v) in enumerate(top)
        ]
    }


# ── ENDPOINT 5: Model Metadata ────────────────────────────────────────────────
@app.get("/model/info", tags=["System"])
def get_model_info():
    """Returns full model metadata including all 3 model CV scores."""
    return predictor.metadata
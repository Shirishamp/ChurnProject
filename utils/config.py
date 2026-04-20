# utils/config.py
# ============================================================
# SINGLE SOURCE OF TRUTH — used by training, backend, frontend
# DO NOT hardcode these values anywhere else in the project
# ============================================================

TARGET_COL = "Churn"
ID_COL     = "CustomerID"
DROP_COLS  = [ID_COL, TARGET_COL]

NUMERICAL_FEATURES = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "CityTier",
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# Artifact paths (all relative to project root)
PIPELINE_PATH           = "artifacts/churn_pipeline.joblib"
SHAP_EXPLAINER_PATH     = "artifacts/shap_explainer.joblib"
FEATURE_IMPORTANCE_PATH = "artifacts/feature_importance.json"
FEATURE_ORDER_FILE      = "artifacts/feature_order.json"
MODEL_METADATA_PATH     = "artifacts/model_metadata.json"

# Training config
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Recommendation config
TOP_SHAP_FEATURES = 3

# All candidate model names
MODEL_CANDIDATES = ["LogisticRegression", "RandomForest", "XGBoost"]
# backend/schemas.py
# ============================================================
# Pydantic models for request validation and response structure
# All field names EXACTLY match the dataset column names
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional


class CustomerInput(BaseModel):
    """
    Input schema for a single customer prediction request.
    All fields match the original dataset columns exactly.
    Optional fields are the ones that had missing values in training data.
    """
    Tenure:                     Optional[float] = Field(None,  description="Months with company")
    WarehouseToHome:            Optional[float] = Field(None,  description="Distance warehouse to home (km)")
    HourSpendOnApp:             Optional[float] = Field(None,  description="Hours spent on app per day")
    NumberOfDeviceRegistered:   int             = Field(...,   description="Number of devices registered")
    SatisfactionScore:          int             = Field(...,   description="Customer satisfaction score (1-5)")
    NumberOfAddress:            int             = Field(...,   description="Number of saved addresses")
    Complain:                   int             = Field(...,   description="1 = raised complaint, 0 = no complaint")
    OrderAmountHikeFromlastYear:Optional[float] = Field(None,  description="% order amount increase from last year")
    CouponUsed:                 Optional[float] = Field(None,  description="Total coupons used last month")
    OrderCount:                 Optional[float] = Field(None,  description="Total orders last month")
    DaySinceLastOrder:          Optional[float] = Field(None,  description="Days since last order")
    CashbackAmount:             float           = Field(...,   description="Average cashback received (INR)")
    CityTier:                   int             = Field(...,   description="City tier: 1, 2, or 3")
    PreferredLoginDevice:       str             = Field(...,   description="Mobile Phone / Phone / Computer / Tablet")
    PreferredPaymentMode:       str             = Field(...,   description="Debit Card / Credit Card / UPI / Cash on Delivery / E wallet / CC")
    Gender:                     str             = Field(...,   description="Male / Female")
    PreferedOrderCat:           str             = Field(...,   description="Laptop & Accessory / Mobile / Mobile Phone / Fashion / Grocery / Others")
    MaritalStatus:              str             = Field(...,   description="Single / Married / Divorced")

    class Config:
        json_schema_extra = {
            "example": {
                "Tenure": 4.0,
                "WarehouseToHome": 6.0,
                "HourSpendOnApp": 3.0,
                "NumberOfDeviceRegistered": 3,
                "SatisfactionScore": 2,
                "NumberOfAddress": 9,
                "Complain": 1,
                "OrderAmountHikeFromlastYear": 11.0,
                "CouponUsed": 1.0,
                "OrderCount": 1.0,
                "DaySinceLastOrder": 5.0,
                "CashbackAmount": 159.93,
                "CityTier": 3,
                "PreferredLoginDevice": "Mobile Phone",
                "PreferredPaymentMode": "Debit Card",
                "Gender": "Female",
                "PreferedOrderCat": "Laptop & Accessory",
                "MaritalStatus": "Single"
            }
        }


class PredictionResponse(BaseModel):
    """Response returned for every prediction request."""
    customer_input:       dict
    churn_prediction:     int           # 0 or 1
    churn_label:          str           # "Will Churn" or "Will Not Churn"
    churn_probability:    float         # probability of churn (0.0 – 1.0)
    risk_level:           str           # Low / Medium / High
    top_shap_features:    list          # top N features driving this prediction
    recommendations:      list          # personalized action items


class HealthResponse(BaseModel):
    status:       str
    model:        str
    test_auc:     float
    cv_auc:       float
    n_features:   int
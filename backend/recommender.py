# backend/recommender.py
# ============================================================
# Generates personalised retention recommendations
# based on top SHAP features driving churn for this customer
# NOTE: customer_value passed here is the RAW original value
#       from customer_input, not the scaled/transformed value
# ============================================================

RECOMMENDATION_RULES = {

    "Complain": {
        "condition": lambda v: float(v) >= 0.5,
        "recommendation": (
            "URGENT: Customer raised a complaint. Assign a dedicated support agent immediately. "
            "Follow up within 24 hours with a goodwill coupon (10-15% off next order)."
        ),
        "priority": "High"
    },

    "Tenure": {
        "condition": lambda v: float(v) < 3,
        "recommendation": (
            "New customer (low tenure). Trigger a welcome loyalty programme — "
            "offer a milestone reward after 3rd and 6th month to build long-term retention."
        ),
        "priority": "High"
    },

    "SatisfactionScore": {
        "condition": lambda v: float(v) <= 2,
        "recommendation": (
            "Low satisfaction score detected. Send a personalised feedback survey within 48 hours. "
            "Offer a service-recovery discount of 10-15% on the next order."
        ),
        "priority": "High"
    },

    "DaySinceLastOrder": {
        "condition": lambda v: float(v) >= 7,
        "recommendation": (
            "Customer inactive for {val} days. Launch a re-engagement campaign: "
            "personalised product recommendations + time-limited offer (expires in 48 hours)."
        ),
        "priority": "Medium",
        "use_value": True
    },

    "CashbackAmount": {
        "condition": lambda v: float(v) < 150,
        "recommendation": (
            "Below-average cashback usage. Highlight cashback benefits in the next push notification "
            "to reinforce platform value and drive repeat purchases."
        ),
        "priority": "Medium"
    },

    "CouponUsed": {
        "condition": lambda v: float(v) == 0,
        "recommendation": (
            "Customer has never used coupons. Send personalised coupon bundles matching "
            "their preferred order category to incentivise the next purchase."
        ),
        "priority": "Medium"
    },

    "OrderCount": {
        "condition": lambda v: float(v) <= 1,
        "recommendation": (
            "Low order frequency. Introduce a 'Buy 3 Get 1 Free' offer or "
            "a monthly subscription benefit to increase order cadence."
        ),
        "priority": "Medium"
    },

    "HourSpendOnApp": {
        "condition": lambda v: float(v) <= 1,
        "recommendation": (
            "Low app engagement. Push a personalised in-app notification with "
            "curated deals to increase session time and product discovery."
        ),
        "priority": "Medium"
    },

    "NumberOfDeviceRegistered": {
        "condition": lambda v: float(v) <= 1,
        "recommendation": (
            "Only one device registered. Prompt customer to register additional devices "
            "with a multi-device bonus to increase platform stickiness."
        ),
        "priority": "Low"
    },

    "WarehouseToHome": {
        "condition": lambda v: float(v) > 20,
        "recommendation": (
            "Long delivery distance may be causing frustration. "
            "Offer express delivery upgrade or highlight nearest fulfilment option."
        ),
        "priority": "Medium"
    },

    "OrderAmountHikeFromlastYear": {
        "condition": lambda v: float(v) > 20,
        "recommendation": (
            "Order value has increased significantly — customer may be price-sensitive. "
            "Offer a loyalty price-lock or introduce a premium membership with free delivery."
        ),
        "priority": "Medium"
    },

    "NumberOfAddress": {
        "condition": lambda v: float(v) >= 5,
        "recommendation": (
            "High number of saved addresses suggests frequent location changes. "
            "Ensure delivery reliability and offer flexible address management features."
        ),
        "priority": "Low"
    },

    "CityTier": {
        "condition": lambda v: float(v) >= 2,
        "recommendation": (
            "Customer is in a Tier 2/3 city. Ensure competitive delivery SLAs in their region "
            "and highlight localised offers and regional deals."
        ),
        "priority": "Low"
    },
}

# OHE prefix rules — fired when OHE-derived feature appears in top SHAP
OHE_RULES = {
    "MaritalStatus_Single": {
        "recommendation": (
            "Single customer profile shows higher churn tendency. "
            "Target with referral programmes or social/community features."
        ),
        "priority": "Low"
    },
    "PreferedOrderCat_Mobile": {
        "recommendation": (
            "Frequent mobile buyer. Offer early access to new mobile launches "
            "or trade-in upgrade programmes."
        ),
        "priority": "Low"
    },
    "PreferedOrderCat_Fashion": {
        "recommendation": (
            "Fashion buyer. Send seasonal collection previews "
            "and exclusive early-sale access."
        ),
        "priority": "Low"
    },
    "PreferredPaymentMode_Cash on Delivery": {
        "recommendation": (
            "Customer prefers Cash on Delivery. Offer a digital payment incentive "
            "(extra cashback on UPI/card) to migrate to faster payment methods."
        ),
        "priority": "Low"
    },
}

FALLBACK_RECOMMENDATIONS = [
    "Enroll customer in the VIP loyalty programme to increase perceived value.",
    "Send a personalised monthly digest of top deals matching order history.",
    "Offer a surprise reward on the next order to delight and retain.",
    "Enable smart push notifications for restocks in preferred categories.",
    "Proactively reach out via in-app chat to check on customer experience.",
]


def generate_recommendations(
    top_shap_features: list,
    customer_data: dict,
    churn_probability: float,
    n: int = 3
) -> list:
    """
    Generate up to n personalised recommendations.

    Uses RAW customer_data values (not scaled) for condition checks.
    Falls back to generic recommendations if specific rules don't fire.
    """
    recommendations = []
    seen_features   = set()

    for feat_info in top_shap_features:
        if len(recommendations) >= n:
            break

        feature  = feat_info["feature"]
        shap_val = feat_info["shap_value"]

        # Only act on features pushing toward churn
        if shap_val <= 0:
            continue

        if feature in seen_features:
            continue

        # ── Try exact match in RECOMMENDATION_RULES ──────────────────────
        if feature in RECOMMENDATION_RULES:
            rule = RECOMMENDATION_RULES[feature]
            raw_value = customer_data.get(feature)

            if raw_value is not None:
                try:
                    if rule["condition"](raw_value):
                        rec_text = rule["recommendation"]
                        # Substitute {val} placeholder if needed
                        if rule.get("use_value"):
                            rec_text = rec_text.format(val=int(float(raw_value)))

                        recommendations.append({
                            "driven_by":      feature,
                            "shap_impact":    round(shap_val, 4),
                            "recommendation": rec_text,
                            "priority":       rule["priority"]
                        })
                        seen_features.add(feature)
                        continue
                except Exception:
                    pass

        # ── Try OHE prefix match ──────────────────────────────────────────
        matched_ohe = False
        for ohe_key, ohe_rule in OHE_RULES.items():
            if feature.startswith(ohe_key) or feature == ohe_key:
                recommendations.append({
                    "driven_by":      feature,
                    "shap_impact":    round(shap_val, 4),
                    "recommendation": ohe_rule["recommendation"],
                    "priority":       ohe_rule["priority"]
                })
                seen_features.add(feature)
                matched_ohe = True
                break

        if matched_ohe:
            continue

        # ── Try base feature name from OHE (e.g. Complain from Complain_1) ─
        base_feature = feature.split("_")[0]
        if base_feature in RECOMMENDATION_RULES and base_feature not in seen_features:
            rule = RECOMMENDATION_RULES[base_feature]
            raw_value = customer_data.get(base_feature)
            if raw_value is not None:
                try:
                    if rule["condition"](raw_value):
                        rec_text = rule["recommendation"]
                        if rule.get("use_value"):
                            rec_text = rec_text.format(val=int(float(raw_value)))
                        recommendations.append({
                            "driven_by":      base_feature,
                            "shap_impact":    round(shap_val, 4),
                            "recommendation": rec_text,
                            "priority":       rule["priority"]
                        })
                        seen_features.add(base_feature)
                        continue
                except Exception:
                    pass

    # ── Fill remaining slots with fallbacks ──────────────────────────────────
    for fallback in FALLBACK_RECOMMENDATIONS:
        if len(recommendations) >= n:
            break
        recommendations.append({
            "driven_by":      "general_retention",
            "shap_impact":    None,
            "recommendation": fallback,
            "priority":       "Low"
        })

    return recommendations
"""
FastAPI Churn Scoring Service
D2C Customer Churn Intelligence Capstone — Part 4

Loads the trained model (model.pkl from Part 3) and exposes endpoints for the
internal CRM tool to score customer churn risk.
"""

from typing import Optional, List, Literal
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = "model.pkl"
DECISION_THRESHOLD = 0.40  # same business-justified threshold chosen in Part 3

ml_model = None  # populated at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    try:
        ml_model = joblib.load(MODEL_PATH)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not find {MODEL_PATH}. Run train_model.py first, or copy model.pkl "
            f"from the Part 3 repository into this directory."
        ) from e
    yield
    ml_model = None


app = FastAPI(
    title="D2C Churn Scoring API",
    description="Internal churn-risk scoring service for the CRM tool. "
                "Scores customers using a trained XGBoost model (see Part 3 model_card.md).",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------
# Pydantic schemas — input validation
# -----------------------------------------------------------------------

CityTier = Literal["Tier 1", "Tier 2", "Tier 3"]
AgeGroup = Literal["18-24", "25-34", "35-44", "45+"]
AcquisitionChannel = Literal[
    "Google Search", "Instagram", "Influencer", "Referral", "Marketplace", "Organic"
]
LoyaltyTier = Literal["Silver", "Gold", "Platinum"]
PreferredCategory = Literal[
    "Skin Care", "Hair Care", "Makeup", "Fragrance", "Wellness", "Baby Care"
]
MarketingConsent = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """One customer's feature payload, matching the rfm_modeling_snapshot.csv schema
    (excluding customer_id, snapshot_date, churn_next_60d, and split — none of which are
    model inputs)."""

    customer_id: str = Field(..., description="Customer identifier, echoed back in the response only")

    city_tier: CityTier
    age_group: AgeGroup
    acquisition_channel: AcquisitionChannel
    loyalty_tier: Optional[LoyaltyTier] = Field(
        None, description="Null/omitted means not enrolled in the loyalty program"
    )
    preferred_category: PreferredCategory
    marketing_consent: MarketingConsent

    recency_days: int = Field(..., ge=0, description="Days since the customer's last pre-snapshot order")
    frequency_180d: int = Field(..., ge=0, description="Order count in the trailing 180 days")
    monetary_180d: float = Field(..., ge=0, description="Total gross spend (INR) in the trailing 180 days")
    return_rate_180d: float = Field(..., ge=0, le=1, description="Proportion of orders returned (180d)")
    avg_discount_pct_180d: float = Field(..., ge=0, le=1, description="Average discount fraction (180d)")
    avg_rating_180d: Optional[float] = Field(None, ge=1, le=5, description="Average order rating (180d); null if no ratings")
    category_diversity_180d: int = Field(..., ge=0, description="Distinct categories purchased (180d)")

    ticket_count_90d: int = Field(..., ge=0, description="Support tickets raised in the trailing 90 days")
    negative_ticket_rate_90d: float = Field(..., ge=0, le=1, description="Proportion of tickets with negative sentiment (90d)")
    avg_resolution_hours_90d: float = Field(..., ge=0, description="Average ticket resolution time in hours (90d)")

    days_since_signup: int = Field(..., ge=0)

    sessions_30d: int = Field(..., ge=0)
    product_views_30d: int = Field(..., ge=0)
    cart_adds_30d: int = Field(..., ge=0)
    wishlist_adds_30d: int = Field(..., ge=0)
    abandoned_carts_30d: int = Field(..., ge=0)
    email_opens_30d: int = Field(..., ge=0)
    campaign_clicks_30d: int = Field(..., ge=0)
    last_visit_days_ago: int = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST00001",
                "city_tier": "Tier 1",
                "age_group": "25-34",
                "acquisition_channel": "Instagram",
                "loyalty_tier": "Gold",
                "preferred_category": "Skin Care",
                "marketing_consent": "Yes",
                "recency_days": 110,
                "frequency_180d": 1,
                "monetary_180d": 620.0,
                "return_rate_180d": 0.0,
                "avg_discount_pct_180d": 0.25,
                "avg_rating_180d": 4.0,
                "category_diversity_180d": 1,
                "ticket_count_90d": 1,
                "negative_ticket_rate_90d": 1.0,
                "avg_resolution_hours_90d": 18.5,
                "days_since_signup": 240,
                "sessions_30d": 2,
                "product_views_30d": 4,
                "cart_adds_30d": 1,
                "wishlist_adds_30d": 0,
                "abandoned_carts_30d": 1,
                "email_opens_30d": 0,
                "campaign_clicks_30d": 0,
                "last_visit_days_ago": 25,
            }
        }
    )


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_class: int
    risk_level: Literal["low", "medium", "high"]
    risk_explanation: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

FEATURE_COLS_NUM = [
    "recency_days", "frequency_180d", "monetary_180d", "return_rate_180d",
    "avg_discount_pct_180d", "avg_rating_180d", "category_diversity_180d",
    "ticket_count_90d", "negative_ticket_rate_90d", "avg_resolution_hours_90d",
    "days_since_signup", "sessions_30d", "product_views_30d", "cart_adds_30d",
    "wishlist_adds_30d", "abandoned_carts_30d", "email_opens_30d",
    "campaign_clicks_30d", "last_visit_days_ago",
]
FEATURE_COLS_CAT = [
    "city_tier", "age_group", "acquisition_channel", "loyalty_tier",
    "preferred_category", "marketing_consent",
]


def _to_feature_frame(customers: List[CustomerFeatures]) -> pd.DataFrame:
    rows = [c.model_dump() for c in customers]
    df = pd.DataFrame(rows)
    return df[FEATURE_COLS_NUM + FEATURE_COLS_CAT]


def _risk_level(probability: float) -> str:
    if probability >= 0.65:
        return "high"
    if probability >= DECISION_THRESHOLD:
        return "medium"
    return "low"


def _risk_explanation(row: pd.Series, probability: float) -> str:
    """Generate a short, rule-based explanation referencing the customer's own values,
    using the same top features identified in Part 3's feature-importance analysis
    (recency_days, negative_ticket_rate_90d, frequency_180d, monetary_180d)."""
    reasons = []
    if row["recency_days"] >= 90:
        reasons.append(f"no order in {int(row['recency_days'])} days")
    if row["negative_ticket_rate_90d"] >= 0.5:
        reasons.append("a high rate of negative support interactions")
    if row["frequency_180d"] <= 1:
        reasons.append("low order frequency in the past 180 days")
    if row["last_visit_days_ago"] >= 20:
        reasons.append(f"no app/site visit in {int(row['last_visit_days_ago'])} days")
    if row["return_rate_180d"] >= 0.3:
        reasons.append("an elevated return rate")

    if not reasons:
        if probability < DECISION_THRESHOLD:
            return "Recent activity and engagement look healthy; no major risk signals detected."
        return "Model flagged elevated risk based on a combination of features without one dominant signal."

    level = _risk_level(probability)
    prefix = {
        "high": "High churn risk driven by",
        "medium": "Moderate churn risk driven by",
        "low": "Some minor risk signals present:",
    }[level]
    return f"{prefix} {', '.join(reasons)}."


def _predict_one(customer: CustomerFeatures) -> PredictionResponse:
    df = _to_feature_frame([customer])
    proba = float(ml_model.predict_proba(df)[:, 1][0])
    predicted_class = int(proba >= DECISION_THRESHOLD)
    explanation = _risk_explanation(df.iloc[0], proba)
    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(proba, 4),
        predicted_class=predicted_class,
        risk_level=_risk_level(proba),
        risk_explanation=explanation,
    )


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=ml_model is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return _predict_one(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictRequest):
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        predictions = [_predict_one(c) for c in request.customers]
        return BatchPredictionResponse(predictions=predictions, count=len(predictions))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

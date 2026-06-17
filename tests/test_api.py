"""
API tests for the D2C Churn Scoring Service.

Run with: pytest tests/test_api.py -v
(Run from the repository root so that 'app' and 'model.pkl' resolve correctly.)
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


SAMPLE_CUSTOMER = {
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

HEALTHY_CUSTOMER = {
    **SAMPLE_CUSTOMER,
    "customer_id": "CUST00002",
    "recency_days": 5,
    "frequency_180d": 5,
    "monetary_180d": 3000.0,
    "negative_ticket_rate_90d": 0.0,
    "last_visit_days_ago": 1,
    "return_rate_180d": 0.0,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """GET /health should return status ok and confirm the model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response_shape(client):
    """POST /predict should return a well-formed prediction with all required fields."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()

    assert body["customer_id"] == "CUST00001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_class"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")
    assert isinstance(body["risk_explanation"], str) and len(body["risk_explanation"]) > 0


def test_predict_high_risk_customer_flagged_high(client):
    """A customer with high recency, low frequency, and bad support sentiment should be
    flagged as elevated risk (medium or high), not low."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in ("medium", "high")
    assert body["predicted_class"] == 1


def test_predict_healthy_customer_flagged_low(client):
    """A customer with very recent activity, high frequency, and good spend should be
    flagged as low risk."""
    response = client.post("/predict", json=HEALTHY_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == "low"
    assert body["predicted_class"] == 0


def test_batch_predict_returns_one_result_per_customer(client):
    """POST /batch_predict should return predictions for every customer submitted, in order."""
    payload = {"customers": [SAMPLE_CUSTOMER, HEALTHY_CUSTOMER]}
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["customer_id"] == "CUST00001"
    assert body["predictions"][1]["customer_id"] == "CUST00002"


def test_predict_rejects_invalid_categorical_value(client):
    """POST /predict should return 422 when an invalid categorical value (not in the
    allowed set) is submitted -- this confirms Pydantic validation is active."""
    bad_payload = {**SAMPLE_CUSTOMER, "city_tier": "Tier 9 - Does Not Exist"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client):
    """POST /predict should return 422 when a required field is missing entirely."""
    bad_payload = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "recency_days"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_value(client):
    """POST /predict should return 422 when a numeric field violates its range
    constraint (e.g., return_rate_180d must be between 0 and 1)."""
    bad_payload = {**SAMPLE_CUSTOMER, "return_rate_180d": 1.5}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_batch_predict_rejects_empty_list(client):
    """POST /batch_predict should reject an empty customer list (min_length=1)."""
    response = client.post("/batch_predict", json={"customers": []})
    assert response.status_code == 422

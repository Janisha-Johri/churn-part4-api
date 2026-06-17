# Part 4: FastAPI Churn Scoring Service & Reproducible ML Workflow

## D2C Customer Churn Intelligence Capstone Project (Part 4 of 4)

A FastAPI service that loads the trained churn model (from Part 3) and exposes endpoints for an
internal CRM tool to score customer churn risk.

---

## Project Overview

This service answers one question for the CRM team: **"How likely is this customer to churn in
the next 60 days, and why?"** It wraps the XGBoost model trained in Part 3 behind a simple REST
API with three endpoints (`/health`, `/predict`, `/batch_predict`), full input validation, and a
documented monitoring/responsible-use plan.

---

## Repository Structure

```
part4/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI app: endpoints, Pydantic schemas, prediction logic
├── tests/
│   └── test_api.py          # 9 test cases covering health, predict, batch_predict, validation
├── data/
│   └── rfm_modeling_snapshot.csv   # used only by train_model.py if retraining is needed
├── model.pkl                 # trained model artifact (scikit-learn Pipeline incl. preprocessing)
├── train_model.py            # fallback script to regenerate model.pkl from scratch
├── requirements.txt
├── Dockerfile
├── monitoring_plan.md
└── README.md
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Model file

`model.pkl` is already included in this repository and will be used as-is. If it's ever missing
or you want to regenerate it from scratch, run:

```bash
python train_model.py
```

This reads `data/rfm_modeling_snapshot.csv` and reproduces the same XGBoost pipeline trained in
Part 3, writing a fresh `model.pkl` to the repository root.

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs (Swagger UI) are
auto-generated at `http://127.0.0.1:8000/docs`.

### 4. Run tests

```bash
pytest tests/test_api.py -v
```

(Run this from the repository root so that `app/` and `model.pkl` resolve correctly.)

---

## Endpoint Details

### `GET /health`
Returns a simple health check confirming the API and model are ready.

**Sample response:**
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
Accepts one customer's feature payload and returns a churn-risk prediction.

**Sample request:**
```json
{
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
  "last_visit_days_ago": 25
}
```

**Sample response:**
```json
{
  "customer_id": "CUST00001",
  "churn_probability": 0.9033,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "High churn risk driven by no order in 110 days, a high rate of negative support interactions, low order frequency in the past 180 days, no app/site visit in 25 days."
}
```

### `POST /batch_predict`
Accepts multiple customer payloads (1-500 per request) and returns a prediction for each.

**Sample request:**
```json
{
  "customers": [ { "...same shape as /predict..." }, { "..." } ]
}
```

**Sample response:**
```json
{
  "predictions": [
    {"customer_id": "CUST00001", "churn_probability": 0.9033, "predicted_class": 1, "risk_level": "high", "risk_explanation": "..."},
    {"customer_id": "CUST00002", "churn_probability": 0.0335, "predicted_class": 0, "risk_level": "low", "risk_explanation": "Recent activity and engagement look healthy; no major risk signals detected."}
  ],
  "count": 2
}
```

---

## Input Validation

All inputs are validated via Pydantic models (`app/main.py`):
- Categorical fields (`city_tier`, `age_group`, `acquisition_channel`, `loyalty_tier`,
  `preferred_category`, `marketing_consent`) are restricted to the exact value sets defined in the
  data dictionary using `Literal` types -- invalid values return `422 Unprocessable Entity`.
- Numeric fields have range constraints (e.g., `return_rate_180d` must be between 0 and 1,
  `avg_rating_180d` between 1 and 5) -- out-of-range values return `422`.
- Missing required fields return `422` with a clear field-level error message.
- `loyalty_tier` and `avg_rating_180d` are optional (nullable), matching real-world missing-data
  cases documented in Part 1's data quality report.

---

## Test Execution

```bash
pytest tests/test_api.py -v
```

9 tests covering: health check, valid prediction shape, correct risk classification on both a
high-risk and a low-risk example customer, batch prediction with multiple customers, and four
input-validation failure cases (invalid category, missing field, out-of-range value, empty batch
list).

---

## Reproducibility

- `requirements.txt` pins exact package versions used during development and testing.
- `train_model.py` allows full model reproduction from raw data without needing the original
  Part 3 repository.
- A working `Dockerfile` is included:
  ```bash
  docker build -t churn-api .
  docker run -p 8000:8000 churn-api
  ```

---

## Model / Source Data Notes

- The model artifact (`model.pkl`) is the same XGBoost pipeline trained and evaluated in Part 3
  (test ROC-AUC 0.868, see that repository's `model_card.md` for full details).
- The decision threshold used in this API (0.40) matches the business-justified threshold selected
  in Part 3, Section 8.
- `risk_level` buckets: `high` (probability >= 0.65), `medium` (>= 0.40 and < 0.65), `low` (< 0.40).
- `risk_explanation` is generated using simple, transparent rules referencing the customer's own
  feature values against the top features identified in Part 3's feature-importance analysis
  (`recency_days`, `negative_ticket_rate_90d`, `frequency_180d`, `return_rate_180d`,
  `last_visit_days_ago`) -- not a black-box explanation.

---

## Responsible Use Note

**How the retention team SHOULD use this API's output:**
- As a **prioritization signal** to decide which customers receive proactive retention outreach,
  combined with the segment-based budget strategy from Part 2 (e.g., reserve high-cost
  interventions for customers who are both high-risk AND high historical value).
- As an input to human-reviewed campaigns, not as the sole trigger for fully automated actions.
- Alongside other knowledge the team has (e.g., a customer's recent direct complaint should
  trigger action regardless of their model score -- see Part 3's `error_analysis.md`, Case 9,
  where a customer with an explicit negative support ticket was still missed by the model).

**How the retention team SHOULD NOT use this API's output:**
- As the sole basis for denying a customer service, refunds, or account access.
- As a measure of a customer's worth as a person, rather than a behavioral churn likelihood.
- As a guarantee for any individual customer -- per the Part 3 error analysis, the model is
  sometimes wrong in both directions (flagging loyal customers as high-risk, and missing engaged
  customers who churn suddenly). It is intended for **prioritization at scale across the customer
  base**, not certainty about any one customer.
- Without monitoring -- see `monitoring_plan.md` for what must be tracked before relying on this
  model's outputs long-term in production.

See `monitoring_plan.md` for the full data-drift, prediction-distribution, business-outcome,
API-error, and retraining-trigger plan.

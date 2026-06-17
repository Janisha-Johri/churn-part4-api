"""
Fallback training script for the D2C Churn Scoring API.

This reproduces the final model from Part 3 (XGBoost pipeline) in case model.pkl is not
present. Run this once before starting the API if needed:

    python train_model.py

It writes model.pkl to the repository root, matching the format app/main.py expects.
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression  # noqa: F401 (kept for reference/baseline)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

RANDOM_STATE = 42

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


def main():
    df = pd.read_csv("data/rfm_modeling_snapshot.csv")

    X = df[FEATURE_COLS_NUM + FEATURE_COLS_CAT].copy()
    y = df["churn_next_60d"].copy()

    train_mask = df["split"] == "train"
    X_train, y_train = X[train_mask], y[train_mask]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, FEATURE_COLS_NUM),
        ("cat", categorical_transformer, FEATURE_COLS_CAT),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, eval_metric="logloss",
        )),
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "model.pkl")
    print(f"Trained on {len(X_train)} customers. Saved model.pkl")


if __name__ == "__main__":
    main()

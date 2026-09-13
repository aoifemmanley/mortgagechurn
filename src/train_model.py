"""Train churn model on the features table.

Uses sklearn's HistGradientBoosting (no external native deps on macOS).

Writes:
  models/churn_model.pkl   — trained model
  models/feature_list.txt  — feature names used at training time
  data/predictions.csv     — per-customer churn_probability
  models/metrics.txt       — quick model metrics
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

FEATURE_COLS = [
    "rate_gap_bps",
    "current_balance",
    "remaining_term_months",
    "ltv",
    "credit_score",
    "tenure_years",
    "num_products",
    "deposit_balance",
    "checking_flag",
    "savings_flag",
    "delinquency_flag",
    "digital_events_total_30d",
    "digital_events_total_90d",
    "digital_activity_change_pct",
    "refinance_page_view_30d",
    "refinance_calculator_use_30d",
    "payoff_info_view_30d",
    "rate_page_view_30d",
    "mortgage_page_view_30d",
    "statement_download_30d",
    "appointment_request_30d",
    "call_count_90d",
    "refinance_intent_flag",
    "competitor_shopping_flag",
    "payoff_intent_flag",
    "avg_call_sentiment",
    "max_rate_sensitivity",
    "branch_visits_90d",
    "mortgage_discussion_count_90d",
    "rate_discussion_flag_90d",
    "refinance_discussion_flag_90d",
]


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))

    # Ensure every training feature exists (some event types may be absent)
    for col in FEATURE_COLS:
        if col not in features.columns:
            features[col] = 0

    X = features[FEATURE_COLS].fillna(0)
    y = features["churned_within_90d"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42,
    )

    # class_weight='balanced' helps the base learner separate the ~2%
    # positive class; isotonic calibration then maps the resulting scores
    # back to actual empirical churn rates so predictions are not
    # over-confident.
    base = HistGradientBoostingClassifier(
        max_iter=250,
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=25,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X_train, y_train)

    prob_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob_test)
    ap = average_precision_score(y_test, prob_test)

    # Top-decile lift
    order = np.argsort(-prob_test)
    top_n = max(1, len(prob_test) // 10)
    top_idx = order[:top_n]
    top_decile_rate = y_test.values[top_idx].mean()
    base_rate = y_test.mean()
    lift = top_decile_rate / base_rate if base_rate > 0 else float("nan")

    # Precision/recall at a working threshold
    thresh = 0.35
    y_pred = (prob_test >= thresh).astype(int)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    metrics = {
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "base_rate": round(float(base_rate), 4),
        "top_decile_churn_rate": round(float(top_decile_rate), 4),
        "top_decile_lift_x": round(float(lift), 2),
        "precision_at_threshold_0.35": round(prec, 4),
        "recall_at_threshold_0.35": round(rec, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Score full population for the app
    prob_all = model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({
        "customer_id": features["customer_id"],
        "churn_probability": prob_all.round(4),
    })
    predictions.to_csv(os.path.join(DATA_DIR, "predictions.csv"), index=False)

    with open(os.path.join(MODEL_DIR, "churn_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "feature_list.txt"), "w") as f:
        f.write("\n".join(FEATURE_COLS))
    with open(os.path.join(MODEL_DIR, "metrics.txt"), "w") as f:
        f.write(json.dumps(metrics, indent=2))

    # Permutation importance on the test set — used as explainability fallback
    perm = permutation_importance(model, X_test, y_test, n_repeats=3,
                                  random_state=42, scoring="roc_auc", n_jobs=1)
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": perm.importances_mean.round(5),
    }).sort_values("importance", ascending=False)
    importance.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

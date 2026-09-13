"""Build the model-ready per-customer feature table.

Reads the six CSVs produced by generate_data.py, rolls up events
into 30- and 90-day windows, joins to customer/mortgage attributes,
and writes data/features.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
AS_OF_DATE = pd.Timestamp("2026-09-11")
MARKET_RATE_PCT = 5.4


def _digital_features(digital: pd.DataFrame) -> pd.DataFrame:
    digital["timestamp"] = pd.to_datetime(digital["timestamp"])
    digital["days_ago"] = (AS_OF_DATE - digital["timestamp"]).dt.days
    d30 = digital[digital["days_ago"] <= 30]
    d90 = digital

    def _pivot(df, suffix):
        return (
            df.pivot_table(index="customer_id", columns="event_type",
                           values="days_ago", aggfunc="count", fill_value=0)
            .add_suffix(suffix)
        )

    p30 = _pivot(d30, "_30d")
    p90 = _pivot(d90, "_90d")
    total_30 = d30.groupby("customer_id").size().rename("digital_events_total_30d")
    total_90 = d90.groupby("customer_id").size().rename("digital_events_total_90d")

    out = p30.join(p90, how="outer").join(total_30, how="outer").join(total_90, how="outer")
    out = out.fillna(0)
    # Activity change: 30d rate vs prior 60d rate
    d31_90 = digital[(digital["days_ago"] > 30) & (digital["days_ago"] <= 90)]
    prior_60 = d31_90.groupby("customer_id").size().rename("prior_60d")
    out = out.join(prior_60, how="outer").fillna(0)
    out["digital_activity_change_pct"] = np.where(
        out["prior_60d"] > 0,
        (out["digital_events_total_30d"] * 2 - out["prior_60d"]) / out["prior_60d"],
        0.0,
    )
    out = out.drop(columns=["prior_60d"])
    return out.reset_index()


def _call_features(calls: pd.DataFrame) -> pd.DataFrame:
    if calls.empty:
        return pd.DataFrame(columns=[
            "customer_id", "call_count_90d", "refinance_intent_flag",
            "competitor_shopping_flag", "payoff_intent_flag",
            "avg_call_sentiment", "max_rate_sensitivity",
        ])
    g = calls.groupby("customer_id")
    return pd.DataFrame({
        "customer_id": g.size().index,
        "call_count_90d": g.size().values,
        "refinance_intent_flag": g["refinance_intent"].max().astype(int).values,
        "competitor_shopping_flag": g["competitor_shopping"].max().astype(int).values,
        "payoff_intent_flag": g["payoff_intent"].max().astype(int).values,
        "avg_call_sentiment": g["sentiment"].mean().round(3).values,
        "max_rate_sensitivity": g["rate_sensitivity"].max().round(3).values,
    })


def _branch_features(branch: pd.DataFrame) -> pd.DataFrame:
    if branch.empty:
        return pd.DataFrame(columns=[
            "customer_id", "branch_visits_90d", "mortgage_discussion_count_90d",
            "rate_discussion_flag_90d", "refinance_discussion_flag_90d",
        ])
    g = branch.groupby("customer_id")
    return pd.DataFrame({
        "customer_id": g.size().index,
        "branch_visits_90d": g.size().values,
        "mortgage_discussion_count_90d": g["mortgage_discussion_flag"].sum().values,
        "rate_discussion_flag_90d": g["rate_discussion_flag"].max().values,
        "refinance_discussion_flag_90d": g["refinancing_discussion_flag"].max().values,
    })


def build() -> pd.DataFrame:
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    mortgages = pd.read_csv(os.path.join(DATA_DIR, "mortgages.csv"))
    digital = pd.read_csv(os.path.join(DATA_DIR, "digital_events.csv"))
    calls = pd.read_csv(os.path.join(DATA_DIR, "call_interactions.csv"))
    branch = pd.read_csv(os.path.join(DATA_DIR, "branch_interactions.csv"))
    outcomes = pd.read_csv(os.path.join(DATA_DIR, "outcomes.csv"))

    features = customers.merge(mortgages, on="customer_id", how="left")
    features["rate_gap_bps"] = ((features["mortgage_rate"] - MARKET_RATE_PCT) * 100).round().astype(int)

    features = features.merge(_digital_features(digital), on="customer_id", how="left")
    features = features.merge(_call_features(calls), on="customer_id", how="left")
    features = features.merge(_branch_features(branch), on="customer_id", how="left")
    features = features.merge(outcomes, on="customer_id", how="left")

    # Fill NAs for customers with no events
    fill_zero_cols = [c for c in features.columns
                      if c.endswith(("_30d", "_90d", "_flag"))
                      or c in ("call_count_90d", "avg_call_sentiment",
                               "max_rate_sensitivity", "digital_activity_change_pct")]
    features[fill_zero_cols] = features[fill_zero_cols].fillna(0)
    return features


def main() -> None:
    df = build()
    out_path = os.path.join(DATA_DIR, "features.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} feature rows to {out_path}. Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()

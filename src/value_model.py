"""Economic value-at-risk and expected intervention value.

All assumptions are configurable constants at the top of the file so
the deck can defend them as scenario ranges rather than point claims.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# --- Configurable assumptions (kept in sync with the Slide 2 financial model) ---
# Original net interest margin the bank earns on the mortgage book annually.
# Retail mortgage NIM is typically 100-200 bps; we assume a conservative 150.
NET_INTEREST_MARGIN = 0.015
# Share of NIM kept after any retention rate concession. Retention often
# requires giving up 30-50% of NIM to match a competitor rate; the effective
# NIM on retained balance is therefore lower than the book NIM.
NIM_KEPT_AFTER_CONCESSION = 0.65
# Effective NIM used for all value calculations.
EFFECTIVE_NIM = NET_INTEREST_MARGIN * NIM_KEPT_AFTER_CONCESSION  # 0.00975
# Average remaining life of a retained mortgage, capped conservatively below
# the contractual remaining term to reflect prepayment and moves.
LIFE_CAP_YEARS = 7.0
# Simple discount factor applied to the NIM stream (assume ~0.9 blended).
DISCOUNT_FACTOR = 0.9
# Additional relationship value uplift when the customer holds deposits and
# cross-sold products — churn of the mortgage often triggers wider attrition.
RELATIONSHIP_MULTIPLIER = 1.15

# Intervention economics — tuned to the Slide 2 base case
INTERVENTION_SUCCESS_PROB = 0.22
INTERVENTION_COST = 200.0  # banker time + offer cost per outreach

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def economic_value_if_lost(current_balance: np.ndarray,
                           remaining_term_months: np.ndarray) -> np.ndarray:
    """Dollar value the bank forgoes if this mortgage refinances away.

    Uses the *effective* NIM (post-concession) to match the Slide 2 model —
    retaining a customer typically requires giving up part of the NIM.
    """
    life_years = np.minimum(remaining_term_months / 12.0, LIFE_CAP_YEARS)
    return (current_balance
            * EFFECTIVE_NIM
            * life_years
            * DISCOUNT_FACTOR
            * RELATIONSHIP_MULTIPLIER)


def value_at_risk(churn_prob: np.ndarray, economic_value: np.ndarray) -> np.ndarray:
    return churn_prob * economic_value


def expected_intervention_value(voar: np.ndarray) -> np.ndarray:
    return INTERVENTION_SUCCESS_PROB * voar - INTERVENTION_COST


def enrich(features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(predictions, on="customer_id", how="left")
    df["churn_probability"] = df["churn_probability"].fillna(0)
    df["economic_value_if_lost"] = economic_value_if_lost(
        df["current_balance"].values, df["remaining_term_months"].values,
    ).round(0)
    df["value_at_risk"] = value_at_risk(
        df["churn_probability"].values, df["economic_value_if_lost"].values,
    ).round(0)
    df["expected_intervention_value"] = expected_intervention_value(
        df["value_at_risk"].values,
    ).round(0)
    return df


def main() -> None:
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    predictions = pd.read_csv(os.path.join(DATA_DIR, "predictions.csv"))
    enriched = enrich(features, predictions)
    out_path = os.path.join(DATA_DIR, "scored_customers.csv")
    enriched.to_csv(out_path, index=False)
    total_voar = enriched["value_at_risk"].sum()
    at_risk = (enriched["churn_probability"] >= 0.15).sum()
    print(f"Wrote {len(enriched)} scored customers to {out_path}")
    print(f"Total value at risk: ${total_voar:,.0f}")
    print(f"Customers at elevated risk (>=15%): {at_risk}")


if __name__ == "__main__":
    main()

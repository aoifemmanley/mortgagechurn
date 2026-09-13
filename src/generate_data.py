"""Generate synthetic bank data for the mortgage retention prototype.

Produces six CSVs in data/ modelled on the case's data sources:
  customers.csv, mortgages.csv, digital_events.csv,
  call_interactions.csv, branch_interactions.csv, outcomes.csv

A latent per-customer "intent" variable drives both event frequencies
and the 90-day churn outcome, which produces realistic correlation
without label leakage. Substantial noise is added so a downstream model
is informative but not trivially perfect.
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
N_CUSTOMERS = 10_000
AS_OF_DATE = pd.Timestamp("2026-09-11")
LOOKBACK_DAYS = 90
MARKET_RATE_PCT = 5.4

STATES = [
    "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_customers_and_mortgages(rng: np.random.Generator, fake: Faker):
    n = N_CUSTOMERS
    customer_ids = [f"C{100000 + i:07d}" for i in range(n)]
    names = [fake.name() for _ in range(n)]
    age_bands = rng.choice(
        ["25-34", "35-44", "45-54", "55-64", "65+"], size=n,
        p=[0.15, 0.30, 0.25, 0.20, 0.10],
    )
    tenure_years = rng.integers(1, 21, size=n)
    states = rng.choice(STATES, size=n)
    credit_score = np.clip(rng.normal(720, 45, size=n), 580, 820).astype(int)
    num_products = rng.choice([1, 2, 3, 4], size=n, p=[0.35, 0.35, 0.20, 0.10])
    checking_flag = (num_products >= 2).astype(int)
    savings_flag = (num_products >= 3).astype(int)
    deposit_balance = np.clip(rng.lognormal(9.0, 1.2, size=n), 100, 500_000).round(0)

    mortgage_ids = [f"M{100000 + i:07d}" for i in range(n)]
    orig_days_ago = rng.integers(365, 365 * 7, size=n)
    orig_dates = [(AS_OF_DATE - timedelta(days=int(d))).date() for d in orig_days_ago]
    original_balance = np.clip(rng.lognormal(12.7, 0.5, size=n), 80_000, 2_500_000)
    original_balance = (original_balance / 1000).round(0) * 1000
    # Rate distribution — mean 5.5%, spread so ~35% land meaningfully above market 5.4%
    mortgage_rate = np.clip(rng.normal(5.5, 1.1, size=n), 2.8, 8.0).round(3)
    original_term_years = rng.choice([15, 30], size=n, p=[0.15, 0.85])
    remaining_term_months = np.clip(
        original_term_years * 12 - (orig_days_ago / 30.44).astype(int),
        12, 360,
    ).astype(int)

    years_paid = orig_days_ago / 365.25
    paydown_pct = np.clip(years_paid / original_term_years * 0.35, 0, 0.55)
    current_balance = (original_balance * (1 - paydown_pct) / 1000).round(0) * 1000

    r_monthly = mortgage_rate / 100 / 12
    n_months = original_term_years * 12
    monthly_payment = (
        original_balance * r_monthly * (1 + r_monthly) ** n_months
        / ((1 + r_monthly) ** n_months - 1)
    ).round(2)

    appreciation = rng.uniform(1.0, 1.6, size=n)
    property_value = (original_balance / 0.8 * appreciation / 1000).round(0) * 1000
    ltv = np.clip(current_balance / property_value, 0.05, 1.1).round(3)
    delinquency_flag = (rng.random(size=n) < 0.02).astype(int)

    relationship_value_estimate = (
        current_balance * 0.015 * (remaining_term_months / 12)
        + deposit_balance * 0.002
    ).round(0)

    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "name": names,
        "age_band": age_bands,
        "tenure_years": tenure_years,
        "state": states,
        "credit_score": credit_score,
        "num_products": num_products,
        "checking_flag": checking_flag,
        "savings_flag": savings_flag,
        "deposit_balance": deposit_balance,
        "relationship_value_estimate": relationship_value_estimate,
    })

    mortgages_df = pd.DataFrame({
        "mortgage_id": mortgage_ids,
        "customer_id": customer_ids,
        "origination_date": orig_dates,
        "original_balance": original_balance.astype(int),
        "current_balance": current_balance.astype(int),
        "mortgage_rate": mortgage_rate,
        "original_term_years": original_term_years,
        "remaining_term_months": remaining_term_months,
        "monthly_payment": monthly_payment,
        "ltv": ltv,
        "property_value": property_value.astype(int),
        "delinquency_flag": delinquency_flag,
    })

    return customers_df, mortgages_df, mortgage_rate


def build_propensity(rng, mortgage_rate, tenure_years):
    """Latent refinance propensity in [0, 1]. Not directly observed by the model."""
    rate_gap_pct = mortgage_rate - MARKET_RATE_PCT
    shopper_trait = rng.beta(1.5, 6.0, size=len(mortgage_rate))
    tenure_pull = -0.03 * (tenure_years - 5) / 15
    logit = 1.4 * rate_gap_pct + 3.0 * shopper_trait + tenure_pull - 1.2
    return _sigmoid(logit)


def build_digital_events(rng, customer_ids, propensity):
    baseline = {
        "login": 12.0,
        "mortgage_page_view": 1.0,
        "rate_page_view": 0.3,
        "refinance_page_view": 0.2,
        "refinance_calculator_use": 0.05,
        "payoff_info_view": 0.05,
        "statement_download": 1.5,
        "faq_view": 0.3,
        "appointment_request": 0.02,
    }
    prop_mult = {
        "login": 0.5,
        "mortgage_page_view": 3.0,
        "rate_page_view": 6.0,
        "refinance_page_view": 8.0,
        "refinance_calculator_use": 5.0,
        "payoff_info_view": 4.0,
        "statement_download": 2.0,
        "faq_view": 2.0,
        "appointment_request": 3.0,
    }
    n = len(customer_ids)
    cid_arr = np.array(customer_ids)
    frames = []
    refi_biased = {"refinance_page_view", "refinance_calculator_use",
                   "payoff_info_view", "rate_page_view"}
    for event_type, base in baseline.items():
        lam = base + prop_mult[event_type] * propensity
        counts = rng.poisson(lam)
        idx = np.repeat(np.arange(n), counts)
        if len(idx) == 0:
            continue
        if event_type in refi_biased:
            days_ago = rng.beta(1.5, 3.0, size=len(idx)) * LOOKBACK_DAYS
        else:
            days_ago = rng.uniform(0, LOOKBACK_DAYS, size=len(idx))
        timestamps = [AS_OF_DATE - timedelta(days=float(d)) for d in days_ago]
        channels = rng.choice(["web", "mobile"], size=len(idx), p=[0.55, 0.45])
        frames.append(pd.DataFrame({
            "customer_id": cid_arr[idx],
            "timestamp": timestamps,
            "channel": channels,
            "event_type": event_type,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["customer_id", "timestamp"], inplace=True)
    return df


CALL_TEMPLATES_REFI = [
    "Customer asked what our current 30-year rates look like and mentioned they saw {rate}% at {competitor}.",
    "Called about refinancing options — wanted to know break-even on closing costs against a {rate}% offer.",
    "Asked about the process to pay off the loan early and whether there are any prepayment penalties.",
    "Wants to speak to a mortgage specialist about refinancing. Mentioned rate shopping.",
    "Inquired about payoff amount and where to send funds. Sounded like they may be refinancing elsewhere.",
]
CALL_TEMPLATES_NEUTRAL = [
    "Requested a copy of the most recent mortgage statement for tax purposes.",
    "Asked how to set up autopay for the mortgage.",
    "Question about escrow analysis and property tax increase.",
    "Confirmed next payment due date.",
    "General question about online banking access.",
]


def build_call_interactions(rng, customer_ids, propensity):
    n = len(customer_ids)
    had_call = rng.random(n) < (0.20 + 0.30 * propensity)
    rows = []
    for i in np.where(had_call)[0]:
        n_calls = int(rng.integers(1, 3))
        for _ in range(n_calls):
            is_refi_call = rng.random() < (0.10 + 0.70 * propensity[i])
            if is_refi_call:
                tmpl = rng.choice(CALL_TEMPLATES_REFI)
                rate = round(MARKET_RATE_PCT + rng.uniform(-0.3, 0.2), 2)
                competitor = rng.choice(["a competitor", "Rocket", "Chase", "an online lender"])
                snippet = tmpl.format(rate=rate, competitor=competitor)
                reason_code = "MORTGAGE_INQUIRY"
                refinance_intent = True
                competitor_shopping = bool(rng.random() < 0.7)
                payoff_intent = "pay off" in snippet.lower() or "payoff" in snippet.lower()
                rate_sensitivity = float(np.clip(rng.normal(0.75, 0.15), 0.3, 1.0))
                sentiment = float(np.clip(rng.normal(-0.1, 0.3), -1.0, 1.0))
            else:
                snippet = str(rng.choice(CALL_TEMPLATES_NEUTRAL))
                reason_code = str(rng.choice(["STATEMENT", "PAYMENT", "ESCROW", "GENERAL"]))
                refinance_intent = False
                competitor_shopping = False
                payoff_intent = False
                rate_sensitivity = float(np.clip(rng.normal(0.15, 0.15), 0.0, 1.0))
                sentiment = float(np.clip(rng.normal(0.2, 0.3), -1.0, 1.0))
            days_ago = rng.uniform(0, LOOKBACK_DAYS)
            rows.append({
                "customer_id": customer_ids[i],
                "timestamp": AS_OF_DATE - timedelta(days=float(days_ago)),
                "reason_code": reason_code,
                "duration_sec": int(rng.integers(60, 900)),
                "transcript_snippet": snippet,
                "refinance_intent": bool(refinance_intent),
                "competitor_shopping": bool(competitor_shopping),
                "payoff_intent": bool(payoff_intent),
                "rate_sensitivity": round(rate_sensitivity, 2),
                "sentiment": round(sentiment, 2),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["customer_id", "timestamp"], inplace=True)
    return df


BRANCH_NOTES_REFI = [
    "Client came in asking about refinancing options; mentioned they are shopping around.",
    "Wanted to understand payoff process for the mortgage. Discussed remaining term.",
    "Discussion about current rates vs. their existing mortgage rate.",
]
BRANCH_NOTES_NEUTRAL = [
    "Routine deposit; brief chat about upcoming home renovation.",
    "Client came in to update address on file.",
    "Signed paperwork for savings account.",
    "Discussed CD options.",
]


def build_branch_interactions(rng, customer_ids, propensity):
    n = len(customer_ids)
    had_visit = rng.random(n) < (0.10 + 0.15 * propensity)
    rows = []
    for i in np.where(had_visit)[0]:
        is_mortgage_visit = rng.random() < (0.05 + 0.55 * propensity[i])
        if is_mortgage_visit:
            note = str(rng.choice(BRANCH_NOTES_REFI))
            mortgage_discussion = 1
            rate_discussion = 1 if "rate" in note.lower() else int(rng.random() < 0.4)
            refi_discussion = 1 if ("refinanc" in note.lower() or "payoff" in note.lower()) else 0
            interaction_type = "MORTGAGE_CONSULT"
        else:
            note = str(rng.choice(BRANCH_NOTES_NEUTRAL))
            mortgage_discussion = 0
            rate_discussion = 0
            refi_discussion = 0
            interaction_type = str(rng.choice(["TELLER", "ACCOUNT_SERVICE", "GENERAL"]))
        days_ago = rng.uniform(0, LOOKBACK_DAYS)
        rows.append({
            "customer_id": customer_ids[i],
            "timestamp": AS_OF_DATE - timedelta(days=float(days_ago)),
            "branch_id": f"B{int(rng.integers(1, 120)):03d}",
            "interaction_type": interaction_type,
            "banker_note": note,
            "mortgage_discussion_flag": mortgage_discussion,
            "rate_discussion_flag": rate_discussion,
            "refinancing_discussion_flag": refi_discussion,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["customer_id", "timestamp"], inplace=True)
    return df


def build_outcomes(rng, customers_df, mortgage_rate, propensity,
                   digital_events, call_df, branch_df):
    n = len(customers_df)
    rate_gap_pct = mortgage_rate - MARKET_RATE_PCT

    calc_uses_s = digital_events[
        digital_events["event_type"] == "refinance_calculator_use"
    ].groupby("customer_id").size()
    payoff_s = digital_events[
        digital_events["event_type"] == "payoff_info_view"
    ].groupby("customer_id").size()
    call_refi_s = (call_df[call_df["refinance_intent"]].groupby("customer_id").size()
                   if not call_df.empty else pd.Series(dtype=int))
    branch_refi_s = (branch_df[branch_df["refinancing_discussion_flag"] == 1]
                     .groupby("customer_id").size()
                     if not branch_df.empty else pd.Series(dtype=int))

    def _col(s):
        return customers_df["customer_id"].map(s).fillna(0).astype(int).values

    calc_uses = _col(calc_uses_s)
    payoff = _col(payoff_s)
    call_refi = _col(call_refi_s)
    branch_refi = _col(branch_refi_s)

    # Base logit tuned so overall churn rate lands near 2% and the top
    # decile of latent probability is high enough (~10-15%) that isotonic
    # calibration downstream can output realistic ~30-40% predictions for
    # the strongest-signal customers, without producing implausible ~99%
    # scores.
    logit = (
        -8.0
        + 2.0 * propensity
        + 0.6 * np.clip(rate_gap_pct, -1, 3)
        + 0.30 * np.minimum(calc_uses, 5)
        + 0.25 * np.minimum(payoff, 4)
        + 1.10 * np.minimum(call_refi, 3)
        + 1.10 * np.minimum(branch_refi, 2)
        + rng.normal(0, 0.5, size=n)
    )
    prob = _sigmoid(logit)
    churned = (rng.random(n) < prob).astype(int)
    churn_date = [
        (AS_OF_DATE + timedelta(days=int(rng.integers(1, 90)))).date() if c else None
        for c in churned
    ]
    return pd.DataFrame({
        "customer_id": customers_df["customer_id"],
        "churned_within_90d": churned,
        "churn_date": churn_date,
    })


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    fake = Faker()
    Faker.seed(SEED)

    print("Building customers and mortgages...")
    customers_df, mortgages_df, mortgage_rate = build_customers_and_mortgages(rng, fake)

    propensity = build_propensity(rng, mortgage_rate, customers_df["tenure_years"].values)

    print("Building digital events...")
    digital_events = build_digital_events(rng, customers_df["customer_id"].tolist(), propensity)

    print("Building call interactions...")
    call_df = build_call_interactions(rng, customers_df["customer_id"].tolist(), propensity)

    print("Building branch interactions...")
    branch_df = build_branch_interactions(rng, customers_df["customer_id"].tolist(), propensity)

    print("Building outcomes...")
    outcomes_df = build_outcomes(
        rng, customers_df, mortgage_rate, propensity,
        digital_events, call_df, branch_df,
    )

    customers_df.to_csv(os.path.join(DATA_DIR, "customers.csv"), index=False)
    mortgages_df.to_csv(os.path.join(DATA_DIR, "mortgages.csv"), index=False)
    digital_events.to_csv(os.path.join(DATA_DIR, "digital_events.csv"), index=False)
    call_df.to_csv(os.path.join(DATA_DIR, "call_interactions.csv"), index=False)
    branch_df.to_csv(os.path.join(DATA_DIR, "branch_interactions.csv"), index=False)
    outcomes_df.to_csv(os.path.join(DATA_DIR, "outcomes.csv"), index=False)

    print(
        f"Done. customers={len(customers_df)}, mortgages={len(mortgages_df)}, "
        f"digital_events={len(digital_events)}, calls={len(call_df)}, "
        f"branch={len(branch_df)}, churn_rate={outcomes_df['churned_within_90d'].mean():.3%}"
    )


if __name__ == "__main__":
    main()

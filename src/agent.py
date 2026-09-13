"""Simulated next-best-action agent.

Deterministic templates keyed off the customer's top risk signals.
Reads as if an LLM authored the reasoning and outreach draft, but is
fully deterministic so the prototype needs no external API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


APPROVED_OFFERS = {
    "rate_review": "Rate review with a mortgage specialist",
    "in_house_refi": "In-house refinance with waived origination fee",
    "relationship_pricing": "Relationship-pricing rate discount (subject to eligibility)",
    "retention_specialist": "Warm transfer to retention specialist",
    "monitor": "No offer — monitor",
}


@dataclass
class SignalDescription:
    key: str
    label: str
    weight: float  # for ordering


def build_signals(row: Dict[str, Any]) -> List[SignalDescription]:
    """Human-readable signals ranked by quality for this customer.

    Ordering principle: signals from human interactions (call, branch)
    are strictly higher quality than web-behavior signals, and are shown
    first. Web signals still show up but never dominate the summary.
    """
    signals: List[SignalDescription] = []

    # --- Tier 1: explicit human-interaction signals (highest quality) ---
    if int(row.get("refinance_intent_flag", 0) or 0) == 1:
        signals.append(SignalDescription(
            "call_refi",
            "Discussed refinancing on a recent call to the contact center",
            weight=5.0,
        ))

    if int(row.get("competitor_shopping_flag", 0) or 0) == 1:
        signals.append(SignalDescription(
            "competitor",
            "Mentioned a competitor rate on a recent call",
            weight=4.5,
        ))

    if int(row.get("payoff_intent_flag", 0) or 0) == 1:
        signals.append(SignalDescription(
            "call_payoff",
            "Asked about the mortgage payoff process on a recent call",
            weight=4.0,
        ))

    if int(row.get("refinance_discussion_flag_90d", 0) or 0) == 1:
        signals.append(SignalDescription(
            "branch_refi",
            "Discussed refinancing during a recent branch visit",
            weight=3.5,
        ))

    if int(row.get("mortgage_discussion_count_90d", 0) or 0) >= 1:
        signals.append(SignalDescription(
            "branch_mortgage",
            "Raised the mortgage during a recent branch visit",
            weight=2.5,
        ))

    # --- Tier 2: rate context ---
    gap = int(row.get("rate_gap_bps", 0) or 0)
    if gap >= 25:
        # Cap weight so a huge rate gap alone doesn't dominate all other signals
        signals.append(SignalDescription(
            "rate_gap",
            f"Mortgage rate is {gap} bps above the current market rate",
            weight=min(3.0, 1.5 + gap / 200.0),
        ))

    # --- Tier 3: digital behavior (supporting evidence, capped) ---
    calc = int(row.get("refinance_calculator_use_30d", 0) or 0)
    if calc >= 1:
        n = min(calc, 5)
        label = f"Used the refinance calculator {n}× in the last 30 days"
        if calc > 5:
            label = f"Used the refinance calculator 5+ times in the last 30 days"
        signals.append(SignalDescription(
            "refi_calc", label, weight=1.6 + 0.15 * n,
        ))

    payoff = int(row.get("payoff_info_view_30d", 0) or 0)
    if payoff >= 1:
        n = min(payoff, 5)
        label = f"Viewed payoff information {n}× online in the last 30 days"
        if payoff > 5:
            label = "Viewed payoff information 5+ times in the last 30 days"
        signals.append(SignalDescription(
            "payoff_web", label, weight=1.4 + 0.15 * n,
        ))

    refi_pv = int(row.get("refinance_page_view_30d", 0) or 0)
    if refi_pv >= 2:
        n = min(refi_pv, 8)
        label = f"Viewed the refinance page {n}× in the last 30 days"
        if refi_pv > 8:
            label = "Viewed the refinance page 8+ times in the last 30 days"
        signals.append(SignalDescription(
            "refi_pv", label, weight=0.9 + 0.08 * n,
        ))

    change = float(row.get("digital_activity_change_pct", 0) or 0)
    if change >= 0.5:
        signals.append(SignalDescription(
            "activity_up",
            f"Digital activity is up {int(change * 100)}% vs. the prior 60 days",
            weight=0.7 + 0.3 * min(change, 2.0),
        ))

    signals.sort(key=lambda s: s.weight, reverse=True)
    return signals


def _first_name(name: str) -> str:
    return (name or "there").split()[0]


def recommend(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return recommended action, reasoning, offer, and drafted outreach."""
    signals = build_signals(row)
    top = signals[:4]
    prob = float(row.get("churn_probability", 0) or 0)
    voar = float(row.get("value_at_risk", 0) or 0)
    balance = float(row.get("current_balance", 0) or 0)
    rate = float(row.get("mortgage_rate", 0) or 0)
    gap = int(row.get("rate_gap_bps", 0) or 0)
    name = row.get("name", "there")
    signal_keys = {s.key for s in top}

    # Decision rules — pick urgency, channel, action, offer
    if prob >= 0.35 and voar >= 25_000 and (
        "call_refi" in signal_keys or "call_payoff" in signal_keys
        or "competitor" in signal_keys or "branch_refi" in signal_keys
    ):
        urgency = "high"
        channel = "phone"
        action = "Proactive call from mortgage retention specialist within 48 hours"
        offer_key = "retention_specialist"
    elif prob >= 0.25 and voar >= 10_000 and gap >= 40:
        urgency = "high"
        channel = "phone"
        action = "Rate review call from mortgage specialist within 5 business days"
        offer_key = "rate_review"
    elif prob >= 0.15 and voar >= 5_000:
        urgency = "medium"
        channel = "email"
        action = "Personalized rate-review email with option to book a mortgage specialist"
        offer_key = "in_house_refi"
    elif prob >= 0.08:
        urgency = "low"
        channel = "email"
        action = "Nurture email highlighting relationship pricing and in-house refinance options"
        offer_key = "relationship_pricing"
    else:
        urgency = "low"
        channel = "monitor"
        action = "No outreach — continue monitoring for new signals"
        offer_key = "monitor"

    offer = APPROVED_OFFERS[offer_key]

    # Reasoning — chain top signals into a natural sentence
    if top:
        reason_bullets = ", ".join(s.label.lower() for s in top[:3])
        reasoning = (
            f"Model estimates a {prob:.0%} probability that this customer refinances "
            f"in the next 90 days. The strongest signals are: {reason_bullets}. "
            f"Estimated value at risk if lost is ${voar:,.0f}."
        )
    else:
        reasoning = (
            f"Model estimates a {prob:.0%} probability of refinancing in the next 90 days. "
            f"Estimated value at risk if lost is ${voar:,.0f}."
        )

    # Drafted outreach — template varies by channel
    first = _first_name(str(name))
    if channel == "phone":
        draft = (
            f"Call script — {first}\n\n"
            f"Hi {first}, this is [banker name] from your bank's mortgage team. "
            f"I wanted to reach out because I know rates have moved recently and I "
            f"wanted to make sure you're aware of the options we have for existing "
            f"customers. Your current rate is {rate:.2f}% on a balance of "
            f"${balance:,.0f}, and I'd love to walk you through where we can meet "
            f"you before you consider going elsewhere. Do you have a few minutes "
            f"this week?"
        )
    elif channel == "email":
        draft = (
            f"Subject: A quick rate review for your mortgage, {first}\n\n"
            f"Hi {first},\n\n"
            f"I wanted to reach out personally. Your current mortgage rate is "
            f"{rate:.2f}%, and we've recently made options available for existing "
            f"customers that may be worth a closer look — including {offer.lower()}. "
            f"If you'd like, I can set up a 15-minute call with a mortgage specialist "
            f"who can walk you through what's possible before you consider anything "
            f"outside the bank.\n\n"
            f"Best regards,\n[Your mortgage banker]"
        )
    else:
        draft = "No outreach drafted — customer is being monitored."

    return {
        "recommended_action": action,
        "urgency": urgency,
        "channel": channel,
        "suggested_offer": offer,
        "reasoning": reasoning,
        "draft_outreach": draft,
        "top_signals": [s.label for s in top],
    }

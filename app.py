"""Mortgage Retention Command Center — Streamlit prototype.

Palantir-Workshop-styled retention operations UI. All data synthetic.
LLM outputs (call intent, agent recommendation, drafted outreach) are
generated deterministically from templates — labelled as such in
the UI where relevant.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.agent import recommend
from src.value_model import (EFFECTIVE_NIM, INTERVENTION_COST,
                             INTERVENTION_SUCCESS_PROB, NIM_KEPT_AFTER_CONCESSION)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

AS_OF = pd.Timestamp("2026-09-11")
MARKET_RATE_PCT = 5.4

st.set_page_config(
    page_title="Mortgage Retention Command Center",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Palantir-Workshop-inspired dark theme + component styling
# =============================================================================
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"], [class*="st-"], .stMarkdown, .stText, .stButton, .stMetric {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-feature-settings: "cv02","cv03","cv04","cv11";
  }

  .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1700px !important;
  }

  /* Hide default streamlit chrome */
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

  /* Top brand bar */
  .ns-brand {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 18px 0;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 22px;
  }
  .ns-brand-title {
    display: flex; align-items: center; gap: 12px;
    font-size: 15px; font-weight: 600; color: #E4E7EB;
    letter-spacing: -0.01em;
  }
  .ns-brand-mark {
    width: 22px; height: 22px; border-radius: 4px;
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
    display: inline-flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 12px;
  }
  .ns-brand-meta {
    font-size: 12px; color: #94A3B8; letter-spacing: 0.02em;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #141C2E;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 18px 20px 16px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
  }
  [data-testid="stMetricLabel"] p {
    font-size: 11px !important; font-weight: 500 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: #94A3B8 !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 26px !important; font-weight: 700 !important;
    color: #F1F5F9 !important; line-height: 1.15 !important;
    font-feature-settings: "tnum";
  }
  [data-testid="stMetricDelta"] {
    font-size: 12px !important; color: #94A3B8 !important;
    font-weight: 500 !important;
  }

  /* Section headers */
  .ns-section-h {
    font-size: 11px; font-weight: 600; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 22px 0 10px 0;
  }
  .ns-page-title {
    font-size: 26px; font-weight: 700; color: #F1F5F9;
    letter-spacing: -0.02em; margin: 4px 0 2px 0;
  }
  .ns-page-sub {
    font-size: 13px; color: #94A3B8; margin-bottom: 4px;
  }

  /* Card */
  .ns-card {
    background: #141C2E;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 14px;
  }
  .ns-card h4 {
    font-size: 11px; font-weight: 600; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0 0 12px 0;
  }
  .ns-card-lg-value {
    font-size: 38px; font-weight: 700; color: #F1F5F9;
    line-height: 1.05; letter-spacing: -0.02em;
    font-feature-settings: "tnum";
  }
  .ns-card-lg-sub { font-size: 12px; color: #94A3B8; margin-top: 6px; }
  .ns-card-hi { color: #F87171 !important; }
  .ns-card-mid { color: #FBBF24 !important; }
  .ns-card-lo { color: #94A3B8 !important; }

  /* Badges */
  .ns-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 10px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .ns-badge-high { background: rgba(248,113,113,0.14); color: #F87171; border: 1px solid rgba(248,113,113,0.3); }
  .ns-badge-medium { background: rgba(251,191,36,0.12); color: #FBBF24; border: 1px solid rgba(251,191,36,0.28); }
  .ns-badge-low { background: rgba(148,163,184,0.12); color: #94A3B8; border: 1px solid rgba(148,163,184,0.28); }
  .ns-badge-info { background: rgba(59,130,246,0.14); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
  .ns-badge-success { background: rgba(52,211,153,0.14); color: #34D399; border: 1px solid rgba(52,211,153,0.3); }
  .ns-badge-neutral { background: rgba(148,163,184,0.08); color: #64748B; border: 1px solid rgba(148,163,184,0.18); }

  /* Signal list */
  .ns-signal { padding: 8px 0; border-bottom: 1px solid #1E293B; font-size: 13px; color: #CBD5E1; display: flex; gap: 10px; }
  .ns-signal:last-child { border-bottom: none; }
  .ns-signal-dot { color: #F87171; font-size: 8px; margin-top: 6px; }

  /* Timeline */
  .ns-tl-item {
    display: grid; grid-template-columns: 100px 90px 1fr;
    gap: 12px; padding: 8px 0; font-size: 12.5px;
    border-bottom: 1px solid #1E293B; color: #CBD5E1;
  }
  .ns-tl-item:last-child { border-bottom: none; }
  .ns-tl-when { color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
  .ns-tl-kind { color: #94A3B8; }
  .ns-tl-detail { color: #E4E7EB; }

  /* Reasoning block */
  .ns-reasoning {
    background: rgba(59,130,246,0.06);
    border: 1px solid rgba(59,130,246,0.22);
    border-left: 3px solid #3B82F6;
    padding: 12px 14px; border-radius: 6px;
    font-size: 13px; color: #DBEAFE; line-height: 1.55;
    margin: 8px 0 14px 0;
  }

  /* Draft outreach text area */
  textarea {
    background: #0B1220 !important;
    border: 1px solid #1E293B !important;
    color: #E4E7EB !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
  }

  /* Buttons */
  .stButton > button {
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border: 1px solid #1E293B !important;
    background: #141C2E !important;
    color: #E4E7EB !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    background: #1E293B !important;
    border-color: #334155 !important;
  }
  .stButton > button[kind="primary"] {
    background: #3B82F6 !important;
    border-color: #2563EB !important;
    color: white !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #2563EB !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0B1220 !important;
    border-right: 1px solid #1E293B;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 11px !important; font-weight: 600 !important;
    color: #94A3B8 !important; text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] {
    border: 1px solid #1E293B; border-radius: 8px;
    overflow: hidden;
  }

  /* Numeric feel */
  .ns-mono { font-family: 'JetBrains Mono', 'SF Mono', monospace; }

  /* Small breadcrumb */
  .ns-crumb {
    font-size: 12px; color: #64748B; margin-bottom: 6px;
    display: flex; align-items: center; gap: 6px;
  }
  .ns-crumb a { color: #94A3B8; text-decoration: none; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    scored = pd.read_csv(os.path.join(DATA_DIR, "scored_customers.csv"))
    digital = pd.read_csv(os.path.join(DATA_DIR, "digital_events.csv"),
                          parse_dates=["timestamp"])
    calls = pd.read_csv(os.path.join(DATA_DIR, "call_interactions.csv"),
                        parse_dates=["timestamp"])
    branch = pd.read_csv(os.path.join(DATA_DIR, "branch_interactions.csv"),
                         parse_dates=["timestamp"])
    return scored, digital, calls, branch


# ---------- Formatting helpers ----------

def fmt_dollars(v: float) -> str:
    v = float(v)
    if v >= 1e9: return f"${v/1e9:,.2f}B"
    if v >= 1e6: return f"${v/1e6:,.1f}M"
    if v >= 1e3: return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"


def urgency_class(prob: float) -> str:
    if prob >= 0.25: return "high"
    if prob >= 0.10: return "medium"
    return "low"


def risk_color_class(prob: float) -> str:
    if prob >= 0.25: return "ns-card-hi"
    if prob >= 0.10: return "ns-card-mid"
    return "ns-card-lo"


def render_brand_bar(right_meta: str):
    st.markdown(f"""
    <div class="ns-brand">
      <div class="ns-brand-title">
        <span class="ns-brand-mark">◆</span>
        Mortgage Retention Command Center
      </div>
      <div class="ns-brand-meta">{right_meta}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------- Sidebar filters (only for command view) ----------

def sidebar_filters(scored):
    st.sidebar.markdown("### Filters")
    risk = st.sidebar.slider("Min churn risk", 0.00, 0.50, 0.10, 0.01,
                             help="Predicted 90-day refinance probability")
    balance = st.sidebar.slider("Min mortgage balance ($K)", 0, 1000, 100, 25) * 1000
    all_states = sorted(scored["state"].unique().tolist())
    states = st.sidebar.multiselect("State", all_states, default=all_states)
    sort_by = st.sidebar.selectbox(
        "Sort by",
        ["value_at_risk", "churn_probability", "expected_intervention_value", "current_balance"],
        format_func=lambda x: {
            "value_at_risk": "Value at Risk",
            "churn_probability": "Churn Risk",
            "expected_intervention_value": "Net Opportunity",
            "current_balance": "Mortgage Balance",
        }[x],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:11px;color:#64748B;line-height:1.5'>"
        "Prototype on synthetic data. Production MVP trains on the bank's "
        "5-year proprietary history."
        "</div>", unsafe_allow_html=True,
    )
    return risk, balance, states, sort_by


# =============================================================================
# Command Center View
# =============================================================================

def command_center_view(scored: pd.DataFrame):
    render_brand_bar(f"As of {AS_OF.date().isoformat()} · Market rate {MARKET_RATE_PCT:.2f}%")
    risk, balance, states, sort_by = sidebar_filters(scored)

    st.markdown('<div class="ns-page-title">Retention Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="ns-page-sub">Portfolio-wide view of at-risk mortgage customers, prioritized by economic value.</div>', unsafe_allow_html=True)

    # KPIs
    portfolio = scored["current_balance"].sum()
    at_risk_mask = scored["churn_probability"] >= risk
    at_risk = scored[at_risk_mask]
    balance_at_risk = at_risk["current_balance"].sum()
    econ_var = at_risk["value_at_risk"].sum()
    addressable = at_risk[at_risk["expected_intervention_value"] > 0]["expected_intervention_value"].sum()

    st.markdown('<div class="ns-section-h">Portfolio snapshot</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Portfolio", fmt_dollars(portfolio))
    k2.metric("At elevated risk", f"{len(at_risk):,}",
              delta=f"{len(at_risk)/len(scored):.1%} of book", delta_color="off")
    k3.metric("Balance at risk", fmt_dollars(balance_at_risk))
    k4.metric("Value at risk", fmt_dollars(econ_var),
              help=f"Churn prob × economic value if lost. "
                   f"Effective NIM {EFFECTIVE_NIM:.2%} ({NIM_KEPT_AFTER_CONCESSION:.0%} of 150 bps kept "
                   f"after retention rate concession), ~7yr life.")
    k5.metric("Addressable opportunity", fmt_dollars(addressable),
              help=f"Positive expected value ({INTERVENTION_SUCCESS_PROB:.0%} success × VaR − ${INTERVENTION_COST:.0f})")

    # Filtered list
    filtered = scored[
        at_risk_mask
        & (scored["current_balance"] >= balance)
        & (scored["state"].isin(states))
    ].copy().sort_values(sort_by, ascending=False).head(200)

    if filtered.empty:
        st.markdown('<div class="ns-card">No customers match the current filters.</div>',
                    unsafe_allow_html=True)
        return

    # Recommend for visible rows (cached below by dataframe hash indirectly via records order)
    def _fill(row):
        rec = recommend(row.to_dict())
        return pd.Series({
            "top_signals": " · ".join(rec["top_signals"][:2]) if rec["top_signals"] else "—",
            "action": rec["recommended_action"],
            "urgency": rec["urgency"],
        })
    filled = filtered.apply(_fill, axis=1)
    filtered = pd.concat([filtered.reset_index(drop=True), filled.reset_index(drop=True)], axis=1)

    st.markdown(f'<div class="ns-section-h">Prioritized customers · top {len(filtered)}</div>',
                unsafe_allow_html=True)

    display = filtered[[
        "customer_id", "name", "state", "churn_probability", "current_balance",
        "value_at_risk", "expected_intervention_value", "urgency",
        "top_signals", "action",
    ]].copy()
    display["churn_probability"] = (display["churn_probability"] * 100).round(1)
    # Encode customer name as a link: URL query param opens detail; name in fragment
    # is extracted by LinkColumn's display_text regex so the cell reads as the name.
    display["name"] = [
        f"?customer={cid}#{name}"
        for cid, name in zip(display["customer_id"], display["name"])
    ]
    display = display.rename(columns={
        "customer_id": "ID", "name": "Customer", "state": "State",
        "churn_probability": "Churn Risk", "current_balance": "Balance",
        "value_at_risk": "Value at Risk", "expected_intervention_value": "Net Opportunity",
        "urgency": "Urgency", "top_signals": "Top Signals", "action": "Recommended Action",
    })

    st.markdown(
        '<div style="font-size:12px;color:#64748B;margin-bottom:6px">'
        'Click a customer\'s name to open their detail page.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        display, hide_index=True, use_container_width=True, height=560,
        column_config={
            "Customer": st.column_config.LinkColumn(
                "Customer",
                display_text=r"#(.+)$",
                help="Click to open this customer's detail page",
            ),
            "Churn Risk": st.column_config.ProgressColumn(
                "Churn Risk", format="%.1f%%", min_value=0, max_value=50,
                help="90-day refinance probability. Scale capped at 50%.",
            ),
            "Balance": st.column_config.NumberColumn("Balance", format="$%d"),
            "Value at Risk": st.column_config.NumberColumn("Value at Risk", format="$%d"),
            "Net Opportunity": st.column_config.NumberColumn("Net Opportunity", format="$%d"),
        },
        key="customer_table",
    )


# =============================================================================
# Customer Detail View
# =============================================================================

def _timeline(digital, calls, branch, cid, limit=15):
    events = []
    d = digital[digital["customer_id"] == cid]
    for _, r in d.iterrows():
        events.append({"when": r["timestamp"], "kind": "Digital",
                       "detail": r["event_type"].replace("_", " ").title() + f" · {r['channel']}"})
    c = calls[calls["customer_id"] == cid]
    for _, r in c.iterrows():
        events.append({"when": r["timestamp"], "kind": "Call",
                       "detail": r["transcript_snippet"]})
    b = branch[branch["customer_id"] == cid]
    for _, r in b.iterrows():
        events.append({"when": r["timestamp"], "kind": "Branch",
                       "detail": r["banker_note"]})
    if not events:
        return []
    df = pd.DataFrame(events).sort_values("when", ascending=False).head(limit)
    df["when"] = pd.to_datetime(df["when"]).dt.strftime("%Y-%m-%d")
    return df.to_dict("records")


def customer_detail_view(scored, digital, calls, branch):
    cid = st.session_state.get("customer_id")
    if not cid:
        st.warning("No customer selected.")
        return
    if cid not in set(scored["customer_id"]):
        st.warning(f"Customer {cid} not found.")
        return

    row = scored[scored["customer_id"] == cid].iloc[0]
    rec = recommend(row.to_dict())
    prob = float(row["churn_probability"])
    voar = float(row["value_at_risk"])
    balance = float(row["current_balance"])
    rate = float(row["mortgage_rate"])
    gap = int(row["rate_gap_bps"])

    # State for action buttons
    actions_key = f"actions_{cid}"
    if actions_key not in st.session_state:
        st.session_state[actions_key] = []
    actions = st.session_state[actions_key]

    render_brand_bar(f"Customer detail · {cid}")

    # Breadcrumb + back
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.markdown('<div class="ns-crumb">Command Center / Customer</div>',
                    unsafe_allow_html=True)
        header_html = f'<div class="ns-page-title">{row["name"]}</div>'
        chip_html = (
            f'<span class="ns-badge ns-badge-{urgency_class(prob)}">Risk {urgency_class(prob).title()}</span>'
        )
        actioned_html = ""
        if "approved" in actions:
            actioned_html += ' <span class="ns-badge ns-badge-success">Outreach approved</span>'
        if "assigned" in actions:
            actioned_html += ' <span class="ns-badge ns-badge-info">Assigned to banker</span>'
        if "snoozed" in actions:
            actioned_html += ' <span class="ns-badge ns-badge-neutral">Snoozed 30d</span>'

        st.markdown(header_html, unsafe_allow_html=True)
        st.markdown(
            f'<div class="ns-page-sub" style="margin-top:4px">'
            f'{cid} · {row["state"]} · Customer for {int(row["tenure_years"])} yrs · '
            f'{int(row["num_products"])} products · Relationship value ~'
            f'{fmt_dollars(row["relationship_value_estimate"])}'
            f'</div>'
            f'<div style="margin-top:10px">{chip_html}{actioned_html}</div>',
            unsafe_allow_html=True,
        )
    with top_r:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            st.query_params.clear()
            st.session_state["view"] = "command"
            st.rerun()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # --- Top KPI row: Risk / Value / Mortgage ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="ns-card">
          <h4>90-day churn risk</h4>
          <div class="ns-card-lg-value {risk_color_class(prob)}">{prob*100:.1f}%</div>
          <div class="ns-card-lg-sub">Probability the customer refinances within 90 days</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="ns-card">
          <h4>Economic value at risk</h4>
          <div class="ns-card-lg-value">{fmt_dollars(voar)}</div>
          <div class="ns-card-lg-sub">Net opportunity if intervened · {fmt_dollars(row['expected_intervention_value'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        gap_str = f"+{gap} bps" if gap > 0 else f"{gap} bps"
        gap_color = "#F87171" if gap > 0 else "#94A3B8"
        st.markdown(f"""
        <div class="ns-card">
          <h4>Mortgage</h4>
          <div style="display:flex;gap:24px;align-items:baseline">
            <div>
              <div class="ns-card-lg-value" style="font-size:24px">{fmt_dollars(balance)}</div>
              <div class="ns-card-lg-sub">Balance</div>
            </div>
            <div>
              <div class="ns-card-lg-value" style="font-size:24px">{rate:.2f}%</div>
              <div class="ns-card-lg-sub" style="color:{gap_color}">{gap_str} vs market {MARKET_RATE_PCT:.2f}%</div>
            </div>
            <div>
              <div class="ns-card-lg-value" style="font-size:24px">{int(row['remaining_term_months'])}<span style="font-size:14px;color:#94A3B8"> mo</span></div>
              <div class="ns-card-lg-sub">Term left</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Two-col: signals + timeline | AI recommendation ---
    left, right = st.columns([1.05, 1])

    with left:
        st.markdown('<div class="ns-card">'
                    '<h4>Why this customer is flagged</h4>',
                    unsafe_allow_html=True)
        if rec["top_signals"]:
            for s in rec["top_signals"]:
                st.markdown(
                    f'<div class="ns-signal"><span class="ns-signal-dot">●</span><span>{s}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div style="color:#94A3B8;font-size:13px">'
                        'No strong risk signals in the last 90 days.</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Timeline
        st.markdown('<div class="ns-card">'
                    '<h4>Recent activity · last 90 days</h4>',
                    unsafe_allow_html=True)
        tl = _timeline(digital, calls, branch, cid, limit=10)
        if not tl:
            st.markdown('<div style="color:#94A3B8;font-size:13px">No recent activity logged.</div>',
                        unsafe_allow_html=True)
        else:
            for ev in tl:
                st.markdown(
                    f'<div class="ns-tl-item">'
                    f'<span class="ns-tl-when">{ev["when"]}</span>'
                    f'<span class="ns-tl-kind">{ev["kind"]}</span>'
                    f'<span class="ns-tl-detail">{ev["detail"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            f'<div class="ns-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
            f'<h4 style="margin:0">AI-recommended next best action</h4>'
            f'<span class="ns-badge ns-badge-{rec["urgency"]}">Urgency {rec["urgency"]}</span>'
            f'</div>'
            f'<div style="font-size:13px;color:#94A3B8;margin-bottom:6px">Channel · {rec["channel"]}</div>'
            f'<div style="font-size:15px;font-weight:600;color:#F1F5F9;margin-bottom:14px">{rec["recommended_action"]}</div>'
            f'<div style="font-size:11px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">Approved offer</div>'
            f'<div style="font-size:13px;color:#E4E7EB;margin-bottom:14px">{rec["suggested_offer"]}</div>'
            f'<div style="font-size:11px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">Reasoning</div>'
            f'<div class="ns-reasoning">{rec["reasoning"]}</div>'
            f'<div style="font-size:11px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">Drafted outreach <span style="font-weight:400;color:#64748B;text-transform:none;letter-spacing:0">— editable, banker approves before send</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        edited = st.text_area("draft", rec["draft_outreach"], height=180,
                              label_visibility="collapsed", key=f"draft_{cid}")

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Approve outreach", type="primary", use_container_width=True,
                         key=f"approve_{cid}"):
                if "approved" not in actions:
                    actions.append("approved")
                st.toast(f"Outreach approved — queued for send to {row['name']}", icon="✓")
                st.rerun()
        with b2:
            if st.button("Assign to banker", use_container_width=True, key=f"assign_{cid}"):
                if "assigned" not in actions:
                    actions.append("assigned")
                st.toast("Assigned to relationship banker", icon="→")
                st.rerun()
        with b3:
            if st.button("Snooze 30 days", use_container_width=True, key=f"snooze_{cid}"):
                if "snoozed" not in actions:
                    actions.append("snoozed")
                st.toast("Snoozed for 30 days", icon="⏱")
                st.rerun()

    # --- Extracted call intent ---
    calls_c = calls[calls["customer_id"] == cid].sort_values("timestamp", ascending=False)
    if not calls_c.empty:
        st.markdown('<div class="ns-section-h">AI-extracted intent · recent calls</div>',
                    unsafe_allow_html=True)
        for _, r in calls_c.head(3).iterrows():
            chips = []
            if bool(r["refinance_intent"]):
                chips.append('<span class="ns-badge ns-badge-high">Refi intent</span>')
            if bool(r["competitor_shopping"]):
                chips.append('<span class="ns-badge ns-badge-high">Competitor shopping</span>')
            if bool(r["payoff_intent"]):
                chips.append('<span class="ns-badge ns-badge-high">Payoff intent</span>')
            if not chips:
                chips.append('<span class="ns-badge ns-badge-neutral">Neutral</span>')
            when = pd.to_datetime(r["timestamp"]).strftime("%Y-%m-%d")
            st.markdown(f"""
            <div class="ns-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px">
                <div style="flex:1">
                  <div style="font-size:12px;color:#64748B;font-family:'JetBrains Mono',monospace">{when} · {r['reason_code']} · {int(r['duration_sec'])}s</div>
                  <div style="margin-top:8px;font-size:13.5px;color:#E4E7EB;font-style:italic;line-height:1.55">"{r['transcript_snippet']}"</div>
                </div>
                <div style="min-width:260px;text-align:right">
                  <div>{' '.join(chips)}</div>
                  <div style="margin-top:10px;font-size:11.5px;color:#94A3B8">
                    Rate sensitivity <span style="color:#E4E7EB;font-weight:600">{r['rate_sensitivity']:.2f}</span>
                    &nbsp;·&nbsp; Sentiment <span style="color:#E4E7EB;font-weight:600">{r['sentiment']:+.2f}</span>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# Router
# =============================================================================

def main():
    scored, digital, calls, branch = load_data()

    if "view" not in st.session_state:
        st.session_state["view"] = "command"
    if "_url_read" not in st.session_state:
        # One-shot: honor initial URL only on first load
        qp_customer = st.query_params.get("customer")
        if qp_customer and qp_customer in set(scored["customer_id"]):
            st.session_state["view"] = "detail"
            st.session_state["customer_id"] = qp_customer
        st.session_state["_url_read"] = True

    if st.session_state["view"] == "detail":
        customer_detail_view(scored, digital, calls, branch)
    else:
        command_center_view(scored)


if __name__ == "__main__":
    main()

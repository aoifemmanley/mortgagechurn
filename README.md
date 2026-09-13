# Mortgage Retention Command Center — Prototype

Working prototype for a bank's mortgage retention workflow, built as part of a case-study submission.

**All data is synthetic.** This prototype exists to demonstrate the end-to-end workflow — from raw bank data through churn prediction, economic prioritization, and AI-recommended next-best-action, delivered to a banker through a lightweight operational UI. The model metrics reflect how the synthetic data was generated and should not be interpreted as evidence of production performance.

---

## The workflow

```
Bank data → Feature engineering → Churn ML → Value-at-risk → AI recommendation → Banker approval → Outreach
```

Six steps, each implemented as a small standalone script or module in `src/`, plus a Streamlit UI in `app.py`.

| Component | File | What it does |
|---|---|---|
| Synthetic data | `src/generate_data.py` | Generates ~10k customers with linked mortgages, 90 days of digital events, call-center interactions (with transcript snippets and pre-extracted intent fields), branch interactions, and 90-day churn outcomes. A latent per-customer refinance propensity drives both event patterns and outcomes, with substantial noise. |
| Features | `src/feature_engineering.py` | Rolls up events into 30/90-day windows per customer, joins with customer/mortgage attributes. |
| Model | `src/train_model.py` | HistGradientBoosting classifier (sklearn) wrapped in `CalibratedClassifierCV` (isotonic) so predicted probabilities reflect real empirical rates. Writes model, per-customer predictions, and metrics. |
| Value model | `src/value_model.py` | Turns churn probability into economic value-at-risk using explicit configurable assumptions (NIM, remaining life, retention rate concession). Also computes expected value of intervention. |
| AI agent | `src/agent.py` | Simulated LLM. Deterministic templates that select urgency, channel, approved offer, reasoning, and personalized outreach draft based on the customer's top risk signals. No external API calls. |
| UI | `app.py` | Streamlit app with two views: portfolio-level retention command center (KPIs + prioritized customer table), and a customer detail page (risk drivers, recent activity, AI recommendation, drafted outreach, action buttons that persist state). Dark theme; styled to feel like an operational tool rather than a data-science dashboard. |

## Palantir Foundry fit

The data model is intentionally shaped like a Palantir ontology. `Customer` and `Mortgage` are core objects; `DigitalEvent`, `CallInteraction`, and `BranchInteraction` are linked event objects. In a production build these would be Foundry ontology objects, Foundry pipelines would produce the feature table, the churn model would be a model asset bound to `Customer`, AIP Logic would replace the pre-extracted call intent, an AIP Agent would replace `src/agent.py`, and the UI would be a Workshop application with the "Approve Outreach" button implemented as an ontology Action.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data, features, model, and per-customer scored table
python -m src.generate_data
python -m src.feature_engineering
python -m src.train_model
python -m src.value_model

# Launch the app
streamlit run app.py
```

The pipeline is deterministic (random seed 42) — re-running produces identical output.

Then open [http://localhost:8501](http://localhost:8501) (or the port Streamlit reports). To jump straight to a compelling customer, try [http://localhost:8501/?customer=C0105859](http://localhost:8501/?customer=C0105859) (Julie Johnson — 52% churn risk, $827K balance).

## Project structure

```
├── src/
│   ├── generate_data.py         Synthetic data generation
│   ├── feature_engineering.py   30/90-day event rollups
│   ├── train_model.py           Model training + calibration
│   ├── value_model.py           Value-at-risk + intervention economics
│   └── agent.py                 Simulated next-best-action agent
├── app.py                       Streamlit UI
├── scripts/
│   └── capture_screenshots.py   Automated screenshot capture (Playwright)
├── screenshots/                 Prototype screenshots
├── .streamlit/config.toml       Dark theme
├── requirements.txt
└── README.md
```

## What's synthetic

The purpose of this prototype is to demonstrate the *workflow*, not to prove specific ROI. The financial model that accompanies the case-study submission uses the case's stated portfolio and churn figures, not numbers from this prototype.

- Synthetic customers, mortgages, and events were generated with realistic distributions but calibrated for demonstration.
- The 90-day base churn rate in the synthetic data is ~2%, roughly matching the case's stated annual churn figure — a demonstration simplification, not a real behavioral estimate.
- The model achieves ~0.90 AUC and ~5.5× top-decile lift on the synthetic data. These numbers reflect how the data was generated, not underlying customer behavior. In a production MVP the model would be trained and validated on the bank's real 5-year history.
- The "AI-extracted" call intent fields (refinance intent, competitor shopping, payoff intent, rate sensitivity, sentiment) are generated at data-synthesis time and presented in the UI as if extracted by an LLM. In production these would come from AIP Logic on real transcripts.
- The agent's next-best-action recommendation, reasoning, and drafted outreach are produced by deterministic templates in `src/agent.py`. In production this would be an AIP Agent making real LLM calls.

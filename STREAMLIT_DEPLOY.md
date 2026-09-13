# Deploying to Streamlit Community Cloud

This app is designed to boot instantly with no build steps — the pre-generated
data files (`data/scored_customers.csv`, `data/digital_events.csv`, etc.) are
committed to the repo, so Streamlit Cloud just needs to `pip install -r requirements.txt`
and run `streamlit run app.py`.

## Steps

1. **Push this repo to GitHub** (you've done this if you're reading this in the repo).

2. **Sign in to [share.streamlit.io](https://share.streamlit.io)** with the same GitHub account.

3. Click **"New app"** and fill in:
   - **Repository:** `aoifemmanley/mortgagechurn`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - Optionally set a custom subdomain, e.g., `mortgage-retention-prototype` → `https://mortgage-retention-prototype.streamlit.app`

4. Click **Deploy**. First build takes 2–5 minutes (pip install + first launch).

5. Once green, share the URL. Example landing paths:
   - Command center: `https://<your-subdomain>.streamlit.app`
   - Jump to Julie Johnson: `https://<your-subdomain>.streamlit.app/?customer=C0105859`

## Constraints on the free tier

- Public repo only (private needs paid plan)
- ~1 GB RAM per app
- Apps sleep after ~7 days of inactivity — first visitor after that waits ~30s for cold start
- ~3 apps per account

## If you change the code

Push to `main` — Streamlit Cloud auto-redeploys on every push. Takes ~1–2 minutes.

## If you want to regenerate the data

The data files in `data/` are checked in for zero-setup deployment. If you want
fresh data (e.g., different random seed, different customer count):

```bash
python -m src.generate_data
python -m src.feature_engineering
python -m src.train_model
python -m src.value_model
git add data/ models/ && git commit -m "Regenerate data" && git push
```

Streamlit Cloud will pick up the change on next deploy.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The data files are already committed, so the app runs immediately.

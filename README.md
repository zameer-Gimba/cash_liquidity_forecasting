# Predictive Liquidity Model for Agent Banking in Nigeria

A production-oriented machine-learning project for forecasting next-day cash withdrawal and deposit demand for Nigerian POS agents and recommending cash reserves and float top-ups with configurable safety buffers.

## Problem Statement

POS agents frequently face liquidity mismatches: insufficient physical cash for withdrawals or insufficient electronic float for deposits and transfers. This project predicts next-day withdrawal demand and next-day deposit demand, and supports operational decisions that reduce failed transactions, customer dissatisfaction, and idle cash.

## Dataset

The canonical dataset is `data/feature_engineered_dataset/liquidity_dataset.csv`: 1,514 daily rows × 50 columns, built from one Moniepoint POS agent's full account statement (16 December 2021 to 22 April 2026, 194,582 transactions). It holds daily aggregates only — no narrations, names or account numbers. The raw statement is never committed.

Normal trading ran until mid-January 2026; after that the account was barely used (a few transactions a week, then none). Training and evaluation therefore stop at `CANONICAL_TRAINING_END_DATE = 2026-01-16` (`src/utils/config.py`), giving 1,487 usable days.

Rebuild it from a statement export with:

```bash
python -m src.preprocessing.pipeline path/to/statement.xlsx --out data/feature_engineered_dataset/liquidity_dataset.csv
```

### How transactions are categorised

Moniepoint has recorded the same business event in different ways over the years — customer cash deposits were `DEP` entries until 2022, then `TRF` transfers with changing narrations; card cash-outs moved from `WTH` to `PUR` in 2025. An earlier version matched narration text for one hard-coded account, so from late 2023 onward every deposit was missed (deposits read ₦0 for 2024–2025) and 2025 card cash-outs were partly lost.

`src/preprocessing/moniepoint.py` now categorises by the transaction-type code at the start of each `Reference` (`WTH|…`, `TRF|…`) plus the debit/credit side, and uses the narration only to tell the agent's own float movements apart from customers:

| Category | What it means for liquidity | How it is recognised |
|---|---|---|
| Withdrawal | Customer cash-out (pays out physical cash) | Card cash-outs (`WTH`, `PUR`) and transfers *in* from customers (transfer cash-out) |
| Deposit | Customer cash deposit (uses up float) | `DEP`, and transfers *out* to customers |
| WalletTransfer | Agent's own float moves (not customer demand) | `FND`/`WFT`, transfers between masked own wallets, money *from* an own account (credits) or *to* an own account (debits) |
| Levy, Commission, Airtime, BillPayment, Insurance, Reversal, Other | Fees, cashback, VAS, refunds, loans/terminal events | Reference codes and levy/stamp-duty narrations |

Own accounts: the holder name is read from the statement header; other own-account names come from a git-ignored `config/own_accounts.txt` (one per line) or the `LIQUIDITY_OWN_ACCOUNTS` environment variable, so no personal names live in the code.

Every transaction lands in exactly one category, and the category totals reconcile to the statement totals to the naira (`moniepoint.reconciliation`). Corrected yearly totals (₦ millions):

| Year | Withdrawals | Deposits | Own float in / out |
|---|---|---|---|
| 2022 | 668 | 665 | 128 / 144 |
| 2023 | 546 | 516 | 67 / 102 |
| 2024 | 606 | 621 | 21 / 14 |
| 2025 | 395 | 420 | 67 / 47 |

Uploads through the Streamlit sidebar (Moniepoint `.xlsx` exports, or an engineered CSV) go through the same code, entirely in memory (`st.session_state`).

The dataset carries three targets: `Target_Next_Day_Withdrawal_Amount`, `Target_Next_Day_Deposit_Amount`, and `Target_Has_Deposit_Tomorrow` (binary).

## Repository Structure

```text
cash_liquidity_forecasting/
├── data/feature_engineered_dataset/   # canonical liquidity_dataset.csv (immutable)
├── notebooks/
├── src/preprocessing/                 # pipeline.py, path_setup helpers
├── src/models/                        # train_*.py, evaluate_models.py, predict_deposit.py
├── src/utils/                         # config.py, dataset_detector.py, logger.py
├── streamlit_app/pages/
├── assets/
├── models/saved_models/
├── reports/{figures,tables,results}/
├── tests/
├── reference_schema.json
├── requirements.txt
├── requirements-ml.txt
├── runtime.txt
├── Procfile
├── Dockerfile
└── README.md
```

## Methodology

1. Detect uploaded dataset type and auto-clean/classify/engineer raw POS statements when needed (missing values, duplicates, dates, numeric currency fields, reversal/retry handling).
2. Categorise every transaction (see above) and aggregate to daily liquidity observations. The `Transfer` columns are kept for schema compatibility but are always 0, since transfers are now split into Withdrawal, Deposit and WalletTransfer.
3. Engineer calendar, category amount/count, behavioural, lag, and rolling-window features for both withdrawal and deposit series (46 features total).
4. Split chronologically: the last 15% of days (7 June 2025 – 16 January 2026, 224 days) is the test set; models are fit on everything before it (`src/models/common.py:fit_eval_split`).
5. Train and compare Random Forest, XGBoost, LightGBM, ARIMA, SARIMA (weekly seasonality) and an LSTM for withdrawal demand, against three simple baselines. Tree models forecast relative to the current 7-day level (`src/models/level_scaled.py`) because volumes fell sharply in 2025 and trees cannot extrapolate below their training range. ARIMA/SARIMA are scored one step ahead, re-using each day's data as it arrives.
6. Forecast deposits with a two-stage architecture: a classifier predicts whether a deposit occurs tomorrow (`Target_Has_Deposit_Tomorrow`), then a regressor trained only on deposit-active days predicts the amount (`Target_Next_Day_Deposit_Amount`). With deposits now counted correctly, the agent records a deposit on almost every active day, so Stage 1 is near-trivial; it is kept for agents with sparser deposits.
7. Persist models and evaluation tables under `models/saved_models/` and `reports/results/`.
8. Serve results through a Streamlit decision-support dashboard with graceful fallback to a historical baseline when an artifact is missing or incompatible.

## Feature Engineering

The canonical dataset provides 46 model input features (50 columns minus the `TransactionDate` index minus 3 targets), defined in `src/utils/config.py:FEATURE_COLS`:

- Core daily: `Total_Debit`, `Total_Credit`, `Transaction_Count`, `Closing_Balance`, `Net_Flow`
- Category amount + count pairs (11 categories): Withdrawal, Deposit, Airtime, BillPayment, WalletTransfer, Transfer, Levy, Commission, Reversal, Insurance, Other
- Temporal: `DayOfWeek` (integer 0–6), `Month`, `WeekOfMonth`, `Quarter`, `IsWeekend`
- Behavioural: `Average_Withdrawal_Size`, `Average_Deposit_Size`, `Credit_Debit_Ratio`
- Withdrawal and deposit lags: `Lag_{1,7,30}_Withdrawal_Amount`, `Lag_{1,7,30}_Deposit_Amount`
- Withdrawal and deposit rolling averages: `Rolling_{7,30}_Day_Withdrawal_Amount`, `Rolling_{7,30}_Day_Deposit_Amount`
- `Has_Deposit` — current-day deposit flag, the strongest signal for next-day deposit classification

`Liquidity_Risk` (Low/Medium/High) is a UI-only label derived from withdrawal-demand percentiles at display time; it is not a model input or a trained target.

## Training

```bash
pip install -r requirements.txt

python scripts/generate_schema.py            # regenerate reference_schema.json from the canonical CSV
python -m src.models.train_random_forest
python -m src.models.train_xgboost
python -m src.models.train_lightgbm
python -m src.models.train_arima
python -m src.models.train_sarima
python -m src.models.train_lstm
python -m src.models.train_deposit_classifier
python -m src.models.train_deposit_regressor
python -m src.models.evaluate_models
```

All training scripts read the canonical dataset (`src/utils/config.py:CANONICAL_DATASET_PATH`).

## Evaluation

Test period: 7 June 2025 – 16 January 2026 (224 trading days, never used for fitting). Each forecast is for the next active day. Lower MAE (mean absolute error, ₦) is better. A model is only useful if it beats the simple baselines.

**Withdrawals (physical cash needed tomorrow)**

| Model | MAE (₦) | vs best baseline |
|---|---|---|
| Baseline: same as today | 630,137 | |
| Baseline: same weekday last week | 641,243 | |
| Baseline: 7-day average | 510,391 | best baseline |
| Random Forest | 509,168 | −0.2% |
| XGBoost | 530,022 | +3.8% |
| LightGBM | 537,998 | +5.4% |
| ARIMA | 489,180 | −4.2% |
| **SARIMA (weekly)** | **473,646** | **−7.2%** |
| LSTM | 877,082 | +72% |

**Deposit amount (float needed tomorrow)**

| Model | MAE (₦) | vs best baseline |
|---|---|---|
| Baseline: 7-day average | 670,681 | best baseline |
| Random Forest | 654,285 | −2.4% |
| XGBoost | 663,786 | −1.0% |
| LightGBM | 683,055 | +1.8% |
| LSTM | 807,710 | +20% |

What this means, honestly: day-to-day cash demand at a single agent is mostly noise around a weekly rhythm and a slowly moving level. SARIMA, which models exactly that, is the best withdrawal forecaster and cuts the error of the best simple rule by about 7%. The machine-learning models roughly match the 7-day average but do not clearly beat it; the LSTM is worse than every baseline with this much data. R² stays near zero for every method, including the baselines. More agents, and outside signals such as market days, salary dates and holidays, are the most likely way to do better.

Full tables: `reports/results/withdrawal_model_comparison.csv`, `deposit_classifier_comparison.csv`, `deposit_regressor_comparison.csv`. Regenerate with `python -m src.models.evaluate_models`.

## Liquidity Recommendation Engine

```text
Recommended_Cash_Reserve = Predicted_Withdrawal_Demand × (1 + Safety_Buffer_Withdrawal)
Recommended_Float_Top_Up = Predicted_Deposit_Demand × (1 + Safety_Buffer_Deposit)
```

The default safety buffers are 15% (withdrawal) and 10% (deposit), configurable in `src/utils/config.py`.

## Streamlit Dashboard

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py --server.address=0.0.0.0 --server.port=8501
```

When using GitHub Codespaces or GitHub.dev, open the forwarded port preview URL for port `8501` rather than the repository root.

`requirements.txt` includes XGBoost, LightGBM, statsmodels, and TensorFlow alongside the core Streamlit/pandas stack, since the Forecasting page trains any of the 6 withdrawal models live, on demand, in the running app — not just offline. `requirements-ml.txt` is kept as an alias to the same file for local/offline training environments that reference it by name.

Pages:

- **Dashboard**: predicted next-day withdrawal demand, recommended cash reserve, liquidity risk level, predicted next-day deposit demand, and recommended float top-up, plus a withdrawal/deposit history chart.
- **Forecasting**: pick a withdrawal model (Random Forest, XGBoost, LightGBM, ARIMA, SARIMA, or LSTM), see its prediction and recommended reserve, and a two-stage deposit forecast (classifier + regressor) below it. Falls back to a historical (last-30-day average) baseline when an artifact is missing or incompatible.
- **Analytics**: withdrawal trends and day-of-week seasonality, deposit trends and rolling averages, and a category breakdown (Airtime, BillPayment, WalletTransfer, Transfer, Levy, Commission, Reversal, Insurance, Other).
- **Model Comparison**: withdrawal model metrics table and a two-stage deposit comparison (Stage 1 classifier metrics, Stage 2 regressor metrics), read from `reports/results/`.

The dataset loaded in the sidebar (canonical or uploaded) is shared across all pages via `st.session_state['clean_dataset']`.

## Publishing to GitHub

All source files should be committed before publishing. See `docs/GITHUB_REPOSITORY_UPLOAD.md` for step-by-step GitHub upload instructions, or run:

```bash
scripts/publish_to_github.sh https://github.com/<owner>/<repository>.git main
```

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a Streamlit app from the repository.
3. Set the main file path to `streamlit_app/app.py`.
4. Add secrets if external storage is used.

### Render

1. Create a new Web Service from the GitHub repository.
2. Use the included `Procfile`.
3. Set the environment to Python and deploy.

### Railway

1. Create a Railway project from GitHub.
2. Use the Dockerfile or Procfile-based Python deployment.
3. Configure persistent storage if model artifacts are generated at runtime.

### Vercel

Streamlit is not a native Vercel runtime. Deploy a lightweight frontend wrapper on Vercel that calls a hosted prediction API, or embed links to the Streamlit/Render deployment.

## Screenshots

Add dashboard screenshots to `assets/` after launching the app and update this section with image links.

## Future Work

- Add automated model retraining from new transaction feeds.
- Add external features such as holidays, market days, salary dates and local events.
- Pool data from several agents.
- Add calibrated prediction intervals.
- Add FastAPI endpoints for mobile/agent integrations.
- Add explainability reports with SHAP for tree-based models.

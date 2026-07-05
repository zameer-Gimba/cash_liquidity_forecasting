# Predictive Liquidity Model for Agent Banking in Nigeria

A production-oriented machine-learning project for forecasting next-day cash withdrawal and deposit demand for Nigerian POS agents and recommending cash reserves and float top-ups with configurable safety buffers.

## Problem Statement

POS agents frequently face liquidity mismatches: insufficient physical cash for withdrawals or insufficient electronic float for deposits and transfers. This project predicts next-day withdrawal demand and next-day deposit demand, and supports operational decisions that reduce failed transactions, customer dissatisfaction, and idle cash.

## Dataset

The canonical, immutable dataset is `data/feature_engineered_dataset/liquidity_dataset.csv`: 1,513 rows × 50 columns spanning 16 December 2021 to 22 May 2026, with zero missing values. It is never overwritten by any pipeline step; it is the single source of truth for training.

Raw Moniepoint POS agent bank statements (`.xlsx`/`.csv`) can also be uploaded through the Streamlit sidebar. Raw statements include:

- `TransactionDate`, `Narration`, `Reference`, `Debit`, `Credit`, `Balance`

Uploads are auto-detected (`FEATURE_ENGINEERED`, `RAW_STATEMENT`, `PARTIALLY_ENGINEERED`, or `UNKNOWN`), cleaned, classified into 11 narration categories, feature-engineered, and validated against `reference_schema.json` — entirely in-memory (`st.session_state`), never written back to the canonical CSV. See `src/preprocessing/pipeline.py` and `src/utils/dataset_detector.py`.

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
2. Aggregate transactions to daily liquidity observations across 11 narration categories (Withdrawal, Deposit, Airtime, BillPayment, WalletTransfer, Transfer, Levy, Commission, Reversal, Insurance, Other).
3. Engineer calendar, category amount/count, behavioural, lag, and rolling-window features for both withdrawal and deposit series (46 features total).
4. Split chronologically into 70% training, 15% validation, and 15% testing (`src/models/common.py:chronological_split`).
5. Train and compare Random Forest, XGBoost, LightGBM, ARIMA, and SARIMA for withdrawal demand, plus an LSTM sequence model.
6. Forecast deposits with a two-stage architecture: a classifier predicts whether a deposit occurs tomorrow (`Target_Has_Deposit_Tomorrow`), then a regressor trained only on deposit-active days predicts the amount (`Target_Next_Day_Deposit_Amount`). This avoids the zero-inflation problem of a direct regression on a target where ~64% of days have no deposit.
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

All training scripts read directly from the immutable canonical dataset (`src/utils/config.py:CANONICAL_DATASET_PATH`) — there is no separate feature-generation CLI step for the canonical CSV. Raw statement processing only happens transiently through the Streamlit upload flow.

## Evaluation

Regression and classification metrics are written to `reports/results/`:

- `withdrawal_model_comparison.csv` — MAE, RMSE, MAPE, R² for all 6 withdrawal models (Random Forest, XGBoost, LightGBM, ARIMA, SARIMA, LSTM)
- `deposit_classifier_comparison.csv` — Accuracy, Precision, Recall, F1, ROC AUC for the Stage 1 classifiers
- `deposit_regressor_comparison.csv` — MAE, RMSE, MAPE, R² for the Stage 2 regressors (deposit-active days only)

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
- Add external features such as holidays, market days, weather, and local events.
- Add calibrated prediction intervals.
- Add FastAPI endpoints for mobile/agent integrations.
- Add explainability reports with SHAP for tree-based models.

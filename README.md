# Predictive Liquidity Model for Agent Banking in Nigeria

A production-oriented machine-learning project for forecasting next-day cash withdrawal demand for Nigerian POS agents and recommending cash reserves with a configurable safety buffer.

## Problem Statement

POS agents frequently face liquidity mismatches: insufficient physical cash for withdrawals or insufficient electronic float for deposits and transfers. This project predicts next-day withdrawal demand and supports operational decisions that reduce failed transactions, customer dissatisfaction, and idle cash.

## Dataset

The expected raw transaction dataset spans approximately December 2021 through 2026 and includes:

- `TransactionDate`
- `TransactionTime`
- `Narration`
- `Reference`
- `Debit`
- `Credit`
- `Balance`

The model-ready feature dataset is stored under `data/feature_engineered_dataset/` and includes `Target_Cash_Demand_Next_Day`, defined as the next day's withdrawal demand.

## Repository Structure

```text
agent-liquidity-prediction/
├── data/feature_engineered_dataset/
├── notebooks/
├── src/preprocessing/
├── src/models/
├── src/visualization/
├── src/utils/
├── streamlit_app/pages/
├── assets/
├── models/saved_models/
├── reports/{figures,tables,results}/
├── tests/
├── requirements.txt
├── runtime.txt
├── Procfile
├── Dockerfile
└── README.md
```

## Methodology

1. Clean raw POS transactions: missing values, duplicates, dates, numeric currency fields, and outlier reports.
2. Aggregate data to daily liquidity observations.
3. Engineer calendar, lag, rolling-window, cash-flow-ratio, and transaction-intensity features.
4. Split chronologically into 70% training, 15% validation, and 15% testing.
5. Tune tree models with `RandomizedSearchCV` and `TimeSeriesSplit(n_splits=5)`.
6. Train and compare Random Forest, XGBoost, LightGBM, ARIMA, SARIMA, and LSTM.
7. Persist models, scalers, encoders, metrics, and comparison tables.
8. Serve results through a Streamlit decision-support dashboard.

## Feature Engineering

Implemented features include:

- `DayOfWeek` (one-hot encoded for tree-based models)
- `Month`
- `WeekOfMonth`
- `IsWeekend`
- `Withdrawal_Count`
- `Deposit_Count`
- `Rolling_7_Day_Average`
- `Rolling_30_Day_Average`
- `Lag_1_Day`, `Lag_7_Day`, `Lag_30_Day`
- `Net_Flow`
- `Cash_Flow_Ratio`
- `Transaction_Intensity`
- `Liquidity_Risk` labels from withdrawal-demand percentiles

## Training

```bash
python -m src.preprocessing.pipeline data/raw_transactions.csv --output-path data/feature_engineered_dataset/features.csv
python -m src.models.train_random_forest data/feature_engineered_dataset/features.csv
python -m src.models.train_xgboost data/feature_engineered_dataset/features.csv
python -m src.models.train_lightgbm data/feature_engineered_dataset/features.csv
python -m src.models.train_arima data/feature_engineered_dataset/features.csv
python -m src.models.train_sarima data/feature_engineered_dataset/features.csv
python -m src.models.train_lstm data/feature_engineered_dataset/features.csv
python -m src.models.train_risk_classifiers data/feature_engineered_dataset/features.csv
python -m src.models.evaluate_models
```

## Evaluation

Regression metrics are saved in `reports/results/`:

| Model | MAE | RMSE | MAPE | R² |
| --- | ---: | ---: | ---: | ---: |
| Generated after training | - | - | - | - |

Classification models for liquidity risk report accuracy and weighted F1.

## Liquidity Recommendation Engine

```text
Recommended_Cash_Reserve = Predicted_Withdrawal_Demand × (1 + Safety_Buffer)
```

The default safety buffer is 15% and can be configured in `src/utils/config.py` or the Streamlit Forecasting page.

## Streamlit Dashboard

Run locally:

```bash
streamlit run streamlit_app/app.py
```

Pages:

- Dashboard: current date, predicted tomorrow demand, recommended reserve, risk level, model used, confidence score.
- Forecasting: select a future date, generate prediction, and display reserve plus safety buffer.
- Analytics: historical withdrawals, deposits, monthly trends, transaction trends, seasonality, and rolling averages.
- Model Comparison: metrics table, ranking, visual comparisons, and feature importance placeholder support.


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

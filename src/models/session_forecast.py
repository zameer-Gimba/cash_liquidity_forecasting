"""Interactive, session-scoped model training and date-aware prediction.

Used by the Streamlit Forecasting page to train a model on whatever dataset
is currently loaded (canonical or uploaded) and generate a prediction for an
arbitrary future date, without touching the offline-trained artefacts in
models/saved_models/.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import DATE_COL, FEATURE_COLS, LSTM_SEQUENCE_LENGTH, RANDOM_STATE, TARGET_WITHDRAWAL

MODEL_OPTIONS: dict[str, str] = {
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "LightGBM": "lgbm",
    "ARIMA": "arima",
    "SARIMA": "sarima",
    "LSTM": "lstm",
}


def _calendar_fields(target_date: pd.Timestamp) -> dict[str, int]:
    return {
        "DayOfWeek": int(target_date.dayofweek),
        "Month": int(target_date.month),
        "WeekOfMonth": int((target_date.day - 1) // 7) + 1,
        "Quarter": int(target_date.quarter),
        "IsWeekend": int(target_date.dayofweek in (5, 6)),
    }


def build_future_feature_row(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Feature row for target_date: its own calendar fields, latest known values elsewhere."""
    last_row = df.sort_values(DATE_COL).iloc[-1]
    row = {col: last_row[col] for col in FEATURE_COLS}
    row.update(_calendar_fields(pd.Timestamp(target_date)))
    return pd.DataFrame([row], columns=FEATURE_COLS)


def train_withdrawal_model(model_key: str, df: pd.DataFrame) -> tuple[Any, dict]:
    """Fit model_key on the full current dataset. Returns (model, extra_artifacts)."""
    ordered = df.sort_values(DATE_COL).reset_index(drop=True)
    # clean_dataset stores TransactionDate as a plain string; asfreq('D') below silently produces an
    # all-NaN series (and a degenerate ARIMA/SARIMA fit) unless the index is a real DatetimeIndex.
    ordered[DATE_COL] = pd.to_datetime(ordered[DATE_COL])
    X, y = ordered[FEATURE_COLS], ordered[TARGET_WITHDRAWAL]

    if model_key in ("rf", "xgb", "lgbm"):
        from src.models.level_scaled import tree_regressor
        return tree_regressor(model_key, "Withdrawal").fit(X, y), {}

    if model_key in ("arima", "sarima"):
        # Daily withdrawals (inactive days carried forward); forecasting k days past the last
        # date gives the expected withdrawal on that date.
        series = ordered.set_index(DATE_COL)["Withdrawal_Amount"].asfreq("D").ffill()
        if model_key == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series, order=(1, 1, 1)).fit()
        else:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
        return model, {"last_date": series.index.max()}

    if model_key == "lstm":
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler

        n = LSTM_SEQUENCE_LENGTH
        if len(ordered) <= n:
            raise ValueError(f"Not enough rows to train an LSTM (need more than {n}).")
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        ys = y.to_numpy()
        xs = np.array([Xs[i - n:i] for i in range(n, len(Xs))])
        ys = ys[n:]
        y_scale = float(ys.mean()) or 1.0
        model = tf.keras.Sequential([tf.keras.layers.Input((n, len(FEATURE_COLS))), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(xs, ys / y_scale, epochs=25, batch_size=32, verbose=0)
        return model, {"scaler": scaler, "y_scale": y_scale}

    raise ValueError(f"Unknown model key: {model_key}")


def predict_withdrawal_for_date(model_key: str, model: Any, extra: dict, df: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """Predict Target_Next_Day_Withdrawal_Amount for an arbitrary future date."""
    target_date = pd.Timestamp(target_date)

    if model_key in ("arima", "sarima"):
        last_date = extra["last_date"]
        steps = max((target_date - last_date).days, 1)
        if hasattr(model, "get_forecast"):
            preds = np.asarray(model.get_forecast(steps=steps).predicted_mean).ravel()
        else:
            preds = np.asarray(model.forecast(steps=steps)).ravel()
        return float(preds[steps - 1])

    if model_key == "lstm":
        ordered = df.sort_values(DATE_COL)
        seq = ordered[FEATURE_COLS].tail(LSTM_SEQUENCE_LENGTH).copy()
        if len(seq) < LSTM_SEQUENCE_LENGTH:
            raise ValueError("Not enough history for the LSTM sequence window.")
        for col, value in _calendar_fields(target_date).items():
            seq.iloc[-1, seq.columns.get_loc(col)] = value
        X = extra["scaler"].transform(seq.values)[None, :, :]
        return float(model.predict(X, verbose=0).ravel()[0]) * extra["y_scale"]

    row = build_future_feature_row(df, target_date)
    return float(model.predict(row[FEATURE_COLS])[0])


def forecast_path(model_key: str, model: Any, extra: dict, df: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.Series:
    """Withdrawal forecasts for several future dates at once (not available for the LSTM)."""
    dates = [pd.Timestamp(d) for d in dates]
    if model_key in ("arima", "sarima"):
        last_date = extra["last_date"]
        steps = max((max(dates) - last_date).days, 1)
        mean = model.get_forecast(steps=steps).predicted_mean
        mean.index = [last_date + pd.Timedelta(days=i + 1) for i in range(steps)]
        return mean.reindex(dates).clip(lower=0)
    if model_key in ("rf", "xgb", "lgbm"):
        rows = pd.concat([build_future_feature_row(df, d) for d in dates], ignore_index=True)
        return pd.Series(model.predict(rows[FEATURE_COLS]), index=dates).clip(lower=0)
    raise ValueError("A multi-day forecast path is not available for this model.")

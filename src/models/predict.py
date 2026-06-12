"""Prediction API wrapper and liquidity recommendation engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.common import feature_columns
from src.utils.config import FEATURE_DATA_DIR, MODEL_DIR, SAFETY_BUFFER, TARGET_COLUMN

MODEL_ARTIFACTS: dict[str, str] = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
    "ARIMA": "arima.joblib",
    "SARIMA": "sarima.joblib",
    "LSTM": "lstm.keras",
    "LSTM Fallback MLP": "lstm_fallback_mlp.joblib",
}


@dataclass(frozen=True)
class PredictionResult:
    """Structured result returned by date-based prediction helpers."""

    model_name: str
    forecast_date: pd.Timestamp
    predicted_withdrawal_demand: float
    recommended_cash_reserve: float
    safety_buffer: float
    risk_level: str
    confidence_score: str
    source: str


def recommended_cash_reserve(predicted_withdrawal_demand: float, safety_buffer: float = SAFETY_BUFFER) -> float:
    """Return cash reserve recommendation using configurable safety buffer."""
    return float(max(predicted_withdrawal_demand, 0.0) * (1 + safety_buffer))


def risk_level_from_prediction(prediction: float, historical_target: pd.Series) -> str:
    """Classify predicted demand as Low, Medium, or High using historical percentiles."""
    history = pd.to_numeric(historical_target, errors="coerce").dropna()
    if history.empty:
        return "Unknown"
    low_cut = history.quantile(0.33)
    high_cut = history.quantile(0.67)
    if prediction <= low_cut:
        return "Low"
    if prediction <= high_cut:
        return "Medium"
    return "High"


def find_feature_dataset(default_path: Path = FEATURE_DATA_DIR / "features.csv") -> Path | None:
    """Return the most likely feature dataset path, if one exists."""
    if default_path.exists():
        return default_path
    candidates = sorted(FEATURE_DATA_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_feature_history(path: str | Path | None = None) -> pd.DataFrame:
    """Load and sort the feature-engineered history used for date-based prediction."""
    dataset_path = Path(path) if path is not None else find_feature_dataset()
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError("No feature dataset found. Upload or create data/feature_engineered_dataset/features.csv first.")
    history = pd.read_csv(dataset_path, parse_dates=["TransactionDate"])
    if history.empty:
        raise ValueError("Feature dataset is empty; cannot generate a forecast.")
    return history.sort_values("TransactionDate").reset_index(drop=True)


def available_model_artifacts(model_dir: Path = MODEL_DIR) -> dict[str, Path]:
    """Return saved model artifacts that are currently available on disk."""
    available: dict[str, Path] = {}
    for model_name, filename in MODEL_ARTIFACTS.items():
        path = model_dir / filename
        if path.exists():
            available[model_name] = path
    return available


def _target_date(value: date | datetime | pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value).normalize()
    if pd.isna(timestamp):
        raise ValueError("A valid forecast date is required.")
    return timestamp


def _safe_last_numeric(history: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in history.columns:
        return default
    values = pd.to_numeric(history[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else default


def build_future_feature_row(history: pd.DataFrame, forecast_date: date | datetime | pd.Timestamp | str) -> pd.DataFrame:
    """Build one model-ready feature row for a selected forecast date.

    The trained tabular models need the same engineered feature columns used during
    training. For a future date, unknown transactional values are estimated from the
    latest history and rolling windows, while calendar fields are recomputed from the
    selected date so predictions change when the user chooses a different date.
    """
    if history.empty:
        raise ValueError("Feature history is required to build prediction features.")
    data = history.sort_values("TransactionDate").reset_index(drop=True).copy()
    forecast_ts = _target_date(forecast_date)
    last_row = data.iloc[-1].copy()
    debit_series = pd.to_numeric(data.get("Total_Debit", pd.Series(dtype=float)), errors="coerce").dropna()
    credit_series = pd.to_numeric(data.get("Total_Credit", pd.Series(dtype=float)), errors="coerce").dropna()
    transaction_series = pd.to_numeric(data.get("Transaction_Count", pd.Series(dtype=float)), errors="coerce").dropna()

    row = last_row.to_dict()
    row["TransactionDate"] = forecast_ts
    row["DayOfWeek"] = forecast_ts.day_name()
    row["Month"] = int(forecast_ts.month)
    row["WeekOfMonth"] = int((forecast_ts.day - 1) // 7 + 1)
    row["IsWeekend"] = int(forecast_ts.dayofweek in [5, 6])

    row["Total_Debit"] = float(debit_series.tail(7).mean()) if not debit_series.empty else 0.0
    row["Total_Credit"] = float(credit_series.tail(7).mean()) if not credit_series.empty else 0.0
    row["Transaction_Count"] = float(transaction_series.tail(7).mean()) if not transaction_series.empty else 0.0
    row["Closing_Balance"] = _safe_last_numeric(data, "Closing_Balance")
    row["Withdrawal_Count"] = _safe_last_numeric(data, "Withdrawal_Count")
    row["Deposit_Count"] = _safe_last_numeric(data, "Deposit_Count")
    row["Net_Flow"] = row["Total_Credit"] - row["Total_Debit"]
    row["Rolling_7_Day_Average"] = float(debit_series.tail(7).mean()) if not debit_series.empty else 0.0
    row["Rolling_30_Day_Average"] = float(debit_series.tail(30).mean()) if not debit_series.empty else row["Rolling_7_Day_Average"]
    row["Lag_1_Day"] = float(debit_series.iloc[-1]) if len(debit_series) >= 1 else row["Rolling_7_Day_Average"]
    row["Lag_7_Day"] = float(debit_series.iloc[-7]) if len(debit_series) >= 7 else row["Lag_1_Day"]
    row["Lag_30_Day"] = float(debit_series.iloc[-30]) if len(debit_series) >= 30 else row["Lag_7_Day"]
    row["Cash_Flow_Ratio"] = float(row["Total_Debit"] / row["Total_Credit"]) if row["Total_Credit"] else 0.0
    row["Transaction_Intensity"] = float(row["Transaction_Count"] / row["Rolling_7_Day_Average"]) if row["Rolling_7_Day_Average"] else 0.0
    row[TARGET_COLUMN] = np.nan
    row["Liquidity_Risk"] = "Unknown"
    return pd.DataFrame([row])


def _pipeline_feature_names(model: Any, fallback_row: pd.DataFrame) -> list[str]:
    """Extract feature names expected by a saved sklearn pipeline."""
    preprocess = getattr(model, "named_steps", {}).get("preprocess") if hasattr(model, "named_steps") else None
    if preprocess is None:
        return feature_columns(fallback_row)
    names: list[str] = []
    for _, _, columns in getattr(preprocess, "transformers", []):
        if isinstance(columns, str):
            names.append(columns)
        else:
            names.extend(list(columns))
    return [name for name in names if name in fallback_row.columns]


def _predict_tabular_model(model_path: Path, feature_row: pd.DataFrame) -> float:
    """Predict with a saved sklearn tabular model artifact."""
    model = joblib.load(model_path)
    features = _pipeline_feature_names(model, feature_row)
    if not features:
        features = feature_columns(feature_row)
    prediction = model.predict(feature_row[features])
    return float(np.asarray(prediction).ravel()[0])


def _forecast_steps(history: pd.DataFrame, forecast_date: pd.Timestamp) -> int:
    """Return how many daily steps a time-series model must forecast."""
    last_date = pd.Timestamp(history["TransactionDate"].max()).normalize()
    return max(int((forecast_date - last_date).days), 1)


def _predict_time_series_model(model_path: Path, history: pd.DataFrame, forecast_date: pd.Timestamp) -> float:
    """Predict with ARIMA/SARIMA artifacts or their naive fallback dictionaries."""
    artifact = joblib.load(model_path)
    steps = _forecast_steps(history, forecast_date)
    if isinstance(artifact, dict):
        if artifact.get("strategy") == "weekly_seasonal_naive" and artifact.get("pattern"):
            pattern = np.asarray(artifact["pattern"], dtype=float)
            return float(np.resize(pattern, steps)[-1])
        return float(artifact.get("value", pd.to_numeric(history[TARGET_COLUMN], errors="coerce").dropna().iloc[-1]))
    forecast = artifact.forecast(steps=steps)
    return float(np.asarray(forecast).ravel()[-1])


def _predict_lstm_model(model_path: Path, feature_row: pd.DataFrame) -> float:
    """Predict with an LSTM Keras artifact or the saved MLP fallback."""
    scaler_path = MODEL_DIR / "lstm_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError("Missing LSTM scaler metadata at models/saved_models/lstm_scaler.joblib")
    metadata = joblib.load(scaler_path)
    scaler = metadata["scaler"]
    features = [feature for feature in metadata["features"] if feature in feature_row.columns]
    if not features:
        raise ValueError("No matching numeric LSTM features found for prediction.")
    x_scaled = scaler.transform(feature_row[features].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    if model_path.suffix == ".keras":
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError("TensorFlow is required to load the saved LSTM .keras model. Install requirements-ml.txt or use LSTM Fallback MLP.") from exc
        model = keras.models.load_model(model_path)
        prediction = model.predict(x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1])), verbose=0)
    else:
        model = joblib.load(model_path)
        prediction = model.predict(x_scaled)
    return float(np.asarray(prediction).ravel()[0])


def baseline_prediction(history: pd.DataFrame, forecast_date: date | datetime | pd.Timestamp | str) -> float:
    """Return a robust date-aware baseline when no trained artifact is available."""
    data = history.copy()
    data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])
    forecast_ts = _target_date(forecast_date)
    target = pd.to_numeric(data.get(TARGET_COLUMN, data.get("Total_Debit")), errors="coerce")
    same_dow = data[data["TransactionDate"].dt.dayofweek == forecast_ts.dayofweek]
    same_dow_target = pd.to_numeric(same_dow.get(TARGET_COLUMN, same_dow.get("Total_Debit")), errors="coerce").dropna()
    if not same_dow_target.empty:
        return float(same_dow_target.tail(8).mean())
    cleaned = target.dropna()
    return float(cleaned.tail(7).mean()) if not cleaned.empty else 0.0


def predict_for_date(
    model_name: str,
    forecast_date: date | datetime | pd.Timestamp | str,
    history: pd.DataFrame,
    safety_buffer: float = SAFETY_BUFFER,
    model_dir: Path = MODEL_DIR,
) -> PredictionResult:
    """Generate a demand forecast for the selected date with any supported model."""
    forecast_ts = _target_date(forecast_date)
    feature_row = build_future_feature_row(history, forecast_ts)
    artifacts = available_model_artifacts(model_dir)
    source = "trained_artifact"
    selected_model = model_name

    if selected_model == "Best Available":
        selected_model = next(iter(artifacts), "Historical Baseline")

    try:
        if selected_model in {"Random Forest", "XGBoost", "LightGBM"} and selected_model in artifacts:
            prediction = _predict_tabular_model(artifacts[selected_model], feature_row)
        elif selected_model in {"ARIMA", "SARIMA"} and selected_model in artifacts:
            prediction = _predict_time_series_model(artifacts[selected_model], history, forecast_ts)
        elif selected_model in {"LSTM", "LSTM Fallback MLP"} and selected_model in artifacts:
            prediction = _predict_lstm_model(artifacts[selected_model], feature_row)
        else:
            source = "historical_baseline"
            selected_model = "Historical Baseline"
            prediction = baseline_prediction(history, forecast_ts)
    except Exception:
        # Keep the Streamlit workflow usable even if one saved artifact is stale or incompatible.
        source = f"historical_baseline_after_{selected_model}_error"
        selected_model = "Historical Baseline"
        prediction = baseline_prediction(history, forecast_ts)

    prediction = max(float(prediction), 0.0)
    return PredictionResult(
        model_name=selected_model,
        forecast_date=forecast_ts,
        predicted_withdrawal_demand=prediction,
        recommended_cash_reserve=recommended_cash_reserve(prediction, safety_buffer),
        safety_buffer=safety_buffer,
        risk_level=risk_level_from_prediction(prediction, history[TARGET_COLUMN]) if TARGET_COLUMN in history.columns else "Unknown",
        confidence_score="Model" if source == "trained_artifact" else "Baseline",
        source=source,
    )


class LiquidityPredictor:
    """Load a persisted tabular model and produce demand, reserve, and risk predictions."""

    def __init__(self, model_path: str | Path = MODEL_DIR / "random_forest.joblib", safety_buffer: float = SAFETY_BUFFER) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        self.model: Any = joblib.load(self.model_path)
        self.safety_buffer = safety_buffer

    def predict(self, rows: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predict withdrawal demand for one or more feature rows with a tabular model."""
        features = _pipeline_feature_names(self.model, rows)
        if not features:
            features = feature_columns(rows) if TARGET_COLUMN in rows.columns else [column for column in rows.columns if column not in {"TransactionDate", "Liquidity_Risk"}]
        predictions = self.model.predict(rows[features])
        result = rows[["TransactionDate"]].copy() if "TransactionDate" in rows.columns else pd.DataFrame(index=rows.index)
        result["Predicted_Withdrawal_Demand"] = predictions
        result["Recommended_Cash_Reserve"] = [recommended_cash_reserve(value, self.safety_buffer) for value in predictions]
        if history is not None and TARGET_COLUMN in history.columns:
            result["Risk_Level"] = [risk_level_from_prediction(value, history[TARGET_COLUMN]) for value in predictions]
        return result

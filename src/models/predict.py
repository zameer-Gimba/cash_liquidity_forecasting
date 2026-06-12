"""Prediction API wrapper and liquidity recommendation engine."""
from __future__ import annotations

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.common import feature_columns
from src.utils.config import MODEL_DIR, SAFETY_BUFFER, TARGET_COLUMN


def recommended_cash_reserve(predicted_withdrawal_demand: float, safety_buffer: float = SAFETY_BUFFER) -> float:
    """Return cash reserve recommendation using configurable safety buffer."""
    return float(predicted_withdrawal_demand * (1 + safety_buffer))


def risk_level_from_prediction(prediction: float, historical_target: pd.Series) -> str:
    """Classify predicted demand as Low, Medium, or High using historical percentiles."""
    low_cut = historical_target.quantile(0.33)
    high_cut = historical_target.quantile(0.67)
    if prediction <= low_cut:
        return "Low"
    if prediction <= high_cut:
        return "Medium"
    return "High"


class LiquidityPredictor:
    """Load a persisted model and produce demand, reserve, and risk predictions."""

    def __init__(self, model_path: str | Path = MODEL_DIR / "random_forest.joblib", safety_buffer: float = SAFETY_BUFFER) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if self.model_path.suffix == ".keras":
            self.model = None
        else:
            self.model = joblib.load(self.model_path)
        self.safety_buffer = safety_buffer

    def _get_features(self, rows: pd.DataFrame) -> list[str]:
        if TARGET_COLUMN in rows.columns:
            return feature_columns(rows)
        return [column for column in rows.columns if column not in {"TransactionDate", "Liquidity_Risk", TARGET_COLUMN}]

    def _prepare_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.model_path.name == "lstm_fallback_mlp.joblib":
            scaler_art = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
            scaler = scaler_art["scaler"]
            features = scaler_art["features"]
            X = rows[features].fillna(0.0).astype(float).values
            rows = rows.copy()
            rows[features] = scaler.transform(X)
            return rows
        return rows

    def _prepare_keras(self, rows: pd.DataFrame) -> pd.DataFrame:
        scaler_art = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
        features = scaler_art["features"]
        scaler = scaler_art["scaler"]
        X = rows[features].fillna(0.0).astype(float).values
        rows = rows.copy()
        rows[features] = scaler.transform(X)
        return rows

    def predict(self, rows: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predict withdrawal demand for one or more feature rows."""
        rows = rows.copy()
        if self.model_path.name == "lstm_fallback_mlp.joblib":
            rows = self._prepare_rows(rows)
        elif self.model_path.name == "lstm.keras":
            from tensorflow import keras
            scaler_art = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
            features = scaler_art["features"]
            scaler = scaler_art["scaler"]
            X = rows[features].fillna(0.0).astype(float).values
            X_scaled = scaler.transform(X)
            keras_model = keras.models.load_model(str(self.model_path))
            predictions = keras_model.predict(X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1])), verbose=0).ravel()
            result = rows[["TransactionDate"]].copy() if "TransactionDate" in rows.columns else pd.DataFrame(index=rows.index)
            result["Predicted_Withdrawal_Demand"] = predictions
            result["Recommended_Cash_Reserve"] = [recommended_cash_reserve(value, self.safety_buffer) for value in predictions]
            if history is not None and TARGET_COLUMN in history.columns:
                result["Risk_Level"] = [risk_level_from_prediction(value, history[TARGET_COLUMN]) for value in predictions]
            return result

        if self.model_path.name in {"arima.joblib", "sarima.joblib"}:
            nobs = getattr(self.model, "nobs", None)
            if nobs is None and hasattr(self.model, "data") and getattr(self.model.data, "endog", None) is not None:
                nobs = len(self.model.data.endog)
            if nobs is None:
                raise RuntimeError("Cannot determine the forecast start index for the saved ARIMA/SARIMA model.")

            def _next_arima_index(row_labels: Any, fallback: int) -> Any:
                if row_labels is None:
                    return fallback
                try:
                    idx = pd.Index(row_labels)
                    if pd.api.types.is_datetime64_any_dtype(idx):
                        freq = idx.freq or pd.infer_freq(idx)
                        if freq is None:
                            freq = pd.Timedelta(days=1)
                        return idx[-1] + freq
                    if pd.api.types.is_integer_dtype(idx):
                        return int(idx[-1]) + 1
                    if pd.api.types.is_float_dtype(idx):
                        return float(idx[-1]) + 1.0
                except Exception:
                    pass
                return fallback

            row_labels = None
            if hasattr(self.model, "data"):
                row_labels = getattr(self.model.data, "row_labels", None)
            predictions = None

            if hasattr(self.model, "get_forecast"):
                try:
                    forecast = self.model.get_forecast(steps=1)
                    predictions = np.asarray(forecast.predicted_mean).ravel()
                except Exception:
                    predictions = None

            if predictions is None and hasattr(self.model, "forecast"):
                try:
                    predictions = np.asarray(self.model.forecast(steps=1)).ravel()
                except Exception:
                    predictions = None

            if predictions is None and hasattr(self.model, "predict"):
                start = _next_arima_index(row_labels, nobs)
                try:
                    predictions = np.asarray(self.model.predict(start=start, end=start)).ravel()
                except KeyError:
                    try:
                        predictions = np.asarray(self.model.predict(start=nobs, end=nobs)).ravel()
                    except Exception:
                        predictions = None
                except Exception:
                    predictions = None

            if predictions is None and hasattr(self.model, "forecast"):
                try:
                    predictions = np.asarray(self.model.forecast(steps=1)).ravel()
                except Exception:
                    predictions = None

            if predictions is None:
                raise RuntimeError("Saved ARIMA/SARIMA model cannot produce forecasts.")

            result = pd.DataFrame({"Predicted_Withdrawal_Demand": predictions})
            if "TransactionDate" in rows.columns:
                last_date = rows["TransactionDate"].max()
                result["TransactionDate"] = last_date + pd.Timedelta(days=1)
            else:
                result["TransactionDate"] = pd.date_range(start=pd.Timestamp.today(), periods=len(predictions), freq="D")
            result = result[["TransactionDate", "Predicted_Withdrawal_Demand"]]
            result["Recommended_Cash_Reserve"] = [recommended_cash_reserve(value, self.safety_buffer) for value in predictions]
            if history is not None and TARGET_COLUMN in history.columns:
                result["Risk_Level"] = [risk_level_from_prediction(value, history[TARGET_COLUMN]) for value in predictions]
            return result

        features = self._get_features(rows)
        predictions = self.model.predict(rows[features])
        result = rows[["TransactionDate"]].copy() if "TransactionDate" in rows.columns else pd.DataFrame(index=rows.index)
        result["Predicted_Withdrawal_Demand"] = predictions
        result["Recommended_Cash_Reserve"] = [recommended_cash_reserve(value, self.safety_buffer) for value in predictions]
        if history is not None and TARGET_COLUMN in history.columns:
            result["Risk_Level"] = [risk_level_from_prediction(value, history[TARGET_COLUMN]) for value in predictions]
        return result

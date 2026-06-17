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
        self._last_features_state = None  # Cache for latest feature state

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

    def _generate_features_for_date(self, target_date: pd.Timestamp, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Generate feature row for a specific future date using historical data patterns.
        
        This creates a feature vector for prediction by combining:
        - Calendar features (day of week, month, etc.) for the target date
        - Latest rolling/lag features from historical data
        """
        historical = historical_data.copy()
        historical = historical.sort_values("TransactionDate").reset_index(drop=True)
        target_date = pd.Timestamp(target_date)
        
        # Create feature row with calendar features for target date
        feature_row = pd.DataFrame({
            "TransactionDate": [target_date]
        })
        
        # Add calendar features for the target date
        feature_row["DayOfWeek"] = target_date.day_name()
        feature_row["Month"] = target_date.month
        feature_row["WeekOfMonth"] = ((target_date.day - 1) // 7 + 1)
        feature_row["IsWeekend"] = int(target_date.dayofweek in [5, 6])
        
        # Copy the latest values from historical data for other numeric features
        last_row = historical.iloc[-1]
        
        # Identify numeric features (excluding those we've already set)
        numeric_features = [col for col in historical.columns 
                           if col not in {"TransactionDate", "DayOfWeek", "Month", 
                                         "WeekOfMonth", "IsWeekend", TARGET_COLUMN, 
                                         "Liquidity_Risk"}]
        
        for col in numeric_features:
            if col in last_row.index:
                feature_row[col] = last_row[col]
        
        return feature_row

    def predict_for_date(self, target_date: pd.Timestamp, historical_data: pd.DataFrame) -> dict[str, Any]:
        """Generate a prediction for a specific target date.
        
        Returns a dict with prediction value, confidence, and metadata.
        """
        # Generate feature row for the target date
        feature_row = self._generate_features_for_date(target_date, historical_data)
        
        # For ARIMA/SARIMA, we need to do forecasting differently
        if self.model_path.name in {"arima.joblib", "sarima.joblib"}:
            return self._predict_arima_for_date(target_date, historical_data)
        
        # For tree-based and neural models, use the generated features
        prediction_df = self.predict(feature_row, history=historical_data)
        
        if prediction_df.empty:
            raise RuntimeError("Prediction failed: empty result")
        
        pred_row = prediction_df.iloc[0]
        
        return {
            "date": target_date,
            "predicted_value": float(pred_row["Predicted_Withdrawal_Demand"]),
            "recommended_reserve": float(pred_row["Recommended_Cash_Reserve"]),
            "risk_level": pred_row.get("Risk_Level", "N/A"),
        }

    def _predict_arima_for_date(self, target_date: pd.Timestamp, historical_data: pd.DataFrame) -> dict[str, Any]:
        """Generate ARIMA/SARIMA prediction for a specific date by forecasting ahead.
        
        Enhances ARIMA forecasts with seasonal adjustments based on historical patterns
        to provide date-dependent predictions.
        """
        target_date = pd.Timestamp(target_date)
        historical = historical_data.copy()
        historical = historical.sort_values("TransactionDate").reset_index(drop=True)
        
        last_date = pd.Timestamp(historical["TransactionDate"].max())
        days_ahead = (target_date - last_date).days
        
        if days_ahead <= 0:
            # If target date is in the past or today, use the last available value
            last_val = historical[TARGET_COLUMN].iloc[-1] if TARGET_COLUMN in historical.columns else historical["Predicted_Withdrawal_Demand"].iloc[-1]
            return {
                "date": target_date,
                "predicted_value": float(last_val),
                "recommended_reserve": float(recommended_cash_reserve(last_val, self.safety_buffer)),
                "risk_level": "N/A",
            }
        
        try:
            # Get base ARIMA/SARIMA forecast
            if hasattr(self.model, "get_forecast"):
                forecast = self.model.get_forecast(steps=days_ahead)
                predictions = np.asarray(forecast.predicted_mean).ravel()
            elif hasattr(self.model, "forecast"):
                predictions = np.asarray(self.model.forecast(steps=days_ahead)).ravel()
            else:
                raise RuntimeError("Cannot generate forecast from ARIMA model")
            
            base_pred_value = float(predictions[-1])  # Base forecast value
            
            # Calculate seasonal adjustments based on historical patterns
            seasonal_adjustment = self._calculate_seasonal_adjustment(target_date, historical)
            
            # Apply adjustment to the base forecast
            pred_value = base_pred_value * (1 + seasonal_adjustment)
            
            return {
                "date": target_date,
                "predicted_value": pred_value,
                "recommended_reserve": float(recommended_cash_reserve(pred_value, self.safety_buffer)),
                "risk_level": "N/A",
            }
        except Exception as e:
            raise RuntimeError(f"ARIMA forecasting failed: {str(e)}")

    def _calculate_seasonal_adjustment(self, target_date: pd.Timestamp, historical_data: pd.DataFrame) -> float:
        """Calculate seasonal adjustment factor based on historical patterns.
        
        This uses day-of-week and month patterns to adjust predictions and make them
        vary by date, even for pure time-series models like ARIMA.
        
        Returns a multiplier adjustment (e.g., -0.1 means 10% reduction, 0.05 means 5% increase).
        """
        # Ensure historical data has required columns
        historical = historical_data.copy()
        if "TransactionDate" not in historical.columns:
            return 0.0  # No adjustment if date column not available
        
        historical["TransactionDate"] = pd.to_datetime(historical["TransactionDate"])
        
        # Add calendar features to historical data
        historical["DayOfWeek"] = historical["TransactionDate"].dt.dayofweek
        historical["Month"] = historical["TransactionDate"].dt.month
        historical["IsWeekend"] = (historical["DayOfWeek"] >= 5).astype(int)
        
        # Get target date features
        target_dow = target_date.dayofweek
        target_month = target_date.month
        target_is_weekend = 1 if target_dow >= 5 else 0
        
        # Get actual withdrawal column
        target_col = TARGET_COLUMN if TARGET_COLUMN in historical.columns else "Predicted_Withdrawal_Demand"
        if target_col not in historical.columns:
            return 0.0
        
        # Calculate day-of-week adjustment
        dow_groups = historical.groupby("DayOfWeek")[target_col].agg(["mean", "count"])
        overall_mean = historical[target_col].mean()
        
        # Only use day-of-week if we have enough samples
        dow_adjustment = 0.0
        if target_dow in dow_groups.index and dow_groups.loc[target_dow, "count"] >= 5:
            dow_mean = dow_groups.loc[target_dow, "mean"]
            dow_adjustment = (dow_mean - overall_mean) / overall_mean if overall_mean != 0 else 0.0
            dow_adjustment *= 0.5  # Weight down-of-week effect to 50%
        
        # Calculate month adjustment
        month_groups = historical.groupby("Month")[target_col].agg(["mean", "count"])
        month_adjustment = 0.0
        if target_month in month_groups.index and month_groups.loc[target_month, "count"] >= 10:
            month_mean = month_groups.loc[target_month, "mean"]
            month_adjustment = (month_mean - overall_mean) / overall_mean if overall_mean != 0 else 0.0
            month_adjustment *= 0.3  # Weight month effect to 30%
        
        # Combine adjustments (capped at ±15% to avoid extreme values)
        total_adjustment = dow_adjustment + month_adjustment
        total_adjustment = np.clip(total_adjustment, -0.15, 0.15)
        
        return total_adjustment



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

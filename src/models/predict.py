"""Prediction API wrapper and liquidity recommendation engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
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
        self.model: Any = joblib.load(self.model_path)
        self.safety_buffer = safety_buffer

    def predict(self, rows: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predict withdrawal demand for one or more feature rows."""
        features = feature_columns(rows) if TARGET_COLUMN in rows.columns else [column for column in rows.columns if column not in {"TransactionDate", "Liquidity_Risk"}]
        predictions = self.model.predict(rows[features])
        result = rows[["TransactionDate"]].copy() if "TransactionDate" in rows.columns else pd.DataFrame(index=rows.index)
        result["Predicted_Withdrawal_Demand"] = predictions
        result["Recommended_Cash_Reserve"] = [recommended_cash_reserve(value, self.safety_buffer) for value in predictions]
        if history is not None and TARGET_COLUMN in history.columns:
            result["Risk_Level"] = [risk_level_from_prediction(value, history[TARGET_COLUMN]) for value in predictions]
        return result

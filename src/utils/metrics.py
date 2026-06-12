"""Regression and classification metric helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error_safe(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute MAPE while ignoring zero-valued actuals."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, MAPE, and R² for a regression model."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mean_absolute_percentage_error_safe(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Return accuracy and weighted F1 classification metrics."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_Weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Persist metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str))

"""Train SARIMA baseline for seasonal liquidity demand."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np

from src.models.common import load_feature_data
from src.preprocessing.split_data import chronological_split
from src.utils.config import MODEL_DIR, TARGET_COLUMN
from src.utils.metrics import regression_metrics, save_metrics


def main(data_path: str, order: tuple[int, int, int] = (1, 1, 1), seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7)) -> dict[str, float]:
    """Train SARIMA and return test metrics; uses weekly seasonal naive fallback if unavailable."""
    df = load_feature_data(data_path)
    splits = chronological_split(df)
    train_series = np.r_[splits.train[TARGET_COLUMN].values, splits.validation[TARGET_COLUMN].values]
    test_series = splits.test[TARGET_COLUMN].values
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("statsmodels") is not None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        predictions = model.forecast(steps=len(test_series))
        joblib.dump(model, MODEL_DIR / "sarima.joblib")
    else:
        pattern = train_series[-7:]
        predictions = np.resize(pattern, len(test_series))
        joblib.dump({"strategy": "weekly_seasonal_naive", "pattern": pattern.tolist()}, MODEL_DIR / "sarima.joblib")
    metrics = regression_metrics(test_series, predictions)
    save_metrics(metrics, Path("reports/results/sarima_metrics.json"))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    print(main(parser.parse_args().data_path))

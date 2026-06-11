from __future__ import annotations

from src.models.predict import recommended_cash_reserve
from src.utils.metrics import regression_metrics


def test_recommended_cash_reserve_default_buffer() -> None:
    assert recommended_cash_reserve(1000) == 1150


def test_regression_metrics_keys() -> None:
    metrics = regression_metrics([100, 200, 300], [110, 190, 310])
    assert {"MAE", "RMSE", "MAPE", "R2"}.issubset(metrics)

from __future__ import annotations

import pandas as pd

from src.models.predict import baseline_prediction, build_future_feature_row, predict_for_date


def _history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "TransactionDate": dates,
            "Total_Debit": range(1000, 1040),
            "Total_Credit": range(2000, 2040),
            "Transaction_Count": [5] * 40,
            "Closing_Balance": [10000] * 40,
            "Net_Flow": [1000] * 40,
            "DayOfWeek": dates.day_name(),
            "Month": dates.month,
            "WeekOfMonth": ((dates.day - 1) // 7 + 1).astype(int),
            "IsWeekend": dates.dayofweek.isin([5, 6]).astype(int),
            "Withdrawal_Count": [3] * 40,
            "Deposit_Count": [2] * 40,
            "Rolling_7_Day_Withdrawal_Amount": pd.Series(range(1000, 1040)).rolling(7, min_periods=1).mean(),
            "Rolling_30_Day_Withdrawal_Amount": pd.Series(range(1000, 1040)).rolling(30, min_periods=1).mean(),
            "Lag_1_Withdrawal_Amount": pd.Series(range(1000, 1040)).shift(1),
            "Lag_7_Withdrawal_Amount": pd.Series(range(1000, 1040)).shift(7),
            "Lag_30_Withdrawal_Amount": pd.Series(range(1000, 1040)).shift(30),
            "Credit_Debit_Ratio": [0.5] * 40,
            "Transaction_Intensity": [0.005] * 40,
            "Target_Cash_Demand_Next_Day": range(1001, 1041),
            "Liquidity_Risk": ["Low", "Medium", "High", "Medium"] * 10,
        }
    )


def test_build_future_feature_row_updates_calendar_from_selected_date() -> None:
    row = build_future_feature_row(_history(), "2024-02-15")
    assert row.loc[0, "TransactionDate"] == pd.Timestamp("2024-02-15")
    assert row.loc[0, "DayOfWeek"] == "Thursday"
    assert row.loc[0, "Month"] == 2
    assert row.loc[0, "WeekOfMonth"] == 3


def test_predict_for_date_uses_baseline_when_model_artifact_is_missing(tmp_path) -> None:
    result = predict_for_date("Random Forest", "2024-02-15", _history(), model_dir=tmp_path)
    assert result.model_name == "Historical Baseline"
    assert result.predicted_withdrawal_demand == baseline_prediction(_history(), "2024-02-15")
    assert result.recommended_cash_reserve > result.predicted_withdrawal_demand

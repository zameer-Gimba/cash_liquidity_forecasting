from __future__ import annotations

import pandas as pd

from src.preprocessing.feature_engineering import TARGET_COLUMN, build_feature_dataset


def test_build_feature_dataset_creates_required_features() -> None:
    df = pd.DataFrame(
        {
            "TransactionDate": pd.date_range("2024-01-01", periods=40, freq="D"),
            "Total_Debit": range(1000, 1040),
            "Total_Credit": range(2000, 2040),
            "Transaction_Count": [5] * 40,
            "Closing_Balance": [10000] * 40,
            "Withdrawal_Count": [3] * 40,
            "Deposit_Count": [2] * 40,
        }
    )
    features = build_feature_dataset(df)
    for column in ["DayOfWeek", "Month", "WeekOfMonth", "IsWeekend", "Lag_1_Withdrawal_Amount", "Lag_7_Withdrawal_Amount", "Lag_30_Withdrawal_Amount", "Credit_Debit_Ratio", "Transaction_Intensity", "Liquidity_Risk"]:
        assert column in features.columns
    assert TARGET_COLUMN in features.columns
    assert len(features) == 39

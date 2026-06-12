from __future__ import annotations

import pandas as pd

from src.preprocessing.split_data import chronological_split, get_time_series_cv


def test_chronological_split_sizes_and_order() -> None:
    df = pd.DataFrame({"TransactionDate": pd.date_range("2024-01-01", periods=100), "value": range(100)})
    splits = chronological_split(df)
    assert len(splits.train) == 70
    assert len(splits.validation) == 15
    assert len(splits.test) == 15
    assert splits.train["TransactionDate"].max() < splits.validation["TransactionDate"].min()
    assert splits.validation["TransactionDate"].max() < splits.test["TransactionDate"].min()


def test_time_series_cv_fold_count() -> None:
    cv = get_time_series_cv(5)
    assert cv.n_splits == 5

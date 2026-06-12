"""Chronological data splitting utilities."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class DatasetSplits:
    """Container for chronological train/validation/test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(df: pd.DataFrame, train_size: float = 0.70, validation_size: float = 0.15) -> DatasetSplits:
    """Split data in chronological order into 70/15/15 partitions by default."""
    if not 0 < train_size < 1 or not 0 <= validation_size < 1:
        raise ValueError("Split sizes must be fractions between 0 and 1")
    ordered = df.sort_values("TransactionDate").reset_index(drop=True)
    train_end = int(len(ordered) * train_size)
    validation_end = train_end + int(len(ordered) * validation_size)
    return DatasetSplits(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )


def get_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Return TimeSeriesSplit for model selection."""
    return TimeSeriesSplit(n_splits=n_splits)

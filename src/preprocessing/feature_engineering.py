"""Feature engineering for daily POS liquidity forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "Target_Next_Day_Withdrawal_Amount"


def infer_transaction_type(narration: pd.Series) -> pd.Series:
    """Infer withdrawal/deposit/other labels from narration text."""
    text = narration.fillna("").str.lower()
    return np.select(
        [text.str.contains("withdraw|cashout|cash out|pos", regex=True), text.str.contains("deposit|transfer|credit", regex=True)],
        ["withdrawal", "deposit"],
        default="other",
    )


def aggregate_daily_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw transaction-level data to daily liquidity observations."""
    data = df.copy()
    data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])
    if "Transaction_Type" not in data.columns:
        data["Transaction_Type"] = infer_transaction_type(data.get("Narration", pd.Series(index=data.index, dtype=str)))
    for column in ["Debit", "Credit", "Balance"]:
        if column not in data.columns:
            data[column] = 0.0
    grouped = data.groupby(data["TransactionDate"].dt.date).agg(
        Total_Debit=("Debit", "sum"),
        Total_Credit=("Credit", "sum"),
        Transaction_Count=("Reference", "count") if "Reference" in data.columns else ("Debit", "count"),
        Closing_Balance=("Balance", "last"),
        Withdrawal_Count=("Transaction_Type", lambda s: int((s == "withdrawal").sum())),
        Deposit_Count=("Transaction_Type", lambda s: int((s == "deposit").sum())),
    )
    grouped.index = pd.to_datetime(grouped.index)
    return grouped.reset_index(names="TransactionDate")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features requested by the project specification."""
    featured = df.copy()
    featured["TransactionDate"] = pd.to_datetime(featured["TransactionDate"])
    featured["DayOfWeek"] = featured["TransactionDate"].dt.day_name()
    featured["Month"] = featured["TransactionDate"].dt.month
    featured["WeekOfMonth"] = ((featured["TransactionDate"].dt.day - 1) // 7 + 1).astype(int)
    featured["IsWeekend"] = featured["TransactionDate"].dt.dayofweek.isin([5, 6]).astype(int)
    return featured


def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling, lag, ratio, net-flow, and target features."""
    featured = df.sort_values("TransactionDate").copy()
    for column in ["Total_Debit", "Total_Credit", "Transaction_Count", "Withdrawal_Count", "Deposit_Count", "Closing_Balance"]:
        if column not in featured.columns:
            featured[column] = 0.0
    featured["Net_Flow"] = featured["Total_Credit"] - featured["Total_Debit"]
    featured["Rolling_7_Day_Withdrawal_Amount"] = featured["Total_Debit"].rolling(7, min_periods=1).mean()
    featured["Rolling_30_Day_Withdrawal_Amount"] = featured["Total_Debit"].rolling(30, min_periods=1).mean()
    featured["Lag_1_Withdrawal_Amount"] = featured["Total_Debit"].shift(1)
    featured["Lag_7_Withdrawal_Amount"] = featured["Total_Debit"].shift(7)
    featured["Lag_30_Withdrawal_Amount"] = featured["Total_Debit"].shift(30)
    featured["Credit_Debit_Ratio"] = featured["Total_Debit"] / featured["Total_Credit"].replace(0, np.nan)
    featured["Credit_Debit_Ratio"] = featured["Credit_Debit_Ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    featured["Transaction_Intensity"] = featured["Transaction_Count"] / (featured["Rolling_7_Day_Withdrawal_Amount"].replace(0, np.nan))
    featured["Transaction_Intensity"] = featured["Transaction_Intensity"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    featured[TARGET_COLUMN] = featured["Total_Debit"].shift(-1)
    return featured


def add_risk_labels(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Create Liquidity_Risk labels from withdrawal-demand percentiles."""
    labeled = df.copy()
    low_cut = labeled[target_column].quantile(0.33)
    high_cut = labeled[target_column].quantile(0.67)
    labeled["Liquidity_Risk"] = pd.cut(
        labeled[target_column],
        bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    return labeled


def build_feature_dataset(raw_or_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Build a model-ready daily feature dataset from raw or pre-aggregated data."""
    if {"Total_Debit", "Total_Credit", "Transaction_Count"}.issubset(raw_or_daily_df.columns):
        daily = raw_or_daily_df.copy()
    else:
        daily = aggregate_daily_transactions(raw_or_daily_df)
    featured = add_calendar_features(daily)
    featured = add_liquidity_features(featured)
    featured = featured.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return add_risk_labels(featured)

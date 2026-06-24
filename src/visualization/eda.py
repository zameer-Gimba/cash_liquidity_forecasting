"""Exploratory data analysis plots for liquidity forecasting."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from src.utils.config import FIGURES_DIR, TARGET_COLUMN

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_eda_figures(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate and save all required EDA figures automatically."""
    data = df.copy()
    data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])
    paths: list[Path] = []
    for column, title, filename in [
        ("Total_Debit", "Daily Withdrawal Trend", "daily_withdrawal_trend.png"),
        ("Total_Credit", "Daily Deposit Trend", "daily_deposit_trend.png"),
        ("Net_Flow", "Cash Flow Trend Analysis", "cash_flow_trend.png"),
        ("Transaction_Count", "Transaction Volume Trend", "transaction_volume_trend.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data["TransactionDate"], data[column])
        ax.set_title(title)
        paths.append(_save(fig, filename, output_dir))
    for column, title, filename in [
        ("Total_Debit", "Monthly Withdrawal Distribution", "monthly_withdrawal_distribution.png"),
        ("Total_Credit", "Monthly Deposit Distribution", "monthly_deposit_distribution.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=data, x="Month", y=column, ax=ax)
        ax.set_title(title)
        paths.append(_save(fig, filename, output_dir))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x="DayOfWeek", y="Transaction_Count", estimator="mean", ax=ax)
    ax.set_title("Day-of-Week Transaction Analysis")
    paths.append(_save(fig, "day_of_week_transactions.png", output_dir))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.select_dtypes("number").corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    paths.append(_save(fig, "correlation_heatmap.png", output_dir))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["TransactionDate"], data["Rolling_7_Day_Withdrawal_Amount"], label="7-day")
    ax.plot(data["TransactionDate"], data["Rolling_30_Day_Withdrawal_Amount"], label="30-day")
    ax.legend()
    ax.set_title("Rolling Average Plots")
    paths.append(_save(fig, "rolling_averages.png", output_dir))
    if len(data) >= 14:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            seasonal_decompose = None
        if seasonal_decompose is not None:
            decomposition = seasonal_decompose(data.set_index("TransactionDate")[TARGET_COLUMN].asfreq("D").interpolate(), model="additive", period=7)
            fig = decomposition.plot()
            fig.set_size_inches(12, 8)
            paths.append(_save(fig, "seasonal_decomposition.png", output_dir))
        decomposition = seasonal_decompose(data.set_index("TransactionDate")[TARGET_COLUMN].asfreq("D").interpolate(), model="additive", period=7)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        paths.append(_save(fig, "seasonal_decomposition.png", output_dir))
    return paths

"""Raw POS transaction data cleaning utilities."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

MONEY_COLUMNS = ["Debit", "Credit", "Balance"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def parse_money(series: pd.Series) -> pd.Series:
    """Convert currency-like text values to numeric values."""
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.replace("₦", "", regex=False), errors="coerce")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transaction records: dates, numeric values, missing data, and duplicates."""
    cleaned = df.copy()
    if "TransactionDate" not in cleaned.columns:
        raise ValueError("TransactionDate column is required")
    cleaned["TransactionDate"] = pd.to_datetime(cleaned["TransactionDate"], errors="coerce")
    if "TransactionTime" in cleaned.columns:
        cleaned["TransactionTime"] = cleaned["TransactionTime"].astype(str).str.strip()
    for column in MONEY_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = parse_money(cleaned[column]).fillna(0.0)
    text_columns = cleaned.select_dtypes(include="object").columns
    cleaned[text_columns] = cleaned[text_columns].fillna("").apply(lambda col: col.str.strip())
    cleaned = cleaned.dropna(subset=["TransactionDate"]).drop_duplicates()
    cleaned = cleaned.sort_values(["TransactionDate"] + (["TransactionTime"] if "TransactionTime" in cleaned.columns else [])).reset_index(drop=True)
    LOGGER.info("Cleaned %s rows", len(cleaned))
    return cleaned


def outlier_report(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Report IQR outliers for selected numeric columns."""
    columns = columns or list(df.select_dtypes("number").columns)
    rows: list[dict[str, float | str | int]] = []
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[column] < lower) | (df[column] > upper)).sum())
        rows.append({"column": column, "lower_bound": lower, "upper_bound": upper, "outlier_count": count})
    return pd.DataFrame(rows)

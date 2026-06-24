"""Dataset type detection and safe repair utilities."""
from __future__ import annotations
import json, os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

def _schema_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reference_schema.json'))

def detect_dataset_type(df: "pd.DataFrame") -> str:
    """Returns one of: FEATURE_ENGINEERED | RAW_STATEMENT | PARTIALLY_ENGINEERED | UNKNOWN."""
    with open(_schema_path(), encoding="utf-8") as f:
        schema = json.load(f)
    required = set(schema['required_columns'])
    cols = set(df.columns)
    if required.issubset(cols):
        return 'FEATURE_ENGINEERED'
    raw_signals = {'Narration', 'Reference', 'Debit', 'Credit'}
    if raw_signals.issubset(cols):
        return 'RAW_STATEMENT'
    overlap = len(required.intersection(cols))
    if 10 <= overlap < len(required):
        return 'PARTIALLY_ENGINEERED'
    return 'UNKNOWN'

def get_repair_map() -> dict:
    """Returns the mapping of old column names to new canonical names."""
    return {
        'Target_Withdrawal_Tomorrow': 'Target_Next_Day_Withdrawal_Amount',
        'Rolling_7_Day_Average': 'Rolling_7_Day_Withdrawal_Amount',
        'Rolling_30_Day_Average': 'Rolling_30_Day_Withdrawal_Amount',
        'Lag_1_Day': 'Lag_1_Withdrawal_Amount',
        'Lag_7_Day': 'Lag_7_Withdrawal_Amount',
        'Lag_30_Day': 'Lag_30_Withdrawal_Amount',
        'Cash_Flow_Ratio': 'Credit_Debit_Ratio',
    }

def attempt_auto_repair(df: "pd.DataFrame") -> tuple[pd.DataFrame, list[str]]:
    """Rename legacy columns, parse dates, and recompute safe derived values with an explicit log."""
    import pandas as pd
    df = df.copy()
    log: list[str] = []
    for old, new in get_repair_map().items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
            log.append(f"REPAIR_APPLIED | rename | {old} → {new}")
    if 'TransactionDate' in df.columns:
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%d-%b-%Y']:
            try:
                df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format=fmt).dt.strftime('%Y-%m-%d')
                log.append(f"REPAIR_APPLIED | date_parse | format={fmt}")
                break
            except Exception:
                continue
    if 'Net_Flow' not in df.columns and 'Total_Credit' in df.columns and 'Total_Debit' in df.columns:
        df['Net_Flow'] = df['Total_Credit'] - df['Total_Debit']
        log.append("REPAIR_APPLIED | recompute | Net_Flow = Total_Credit - Total_Debit")
    return df, log

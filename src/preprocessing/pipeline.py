"""Preprocessing pipeline for raw statements and canonical engineered datasets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import DATE_COL, FEATURE_COLS, ROOT_DIR, SCHEMA_PATH, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT, TARGET_WITHDRAWAL
from src.utils.dataset_detector import attempt_auto_repair, detect_dataset_type
from src.preprocessing.moniepoint import categorise, read_statement, tidy_transactions

# "Transfer" is kept for schema compatibility; transfers are now split into
# Withdrawal / Deposit / WalletTransfer by moniepoint.categorise, so it is always 0.
CATEGORIES = ["Withdrawal", "Deposit", "Airtime", "BillPayment", "WalletTransfer", "Transfer", "Levy", "Commission", "Reversal", "Insurance", "Other"]
ORDERED_COLUMNS = [DATE_COL, *FEATURE_COLS[:-1], TARGET_WITHDRAWAL, TARGET_DEPOSIT, "Has_Deposit", TARGET_HAS_DEPOSIT]

def load_input(source) -> pd.DataFrame:
    """Load a statement or engineered dataset from a path or an uploaded file.

    Moniepoint .xlsx exports (header block + transaction table) are parsed with
    :func:`read_statement`; the holder name is kept in ``df.attrs['account_name']``.
    """
    name = getattr(source, 'name', str(source)).lower()
    if name.endswith(('.xlsx', '.xls')):
        try:
            statement = read_statement(source)
            df = statement.transactions
            df.attrs['account_name'] = statement.account_name
            return df
        except ValueError:
            if hasattr(source, 'seek'): source.seek(0)
            return _read_raw_table(source)
    return pd.read_csv(source)

def _read_raw_table(source) -> pd.DataFrame:
    for engine in ('calamine', 'openpyxl'):
        try:
            if hasattr(source, 'seek'): source.seek(0)
            return pd.read_excel(source, engine=engine)
        except Exception:  # noqa: BLE001 - try the next engine
            continue
    raise ValueError('Could not read the spreadsheet.')

def _schema() -> dict:
    return json.loads((ROOT_DIR / SCHEMA_PATH).read_text(encoding='utf-8'))

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    schema = _schema(); required = schema['required_columns']
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Dataset missing required canonical columns: {missing}")
    out = df[required].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors='raise').dt.strftime('%Y-%m-%d')
    for col, dtype in schema['column_dtypes'].items():
        if col == DATE_COL: continue
        if dtype.startswith('int'):
            out[col] = pd.to_numeric(out[col], errors='raise').fillna(0).astype('int64')
        else:
            out[col] = pd.to_numeric(out[col], errors='raise').astype('float64')
    return out

def _date_column(df: pd.DataFrame) -> str:
    for c in [DATE_COL, 'Date', 'Transaction Date', 'Transaction_Date', 'Value Date']:
        if c in df.columns: return c
    raise ValueError('Raw statement must contain a transaction date column.')

def process_raw_statement(df: pd.DataFrame, holder_name: str = '') -> pd.DataFrame:
    """Daily liquidity features from a raw transaction statement (see moniepoint.categorise)."""
    holder_name = holder_name or df.attrs.get('account_name', '')
    df = df.copy()
    date_col = _date_column(df)
    if date_col != 'Date': df['Date'] = df[date_col]
    if 'Balance' not in df.columns: df['Balance'] = np.nan
    df = tidy_transactions(df)
    df['Balance'] = df['Balance'].ffill().fillna(0.0)
    df[DATE_COL] = df['Date'].dt.normalize()
    df['Transaction_Category'] = categorise(df, holder_name)
    rows = []
    for day, g in df.groupby(DATE_COL, sort=True):
        row = {DATE_COL: day.strftime('%Y-%m-%d'), 'Total_Debit': g['Debit'].sum(), 'Total_Credit': g['Credit'].sum(), 'Transaction_Count': len(g), 'Closing_Balance': g['Balance'].iloc[-1]}
        row['Net_Flow'] = row['Total_Credit'] - row['Total_Debit']
        for cat in CATEGORIES:
            cg = g[g['Transaction_Category'] == cat]
            if cat == 'Withdrawal': amt = cg['Credit'].sum()
            elif cat == 'Deposit': amt = cg['Debit'].sum()
            else: amt = np.where(cg['Credit'] > 0, cg['Credit'], cg['Debit']).sum() if len(cg) else 0.0
            row[f'{cat}_Amount'] = float(amt); row[f'{cat}_Count'] = int(len(cg))
        rows.append(row)
    daily = pd.DataFrame(rows)
    daily = daily[~((daily['Total_Debit'] == 0) & (daily['Total_Credit'] == 0)) & (daily['Transaction_Count'] > 0)].reset_index(drop=True)
    dates = pd.to_datetime(daily[DATE_COL])
    daily['DayOfWeek'] = dates.dt.day_of_week; daily['Month'] = dates.dt.month; daily['WeekOfMonth'] = ((dates.dt.day - 1)//7)+1; daily['Quarter'] = dates.dt.quarter; daily['IsWeekend'] = daily['DayOfWeek'].isin([5,6]).astype(int)
    daily['Average_Withdrawal_Size'] = np.where(daily['Withdrawal_Count'] > 0, daily['Withdrawal_Amount']/daily['Withdrawal_Count'], 0.0)
    daily['Average_Deposit_Size'] = np.where(daily['Deposit_Count'] > 0, daily['Deposit_Amount']/daily['Deposit_Count'], 0.0)
    daily['Credit_Debit_Ratio'] = np.where(daily['Total_Debit'] > 0, daily['Total_Credit']/daily['Total_Debit'], np.where(daily['Total_Credit'] > 0, 999.0, 0.0))
    daily['Has_Deposit'] = (daily['Deposit_Amount'] > 0).astype(int)
    for n in [1,7,30]:
        daily[f'Lag_{n}_Withdrawal_Amount'] = daily['Withdrawal_Amount'].shift(n).fillna(0)
        daily[f'Lag_{n}_Deposit_Amount'] = daily['Deposit_Amount'].shift(n).fillna(0)
    daily['Rolling_7_Day_Withdrawal_Amount'] = daily['Withdrawal_Amount'].shift(1).rolling(7, min_periods=1).mean().fillna(0)
    daily['Rolling_30_Day_Withdrawal_Amount'] = daily['Withdrawal_Amount'].shift(1).rolling(30, min_periods=1).mean().fillna(0)
    daily['Rolling_7_Day_Deposit_Amount'] = daily['Deposit_Amount'].shift(1).rolling(7, min_periods=1).mean().fillna(0)
    daily['Rolling_30_Day_Deposit_Amount'] = daily['Deposit_Amount'].shift(1).rolling(30, min_periods=1).mean().fillna(0)
    daily[TARGET_WITHDRAWAL] = daily['Withdrawal_Amount'].shift(-1)
    daily[TARGET_DEPOSIT] = daily['Deposit_Amount'].shift(-1)
    daily[TARGET_HAS_DEPOSIT] = daily['Has_Deposit'].shift(-1)
    daily = daily.dropna(subset=[TARGET_WITHDRAWAL, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT])
    daily[TARGET_HAS_DEPOSIT] = daily[TARGET_HAS_DEPOSIT].astype(int)
    return validate_schema(daily[ORDERED_COLUMNS])

def process_dataset(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    df = load_input(data) if not isinstance(data, pd.DataFrame) else data.copy()
    kind = detect_dataset_type(df)
    if kind == 'RAW_STATEMENT': return process_raw_statement(df)
    if kind in {'FEATURE_ENGINEERED','PARTIALLY_ENGINEERED'}:
        repaired, _ = attempt_auto_repair(df)
        return validate_schema(repaired)
    raise ValueError('Unrecognised file format. Please upload a Moniepoint bank statement or canonical engineered CSV.')

def main(path: str, out_path: str | None = None) -> None:
    out = process_dataset(path)
    print(f"Validated dataset: rows={out.shape[0]} columns={out.shape[1]}")
    if out_path:
        out.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('path'); parser.add_argument('--out')
    args = parser.parse_args(); main(args.path, args.out)

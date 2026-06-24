"""Preprocessing pipeline for raw statements and canonical engineered datasets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import DATE_COL, FEATURE_COLS, ROOT_DIR, SCHEMA_PATH, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT, TARGET_WITHDRAWAL
from src.utils.dataset_detector import attempt_auto_repair, detect_dataset_type

CATEGORIES = ["Withdrawal", "Deposit", "Airtime", "BillPayment", "WalletTransfer", "Transfer", "Levy", "Commission", "Reversal", "Insurance", "Other"]
ORDERED_COLUMNS = [DATE_COL, *FEATURE_COLS[:-1], TARGET_WITHDRAWAL, TARGET_DEPOSIT, "Has_Deposit", TARGET_HAS_DEPOSIT]

def classify_narration(narration: str, reference: str, debit: float, credit: float) -> str:
    n = str(narration or "").upper(); r = str(reference or "").upper()
    if 'RVSL' in r: return 'Reversal'
    if r.startswith('SAV') or 'ELECTRONIC MONEY TRANSFER LEVY' in n or 'VALUE ADDED TAX' in n: return 'Levy'
    if 'AIRTIME' in n or r.startswith('ATP'): return 'Airtime'
    if 'BILL_PAYMENT' in n or 'BILL PAYMENT' in n or r.startswith('BPT'): return 'BillPayment'
    if 'INSURANCE' in n or r.startswith('ISP'): return 'Insurance'
    if 'MONIEPOINT REWARDS' in n or 'CASHOUT' in n or 'COMMISSION' in n or r.startswith('ARF') or r.startswith('RCT'): return 'Commission'
    if r.startswith('WTH') or r.startswith('PUR') or 'MP-WTH' in n or 'WITHDRAWAL FOR' in n or 'POSTING FROM' in n or n == 'WITHDRAWAL FROM *****11200 TO *****26078': return 'Withdrawal'
    if 'DEPOSIT FROM *****26078 TO *****11200' in n or (r.startswith('DEP') and debit > 0) or (r.startswith('TRF') and 'DEPOSIT FROM' in n and debit > 0): return 'Deposit'
    if 'WALLET TRANSFER' in n or 'CARD_TRANSFER' in n or r.startswith('WFT') or r.startswith('MIT') or r.startswith('LTT') or r.startswith('LPT') or r.startswith('WCL'): return 'WalletTransfer'
    if 'USSD' in n or 'FBNMOBILE' in n or n.startswith('TRF') or n.startswith('- FROM') or 'WALLET FUND TRANSFER FROM' in n or r.startswith('FND') or (r.startswith('TRF') and credit > 0 and 'DEPOSIT' not in n) or 'TRANSFER FROM' in n or ('MOBILE' in n and 'TRANSFER' in n): return 'Transfer'
    return 'Other'

def load_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    return pd.read_csv(path)

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

def process_raw_statement(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['Debit','Credit']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    if 'Balance' not in df.columns: df['Balance'] = 0.0
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').ffill().fillna(0.0)
    date_col = _date_column(df)
    df[DATE_COL] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    df = df.dropna(subset=[DATE_COL])
    df = df[~((df['Debit'] == 0) & (df['Credit'] == 0))]
    non_rvsl = ~df['Reference'].astype(str).str.upper().str.contains('RVSL', na=False)
    df = pd.concat([df[non_rvsl].drop_duplicates(['Reference','Debit','Credit'], keep='first'), df[~non_rvsl]], ignore_index=True).sort_values(DATE_COL)
    df['Transaction_Category'] = df.apply(lambda r: classify_narration(r.get('Narration',''), r.get('Reference',''), r['Debit'], r['Credit']), axis=1)
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
    daily['DayOfWeek'] = dates.dt.dayofweek; daily['Month'] = dates.dt.month; daily['WeekOfMonth'] = ((dates.dt.day - 1)//7)+1; daily['Quarter'] = dates.dt.quarter; daily['IsWeekend'] = daily['DayOfWeek'].isin([5,6]).astype(int)
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

def main(path: str) -> None:
    out = process_dataset(path)
    print(f"Validated dataset: rows={out.shape[0]} columns={out.shape[1]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('path')
    main(parser.parse_args().path)

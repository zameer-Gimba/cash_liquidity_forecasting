"""Generate reference_schema.json from the immutable canonical dataset."""
from __future__ import annotations
import csv, json
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src.utils.config import CANONICAL_DATASET_PATH, DATE_COL, FEATURE_COLS, SCHEMA_PATH, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT, TARGET_WITHDRAWAL
TARGET_COLUMNS = [TARGET_WITHDRAWAL, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT]
INT_COLS = {c for c in FEATURE_COLS if c.endswith('_Count')} | {'Transaction_Count','DayOfWeek','Month','WeekOfMonth','Quarter','IsWeekend','Has_Deposit',TARGET_HAS_DEPOSIT}

def _infer_with_pandas(path: Path):
    import pandas as pd
    df = pd.read_csv(path)
    return list(df.columns), {col: str(dtype) for col, dtype in df.dtypes.items()}, [int(df.shape[0]), int(df.shape[1])]

def _infer_with_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        reader=csv.reader(f); header=next(reader); rows=sum(1 for _ in reader)
    dtypes={c: ('object' if c == DATE_COL else 'int64' if c in INT_COLS else 'float64') for c in header}
    return header, dtypes, [rows, len(header)]

def main() -> None:
    dataset_path = ROOT_DIR / CANONICAL_DATASET_PATH
    try: required_columns, dtypes, shape = _infer_with_pandas(dataset_path)
    except Exception: required_columns, dtypes, shape = _infer_with_csv(dataset_path)
    missing_features = [c for c in FEATURE_COLS if c not in required_columns]
    missing_targets = [c for c in TARGET_COLUMNS if c not in required_columns]
    if missing_features or missing_targets or DATE_COL not in required_columns:
        raise ValueError(f"Canonical dataset schema mismatch: missing_features={missing_features}, missing_targets={missing_targets}")
    schema = {"version":"1.0.0","date_column":DATE_COL,"feature_columns":FEATURE_COLS,"target_columns":TARGET_COLUMNS,"required_columns":required_columns,"column_dtypes":dtypes,"canonical_shape":shape}
    out = ROOT_DIR / SCHEMA_PATH; out.write_text(json.dumps(schema, indent=2)+"\n", encoding='utf-8')
    print(f"Wrote {out} with {len(required_columns)} required columns and shape {shape}")
if __name__ == '__main__': main()

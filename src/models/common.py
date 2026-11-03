"""Shared model training and evaluation helpers for canonical liquidity data."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from src.utils.config import CANONICAL_DATASET_PATH, CANONICAL_TRAINING_END_DATE, DATE_COL, FEATURE_COLS, MODEL_DIR, RANDOM_STATE, ROOT_DIR, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT, TARGET_WITHDRAWAL, TRAIN_SPLIT, VAL_SPLIT

def load_canonical_data(path: str | Path | None = None) -> pd.DataFrame:
    p = ROOT_DIR / (path or CANONICAL_DATASET_PATH)
    df = pd.read_csv(p, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    missing = [c for c in [DATE_COL, *FEATURE_COLS, TARGET_WITHDRAWAL, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT] if c not in df.columns]
    if missing: raise KeyError(f"Missing canonical columns: {missing}")
    if path is None:
        # Exclude the canonical file's incomplete trailing window (see config.py) from training/evaluation.
        df = df[df[DATE_COL] <= CANONICAL_TRAINING_END_DATE].reset_index(drop=True)
    return df

def chronological_split(df: pd.DataFrame):
    n = len(df); train_end = int(n * TRAIN_SPLIT); val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5), "MAPE": float(np.nanmean(np.abs((y_true-y_pred)/denom))*100 if np.isfinite(denom).any() else 0.0), "R2": float(r2_score(y_true, y_pred))}

def classifier_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    m = {"Accuracy": float(accuracy_score(y_true, y_pred)), "Precision": float(precision_score(y_true, y_pred, zero_division=0)), "Recall": float(recall_score(y_true, y_pred, zero_division=0)), "F1": float(f1_score(y_true, y_pred, zero_division=0))}
    try: m["ROC_AUC"] = float(roc_auc_score(y_true, y_score if y_score is not None else y_pred))
    except Exception: m["ROC_AUC"] = 0.0
    return m

def save_pickle(model, name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    joblib.dump(model, path)
    return path

def save_results(rows: Iterable[dict], path: str | Path) -> pd.DataFrame:
    out = pd.DataFrame(list(rows)); p = ROOT_DIR / path; p.parent.mkdir(parents=True, exist_ok=True); out.to_csv(p, index=False); return out

def x_y(df: pd.DataFrame, target: str = TARGET_WITHDRAWAL):
    return df[FEATURE_COLS], df[target]

def fit_eval_split(df: pd.DataFrame):
    """(fit, test): the validation slice is not used for tuning, so models are fit on train+val."""
    train, val, test = chronological_split(df)
    return pd.concat([train, val]), test

def naive_baselines(test: pd.DataFrame, amount_col: str, target: str, full: pd.DataFrame | None = None) -> list[dict]:
    """Simple rules a forecast must beat, scored on the same test rows.

    Each row's target is the *next* active day's amount, so:
    - "Same as today":       today's amount
    - "Same weekday last week": the amount 6 active days before today (≈ same weekday as tomorrow)
    - "7-day average":       mean of the last 7 active days including today
    """
    full = test if full is None else full
    series = full[amount_col]
    preds = {
        'Baseline: same as today': series,
        'Baseline: same weekday last week': series.shift(6),
        'Baseline: 7-day average': series.rolling(7, min_periods=1).mean(),
    }
    rows = []
    for label, pred in preds.items():
        p = pred.reindex(test.index).fillna(series.reindex(test.index))
        rows.append({'Model': label, **regression_metrics(test[target], p)})
    return rows

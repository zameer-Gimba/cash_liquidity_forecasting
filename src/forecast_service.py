"""Forecasting service used by the REST API.

It turns a daily liquidity dataset (the canonical 50-column schema, or just
daily withdrawal and deposit totals) into a cash and float recommendation
for a future date. It holds no web framework code, so it can be reused by
other front ends.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.predict_deposit import predict_deposit
from src.models.session_forecast import build_future_feature_row, forecast_path, predict_withdrawal_for_date, train_withdrawal_model
from src.utils.config import (
    CANONICAL_DATASET_PATH,
    CANONICAL_TRAINING_END_DATE,
    DATE_COL,
    FEATURE_COLS,
    MODEL_DIR,
    ROOT_DIR,
    SAFETY_BUFFER_DEPOSIT,
    SAFETY_BUFFER_WITHDRAWAL,
    TARGET_DEPOSIT,
    TARGET_HAS_DEPOSIT,
    TARGET_WITHDRAWAL,
)

# statsmodels notes when it falls back to default start parameters; it is not actionable here.
warnings.filterwarnings("ignore", module="statsmodels")

# The LSTM is left out: it is slow, needs TensorFlow and was the least accurate model in testing.
MODELS: dict[str, str] = {
    "sarima": "SARIMA",
    "arima": "ARIMA",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}
DEFAULT_MODEL = "sarima"
STATESPACE = {"sarima", "arima"}
MIN_DAYS = 60
STALE_AFTER_DAYS = 30
MAX_HORIZON_DAYS = 730
PATH_DAYS = 14


class ForecastError(ValueError):
    """Input that cannot be forecast (reported to API clients as a 422)."""


@dataclass
class Dataset:
    daily: pd.DataFrame          # canonical columns, TransactionDate as Timestamp, sorted
    key: str                     # content hash, used to cache fitted models
    complete: bool               # True when all 46 features are real (not just totals)
    source: str


# ------------------------------------------------------------------ datasets


def _hash(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def dataset_from_canonical(df: pd.DataFrame, source: str) -> Dataset:
    missing = [c for c in [DATE_COL, *FEATURE_COLS] if c not in df.columns]
    if missing:
        raise ForecastError(f"Dataset is missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    daily = df.copy()
    daily[DATE_COL] = pd.to_datetime(daily[DATE_COL])
    daily = daily.sort_values(DATE_COL).reset_index(drop=True)
    if len(daily) < MIN_DAYS:
        raise ForecastError(f"Only {len(daily)} trading days; at least {MIN_DAYS} are needed to forecast.")
    return Dataset(daily, _hash(daily[[DATE_COL, "Withdrawal_Amount", "Deposit_Amount"]]), True, source)


_demo: Dataset | None = None


def demo_dataset() -> Dataset:
    """The anonymised demo agent (daily totals only), up to the end of normal trading."""
    global _demo
    if _demo is None:
        df = pd.read_csv(ROOT_DIR / CANONICAL_DATASET_PATH)
        df = df[df[DATE_COL] <= CANONICAL_TRAINING_END_DATE]
        _demo = dataset_from_canonical(df, "demo")
        _demo.key = "demo"
    return _demo


def dataset_from_totals(rows: list[dict]) -> Dataset:
    """Build a dataset from daily totals only: [{date, withdrawals, deposits}, ...].

    Only the withdrawal and deposit series are known, so the other features are
    zero and only the time-series models (SARIMA, ARIMA) may be used with it.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        raise ForecastError("No daily rows supplied.")
    df[DATE_COL] = pd.to_datetime(df["date"], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ForecastError("Every row needs a valid date (YYYY-MM-DD).")
    if df[DATE_COL].duplicated().any():
        raise ForecastError("Dates must be unique: send one row per day.")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    w = pd.to_numeric(df["withdrawals"], errors="coerce")
    d = pd.to_numeric(df["deposits"], errors="coerce")
    if w.isna().any() or d.isna().any() or (w < 0).any() or (d < 0).any():
        raise ForecastError("withdrawals and deposits must be non-negative numbers.")

    daily = pd.DataFrame(0.0, index=df.index, columns=FEATURE_COLS)
    daily.insert(0, DATE_COL, df[DATE_COL])
    daily["Withdrawal_Amount"], daily["Deposit_Amount"] = w, d
    daily["Total_Credit"], daily["Total_Debit"] = w, d
    daily["Net_Flow"] = w - d
    dates = daily[DATE_COL]
    daily["DayOfWeek"] = dates.dt.day_of_week
    daily["Month"] = dates.dt.month
    daily["WeekOfMonth"] = (dates.dt.day - 1) // 7 + 1
    daily["Quarter"] = dates.dt.quarter
    daily["IsWeekend"] = daily["DayOfWeek"].isin([5, 6]).astype(int)
    daily["Has_Deposit"] = (d > 0).astype(int)
    for n in (1, 7, 30):
        daily[f"Lag_{n}_Withdrawal_Amount"] = w.shift(n).fillna(0)
        daily[f"Lag_{n}_Deposit_Amount"] = d.shift(n).fillna(0)
    for n in (7, 30):
        daily[f"Rolling_{n}_Day_Withdrawal_Amount"] = w.shift(1).rolling(n, min_periods=1).mean().fillna(0)
        daily[f"Rolling_{n}_Day_Deposit_Amount"] = d.shift(1).rolling(n, min_periods=1).mean().fillna(0)
    # Targets are only used by the tree models, which this dataset does not allow.
    daily[TARGET_WITHDRAWAL] = w.shift(-1).fillna(0)
    daily[TARGET_DEPOSIT] = d.shift(-1).fillna(0)
    daily[TARGET_HAS_DEPOSIT] = (d.shift(-1) > 0).astype(int)
    if len(daily) < MIN_DAYS:
        raise ForecastError(f"Only {len(daily)} days supplied; at least {MIN_DAYS} are needed to forecast.")
    return Dataset(daily, _hash(daily[[DATE_COL, "Withdrawal_Amount", "Deposit_Amount"]]), False, "daily totals")


# ------------------------------------------------------------------ models


class _ModelCache:
    """Small thread-safe LRU of fitted models, keyed by (dataset, model, series)."""

    def __init__(self, size: int = 24):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key, build):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        value = build()
        with self.lock:
            self.items[key] = value
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return value


_cache = _ModelCache()
_deposit_models: Any = None


def _fit(ds: Dataset, model_key: str, series: str):
    # train_withdrawal_model fits on Withdrawal_Amount; for the deposit series the
    # deposit column is swapped in (time-series models only use that one column).
    def build():
        frame = ds.daily if series == "withdrawal" else ds.daily.assign(Withdrawal_Amount=ds.daily["Deposit_Amount"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels start-parameter notices
            return train_withdrawal_model(model_key, frame)
    return _cache.get((ds.key, model_key, series), build)


def _saved_deposit_models():
    global _deposit_models
    if _deposit_models is None:
        _deposit_models = (joblib.load(MODEL_DIR / "rf_deposit_classifier.pkl"), joblib.load(MODEL_DIR / "rf_deposit_regressor.pkl"))
    return _deposit_models


def test_results() -> dict[str, dict]:
    """Held-out test error per model (demo agent, 7 Jun 2025 – 16 Jan 2026)."""
    path = ROOT_DIR / "reports" / "results" / "withdrawal_model_comparison.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path).dropna(subset=["MAE"])
    best_rule = table[table["Model"].str.startswith("Baseline")]["MAE"].min()
    out = {}
    for row in table.itertuples():
        out[row.Model] = {"mae": round(float(row.MAE), 2), "vs_best_simple_rule_pct": round((row.MAE / best_rule - 1) * 100, 1)}
    return out


# ------------------------------------------------------------------ forecast


def forecast(ds: Dataset, target: dt.date | None = None, model_key: str = DEFAULT_MODEL, include_path: bool = False) -> dict:
    if model_key not in MODELS:
        raise ForecastError(f"Unknown model '{model_key}'. Choose one of: {', '.join(MODELS)}.")
    if not ds.complete and model_key not in STATESPACE:
        raise ForecastError("Daily totals only support the time-series models (sarima, arima). "
                            "Upload a statement to use the tree models.")
    data = ds.daily
    last = data[DATE_COL].max()
    target_ts = pd.Timestamp(target or (dt.date.today() + dt.timedelta(days=1)))
    horizon = (target_ts - last).days
    if horizon < 1:
        raise ForecastError(f"The forecast date must be after the last data date ({last:%Y-%m-%d}).")
    if horizon > MAX_HORIZON_DAYS:
        raise ForecastError(f"The forecast date is {horizon} days past the data; the limit is {MAX_HORIZON_DAYS}.")

    model, extra = _fit(ds, model_key, "withdrawal")
    withdrawal = max(float(predict_withdrawal_for_date(model_key, model, extra, data, target_ts)), 0.0)

    if model_key in STATESPACE:
        dmodel, dextra = _fit(ds, model_key, "deposit")
        deposit = max(float(predict_withdrawal_for_date(model_key, dmodel, dextra, data, target_ts)), 0.0)
        deposit_method = f"{MODELS[model_key]} on daily deposits"
    else:
        deposit = float(predict_deposit(build_future_feature_row(data, target_ts), *_saved_deposit_models()))
        deposit_method = "Two-stage Random Forest (will there be a deposit, then how much)"

    recent = data.tail(90)["Withdrawal_Amount"]
    q25, q75 = recent.quantile([0.25, 0.75])
    demand = "busy" if withdrawal > q75 else "quiet" if withdrawal < q25 else "normal"

    result = {
        "date": target_ts.date().isoformat(),
        "weekday": f"{target_ts:%A}",
        "model": MODELS[model_key],
        "data": {"source": ds.source, "trading_days": int(len(data)),
                 "first_date": f"{data[DATE_COL].min():%Y-%m-%d}", "last_date": f"{last:%Y-%m-%d}"},
        "horizon_days": int(horizon),
        "withdrawals": {"expected": round(withdrawal, 2), "safety_buffer": SAFETY_BUFFER_WITHDRAWAL,
                        "cash_to_hold": round(withdrawal * (1 + SAFETY_BUFFER_WITHDRAWAL), 2)},
        "deposits": {"expected": round(deposit, 2), "safety_buffer": SAFETY_BUFFER_DEPOSIT,
                     "float_to_hold": round(deposit * (1 + SAFETY_BUFFER_DEPOSIT), 2), "method": deposit_method},
        "demand_level": demand,
        "demand_level_basis": "Expected cash-outs compared with the last 90 trading days (below 25th percentile = quiet, above 75th = busy).",
        "warnings": [],
        "currency": "NGN",
    }
    if horizon > STALE_AFTER_DAYS:
        result["warnings"].append(
            f"The forecast date is {horizon} days after the last data. That far ahead the forecast only carries "
            "the weekly pattern and the last known level; send recent data for an up-to-date forecast."
        )
    if ds.key == "demo":
        result["typical_error"] = test_results().get(MODELS[model_key])
    if include_path:
        start = max(last + pd.Timedelta(days=1), target_ts - pd.Timedelta(days=PATH_DAYS - 1))
        dates = list(pd.date_range(start, target_ts))
        path = forecast_path(model_key, model, extra, data, dates)
        result["path"] = [{"date": f"{d:%Y-%m-%d}", "expected_withdrawals": round(float(v), 2)}
                          for d, v in path.items() if np.isfinite(v)]
    return result

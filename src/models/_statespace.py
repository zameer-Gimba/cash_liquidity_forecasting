"""Rolling one-step-ahead evaluation shared by the ARIMA and SARIMA scripts."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.common import fit_eval_split, load_canonical_data, regression_metrics, save_pickle
from src.utils.config import DATE_COL, TARGET_WITHDRAWAL


def _daily(df: pd.DataFrame, column: str) -> pd.Series:
    # Inactive days are carried forward so the model sees a regular daily index.
    return df.set_index(DATE_COL)[column].asfreq('D').ffill()


def evaluate(build, pickle_name: str, **fit_kwargs) -> dict[str, float]:
    """Fit on train+val, then forecast each test day one step ahead using all data up to it.

    Each canonical row's target is the next active day's withdrawal, so the
    series used here is the day's own ``Withdrawal_Amount`` and the forecast
    for a row is the model's one-step prediction for the following calendar day
    (the same quantity the feature-based models predict).
    """
    df = load_canonical_data()
    fit_df, test = fit_eval_split(df)
    full = _daily(df, 'Withdrawal_Amount')
    cutoff = fit_df[DATE_COL].max()
    result = build(full[:cutoff]).fit(**fit_kwargs)
    save_pickle(result, pickle_name)
    if test.empty:
        return {}
    extended = result.apply(full, refit=False)
    one_step = extended.predict(start=full.index[1], end=full.index[-1] + pd.Timedelta(days=1))
    # Prediction for "the day after row d" -> look up the next active date's forecast.
    dates = pd.to_datetime(df[DATE_COL])
    next_dates = dates.shift(-1).reindex(test.index)
    preds = one_step.reindex(next_dates.to_numpy()).to_numpy(dtype=float)
    mask = ~np.isnan(preds)
    return regression_metrics(test[TARGET_WITHDRAWAL].to_numpy()[mask], preds[mask])

"""Tree regressors that forecast relative to the recent level.

Cash demand at this agent fell sharply in 2025. Trees cannot extrapolate below
the range they were trained on, so a model fit on 2022-2024 volumes keeps
predicting 2024-sized days. Dividing the target and the money-valued features
by the current 7-day average makes the model learn *shape* (weekday effects,
spikes, momentum) and lets the level follow the data. The wrapper computes
that level from the canonical features alone, so ``predict(X)`` takes the same
46 columns as any other model and pickles cleanly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from src.utils.config import FEATURE_COLS, RANDOM_STATE

MONEY_COLS = [c for c in FEATURE_COLS if c.endswith('_Amount') or c.endswith('_Size')
              or c in ('Total_Debit', 'Total_Credit', 'Net_Flow', 'Closing_Balance')]


class LevelScaledRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, amount: str = 'Withdrawal'):
        self.estimator = estimator
        self.amount = amount

    def level(self, X: pd.DataFrame) -> np.ndarray:
        a = self.amount
        # 7-day mean including today = (previous-7-day mean * 7 - value 7 days ago + today) / 7
        lvl = (X[f'Rolling_7_Day_{a}_Amount'] * 7 - X[f'Lag_7_{a}_Amount'] + X[f'{a}_Amount']) / 7
        lvl = np.maximum(lvl.to_numpy(dtype=float), 0.0)
        # Quiet spells (level near zero) would turn ordinary days into huge ratios, so the
        # level is floored at a tenth of the typical training level.
        return np.maximum(lvl, getattr(self, 'floor_', 0.0)) + 1.0

    def _scale(self, X: pd.DataFrame, lvl: np.ndarray) -> pd.DataFrame:
        Xs = X[FEATURE_COLS].astype(float).copy()
        Xs[MONEY_COLS] = Xs[MONEY_COLS].div(lvl, axis=0)
        return Xs

    def fit(self, X, y):
        self.floor_ = 0.0
        self.floor_ = 0.1 * float(np.median(self.level(X)))
        lvl = self.level(X)
        self.estimator_ = clone(self.estimator).fit(self._scale(X, lvl), np.asarray(y, dtype=float) / lvl)
        return self

    def predict(self, X):
        lvl = self.level(X)
        return np.maximum(self.estimator_.predict(self._scale(X, lvl)) * lvl, 0.0)


def tree_regressor(key: str, amount: str = 'Withdrawal') -> LevelScaledRegressor:
    if key == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        base = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    elif key == 'xgb':
        from xgboost import XGBRegressor
        base = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                            objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=2)
    elif key == 'lgbm':
        from lightgbm import LGBMRegressor
        base = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1)
    else:
        raise ValueError(f'Unknown tree model key: {key}')
    return LevelScaledRegressor(base, amount)

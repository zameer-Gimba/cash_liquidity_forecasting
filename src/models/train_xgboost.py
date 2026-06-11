"""Train a tuned XGBoost liquidity demand model."""
from __future__ import annotations

import argparse
import importlib.util

from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.common import load_feature_data, train_tree_regressor
from src.utils.config import RANDOM_STATE


def estimator_and_params():
    """Return XGBoost estimator if installed, otherwise sklearn fallback with compatible tuning."""
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        return XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1), {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        }
    return HistGradientBoostingRegressor(random_state=RANDOM_STATE), {
        "model__max_iter": [100, 200],
        "model__max_leaf_nodes": [15, 31],
        "model__learning_rate": [0.03, 0.05, 0.1],
    }


def main(data_path: str, n_iter: int = 10) -> dict[str, float]:
    """Train XGBoostRegressor or fallback estimator and return test metrics."""
    estimator, params = estimator_and_params()
    return train_tree_regressor(load_feature_data(data_path), estimator, params, "xgboost", n_iter=n_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--n-iter", type=int, default=10)
    print(main(**vars(parser.parse_args())))

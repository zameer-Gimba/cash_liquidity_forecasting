"""Train a tuned Random Forest liquidity demand model."""
from __future__ import annotations

import argparse

from src.models.common import default_random_forest_params, default_random_forest_regressor, load_feature_data, train_tree_regressor


def main(data_path: str, n_iter: int = 10) -> dict[str, float]:
    """Train RandomForestRegressor and return test metrics."""
    df = load_feature_data(data_path)
    return train_tree_regressor(df, default_random_forest_regressor(), default_random_forest_params(), "random_forest", n_iter=n_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--n-iter", type=int, default=10)
    print(main(**vars(parser.parse_args())))

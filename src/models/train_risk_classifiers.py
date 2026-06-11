"""Train liquidity risk classifiers."""
from __future__ import annotations

import argparse
import importlib.util

from sklearn.ensemble import RandomForestClassifier

from src.models.common import load_feature_data, train_risk_classifier
from src.utils.config import RANDOM_STATE


def train_random_forest_risk(data_path: str) -> dict[str, float]:
    """Train RandomForestClassifier for Low/Medium/High liquidity risk."""
    estimator = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    return train_risk_classifier(load_feature_data(data_path), estimator, "risk_random_forest")


def train_xgboost_risk(data_path: str) -> dict[str, float]:
    """Train XGBoostClassifier when installed, otherwise RandomForest fallback for risk classification."""
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model_name = "risk_xgboost"
    else:
        estimator = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
        model_name = "risk_xgboost_fallback_random_forest"
    return train_risk_classifier(load_feature_data(data_path), estimator, model_name)


def main(data_path: str) -> dict[str, dict[str, float]]:
    """Train all configured risk classifiers."""
    return {
        "risk_random_forest": train_random_forest_risk(data_path),
        "risk_xgboost": train_xgboost_risk(data_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    print(main(parser.parse_args().data_path))

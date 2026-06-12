"""Shared model training helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.preprocessing.split_data import chronological_split, get_time_series_cv
from src.utils.config import MODEL_DIR, RANDOM_STATE, TARGET_COLUMN
from src.utils.metrics import classification_metrics, regression_metrics, save_metrics

DATE_COLUMNS = {"TransactionDate"}
EXCLUDED_COLUMNS = {TARGET_COLUMN, "Liquidity_Risk", *DATE_COLUMNS}


def load_feature_data(path: str | Path) -> pd.DataFrame:
    """Load a feature-engineered CSV dataset."""
    df = pd.read_csv(path, parse_dates=["TransactionDate"])
    return df.sort_values("TransactionDate").reset_index(drop=True)


def feature_columns(df: pd.DataFrame, target: str = TARGET_COLUMN) -> list[str]:
    """Return model feature columns, excluding target/date/leakage columns."""
    excluded = set(EXCLUDED_COLUMNS)
    excluded.add(target)
    return [column for column in df.columns if column not in excluded]


def build_tree_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    """Build a preprocessing transformer: one-hot encode DayOfWeek; do not scale numeric features."""
    categorical = [column for column in features if df[column].dtype == "object" or column == "DayOfWeek"]
    numeric = [column for column in features if column not in categorical]
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def train_tree_regressor(
    df: pd.DataFrame,
    estimator: Any,
    param_distributions: dict[str, Any],
    model_name: str,
    n_iter: int = 10,
) -> dict[str, float]:
    """Train, tune, evaluate, and save a tree-based regression model."""
    splits = chronological_split(df)
    features = feature_columns(df)
    preprocessor = build_tree_preprocessor(df, features)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=get_time_series_cv(5),
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    train_validation = pd.concat([splits.train, splits.validation]).reset_index(drop=True)
    search.fit(train_validation[features], train_validation[TARGET_COLUMN])
    predictions = search.predict(splits.test[features])
    metrics = regression_metrics(splits.test[TARGET_COLUMN], predictions)
    metrics["Best_Params"] = search.best_params_
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_DIR / f"{model_name}.joblib")
    save_metrics(metrics, Path("reports/results") / f"{model_name}_metrics.json")
    return metrics


def train_risk_classifier(df: pd.DataFrame, estimator: Any | None = None, model_name: str = "risk_random_forest") -> dict[str, float]:
    """Train and save a liquidity-risk classifier."""
    estimator = estimator or RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced")
    splits = chronological_split(df.dropna(subset=["Liquidity_Risk"]))
    features = feature_columns(df, target="Liquidity_Risk")
    pipeline = Pipeline([("preprocess", build_tree_preprocessor(df, features)), ("model", estimator)])
    target_train = splits.train["Liquidity_Risk"]
    target_test = splits.test["Liquidity_Risk"]
    label_encoder: LabelEncoder | None = None
    if estimator.__class__.__name__ == "XGBClassifier":
        label_encoder = LabelEncoder()
        target_train = pd.Series(label_encoder.fit_transform(target_train), index=target_train.index)
    pipeline.fit(splits.train[features], target_train)
    predictions = pipeline.predict(splits.test[features])
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions.astype(int))
    metrics = classification_metrics(target_test, predictions)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact: Any = {"pipeline": pipeline, "label_encoder": label_encoder} if label_encoder is not None else pipeline
    joblib.dump(artifact, MODEL_DIR / f"{model_name}.joblib")
    save_metrics(metrics, Path("reports/results") / f"{model_name}_metrics.json")
    return metrics


def default_random_forest_params() -> dict[str, list[Any]]:
    """Parameter search space for RandomForestRegressor."""
    return {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }


def default_random_forest_regressor() -> RandomForestRegressor:
    """Return a reproducible RandomForestRegressor."""
    return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

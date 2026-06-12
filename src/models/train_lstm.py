"""Train LSTM model with MinMaxScaler for next-day cash demand."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from src.models.common import feature_columns, load_feature_data
from src.preprocessing.split_data import chronological_split
from src.utils.config import MODEL_DIR, RANDOM_STATE, TARGET_COLUMN
from src.utils.metrics import regression_metrics, save_metrics


def main(data_path: str, lookback: int = 7, epochs: int = 20) -> dict[str, float]:
    """Train TensorFlow LSTM when available, otherwise an MLP fallback using MinMaxScaler."""
    df = load_feature_data(data_path)
    splits = chronological_split(df)
    features = feature_columns(df)
    numeric_features = [column for column in features if df[column].dtype.kind in "biufc"]
    if not numeric_features:
        raise ValueError("No numeric features available for LSTM training. Ensure the feature dataset contains numeric columns.")
    scaler = MinMaxScaler()
    train_validation = np.vstack([splits.train[numeric_features].values, splits.validation[numeric_features].values])
    scaler.fit(train_validation)
    x_train = scaler.transform(train_validation)
    y_train = np.r_[splits.train[TARGET_COLUMN].values, splits.validation[TARGET_COLUMN].values]
    x_test = scaler.transform(splits.test[numeric_features].values)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("tensorflow") is not None:
        keras = importlib.import_module("tensorflow.keras")

        model = keras.Sequential([keras.layers.Input(shape=(1, x_train.shape[1])), keras.layers.LSTM(32), keras.layers.Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train.reshape((x_train.shape[0], 1, x_train.shape[1])), y_train, epochs=epochs, verbose=0)
        predictions = model.predict(x_test.reshape((x_test.shape[0], 1, x_test.shape[1])), verbose=0).ravel()
        model.save(MODEL_DIR / "lstm.keras")
    else:
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        joblib.dump(model, MODEL_DIR / "lstm_fallback_mlp.joblib")
    joblib.dump({"scaler": scaler, "features": numeric_features, "lookback": lookback}, MODEL_DIR / "lstm_scaler.joblib")
    metrics = regression_metrics(splits.test[TARGET_COLUMN], predictions)
    save_metrics(metrics, Path("reports/results/lstm_metrics.json"))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    print(main(args.data_path, epochs=args.epochs))

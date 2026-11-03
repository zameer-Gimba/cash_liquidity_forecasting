from __future__ import annotations
import joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from src.models.common import fit_eval_split, load_canonical_data, regression_metrics
from src.utils.config import FEATURE_COLS, LSTM_SEQUENCE_LENGTH, MODEL_DIR, TARGET_WITHDRAWAL


def _seq(X, y, n=LSTM_SEQUENCE_LENGTH):
    return np.array([X[i-n:i] for i in range(n, len(X))]), np.array(y[n:])


def main() -> dict[str, float]:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    df = load_canonical_data()
    train, test = fit_eval_split(df)
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    xs, ys = _seq(scaler.transform(train[FEATURE_COLS]), train[TARGET_WITHDRAWAL].to_numpy())
    # Test sequences may use history from the fit period, so every test day is scored.
    xa, ya = _seq(scaler.transform(df[FEATURE_COLS]), df[TARGET_WITHDRAWAL].to_numpy())
    xst, yst = xa[-len(test):], ya[-len(test):]
    y_scale = float(ys.mean()) or 1.0
    model = tf.keras.Sequential([tf.keras.layers.Input((LSTM_SEQUENCE_LENGTH, len(FEATURE_COLS))), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse'); model.fit(xs, ys / y_scale, epochs=40, batch_size=32, verbose=0)
    pred = model.predict(xst, verbose=0).ravel() * y_scale
    MODEL_DIR.mkdir(parents=True, exist_ok=True); model.save(MODEL_DIR/'lstm_withdrawal.keras')
    joblib.dump({'scaler': scaler, 'y_scale': y_scale}, MODEL_DIR/'lstm_withdrawal_scaler.pkl')
    return regression_metrics(yst, pred)


if __name__ == '__main__': print(main())

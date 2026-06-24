from __future__ import annotations
import joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from src.models.common import chronological_split, load_canonical_data, regression_metrics
from src.utils.config import FEATURE_COLS, LSTM_SEQUENCE_LENGTH, MODEL_DIR, TARGET_WITHDRAWAL

def _seq(X, y, n=LSTM_SEQUENCE_LENGTH):
    return np.array([X[i-n:i] for i in range(n, len(X))]), np.array(y[n:])
def main() -> dict[str, float]:
    import tensorflow as tf
    train, _, test = chronological_split(load_canonical_data())
    scaler = StandardScaler().fit(train[FEATURE_COLS]); Xtr = scaler.transform(train[FEATURE_COLS]); Xte = scaler.transform(test[FEATURE_COLS])
    xs, ys = _seq(Xtr, train[TARGET_WITHDRAWAL].to_numpy()); xst, yst = _seq(Xte, test[TARGET_WITHDRAWAL].to_numpy())
    model = tf.keras.Sequential([tf.keras.layers.Input((LSTM_SEQUENCE_LENGTH, len(FEATURE_COLS))), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse'); model.fit(xs, ys, epochs=2, batch_size=32, verbose=0)
    pred = model.predict(xst, verbose=0).ravel() if len(xst) else []
    MODEL_DIR.mkdir(parents=True, exist_ok=True); model.save(MODEL_DIR/'lstm_withdrawal.keras'); joblib.dump(scaler, MODEL_DIR/'lstm_withdrawal_scaler.pkl')
    return regression_metrics(yst, pred) if len(yst) else {}
if __name__ == '__main__': print(main())

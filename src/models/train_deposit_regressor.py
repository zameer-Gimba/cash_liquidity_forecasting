from __future__ import annotations
import joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from src.models.common import fit_eval_split, load_canonical_data, naive_baselines, regression_metrics, save_pickle, x_y
from src.models.level_scaled import tree_regressor
from src.utils.config import FEATURE_COLS, LSTM_SEQUENCE_LENGTH, MODEL_DIR, TARGET_DEPOSIT


def _seq(X, y, n=LSTM_SEQUENCE_LENGTH):
    return np.array([X[i-n:i] for i in range(n, len(X))]), np.array(y[n:])


def main() -> list[dict]:
    """Stage 2: how much will be deposited tomorrow, given that a deposit happens.

    Trained and scored only on days whose *next* day has a deposit, which is
    what the two-stage forecast uses this model for.
    """
    df = load_canonical_data()
    df_deposit = df[df[TARGET_DEPOSIT] > 0]
    train, test = fit_eval_split(df_deposit)
    Xtr, ytr = x_y(train, TARGET_DEPOSIT); Xte, yte = x_y(test, TARGET_DEPOSIT)
    rows = naive_baselines(test, 'Deposit_Amount', TARGET_DEPOSIT, df)
    for name in ('rf', 'xgb', 'lgbm'):
        model = tree_regressor(name, 'Deposit').fit(Xtr, ytr)
        save_pickle(model, f'{name}_deposit_regressor.pkl')
        rows.append({'Model': name, **regression_metrics(yte, model.predict(Xte))})
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        scaler = StandardScaler().fit(Xtr)
        xs, ys = _seq(scaler.transform(Xtr), ytr.to_numpy())
        # Test sequences may use history from the fit period, so every test day is scored.
        Xall = scaler.transform(df_deposit[FEATURE_COLS]); yall = df_deposit[TARGET_DEPOSIT].to_numpy()
        xa, ya = _seq(Xall, yall); xst, yst = xa[-len(test):], ya[-len(test):]
        y_scale = float(ys.mean()) or 1.0
        lstm = tf.keras.Sequential([tf.keras.layers.Input((LSTM_SEQUENCE_LENGTH, len(FEATURE_COLS))), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
        lstm.compile(optimizer='adam', loss='mse'); lstm.fit(xs, ys / y_scale, epochs=40, batch_size=32, verbose=0)
        MODEL_DIR.mkdir(parents=True, exist_ok=True); lstm.save(MODEL_DIR/'lstm_deposit_regressor.keras')
        joblib.dump({'scaler': scaler, 'y_scale': y_scale}, MODEL_DIR/'lstm_deposit_regressor_scaler.pkl')
        rows.append({'Model': 'lstm', **regression_metrics(yst, lstm.predict(xst, verbose=0).ravel() * y_scale)})
    except Exception as exc:
        rows.append({'Model': 'lstm', 'Error': str(exc)})
    return rows


if __name__ == '__main__': print(main())

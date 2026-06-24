from __future__ import annotations
import joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.models.common import chronological_split, load_canonical_data, regression_metrics, save_pickle, x_y
from src.utils.config import FEATURE_COLS, LSTM_SEQUENCE_LENGTH, MODEL_DIR, RANDOM_STATE, TARGET_DEPOSIT

def _seq(X, y, n=LSTM_SEQUENCE_LENGTH):
    return np.array([X[i-n:i] for i in range(n, len(X))]), np.array(y[n:])

def main() -> list[dict]:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    df = load_canonical_data(); df_deposit = df[df['Deposit_Amount'] > 0].copy()
    train, _, test = chronological_split(df_deposit)
    Xtr, ytr = x_y(train, TARGET_DEPOSIT); Xte, yte = x_y(test, TARGET_DEPOSIT)
    models = {'rf': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10), 'xgb': XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=2), 'lgbm': LGBMRegressor(n_estimators=80, learning_rate=0.08, random_state=RANDOM_STATE, verbose=-1)}
    rows=[]
    for name, model in models.items():
        model.fit(Xtr, ytr); pred=model.predict(Xte); save_pickle(model, f'{name}_deposit_regressor.pkl'); rows.append({'Model': name, **regression_metrics(yte, pred)})
    try:
        import tensorflow as tf
        scaler = StandardScaler().fit(Xtr); xs, ys = _seq(scaler.transform(Xtr), ytr.to_numpy()); xst, yst = _seq(scaler.transform(Xte), yte.to_numpy())
        lstm = tf.keras.Sequential([tf.keras.layers.Input((LSTM_SEQUENCE_LENGTH, len(FEATURE_COLS))), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
        lstm.compile(optimizer='adam', loss='mse'); lstm.fit(xs, ys, epochs=2, batch_size=32, verbose=0)
        MODEL_DIR.mkdir(parents=True, exist_ok=True); lstm.save(MODEL_DIR/'lstm_deposit_regressor.keras'); joblib.dump(scaler, MODEL_DIR/'lstm_deposit_regressor_scaler.pkl')
        if len(xst): rows.append({'Model': 'lstm', **regression_metrics(yst, lstm.predict(xst, verbose=0).ravel())})
    except Exception as exc:
        rows.append({'Model': 'lstm', 'Error': str(exc)})
    return rows
if __name__ == '__main__': print(main())

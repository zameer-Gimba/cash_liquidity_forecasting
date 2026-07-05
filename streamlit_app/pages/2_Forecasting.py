from __future__ import annotations
import joblib, numpy as np, streamlit as st
from src.models.predict_deposit import predict_deposit
from src.utils.config import FEATURE_COLS, LSTM_SEQUENCE_LENGTH, MODEL_DIR, SAFETY_BUFFER_DEPOSIT, SAFETY_BUFFER_WITHDRAWAL
st.title('Forecasting')
df=st.session_state.get('clean_dataset')
if df is None: st.warning('Load a dataset from the main page.'); st.stop()
st.info(f"Dataset loaded: {st.session_state.get('dataset_source','canonical')} | {st.session_state.get('dataset_type','FEATURE_ENGINEERED')}")
features=df[FEATURE_COLS].tail(1)
def load(name):
    p=MODEL_DIR/name; return joblib.load(p) if p.exists() else None

def predict_withdrawal(model_name: str) -> float:
    p = MODEL_DIR / model_name
    if not p.exists():
        raise FileNotFoundError(f'Missing artefact: {model_name}')
    if model_name.endswith('.keras'):
        from tensorflow import keras
        scaler_art = joblib.load(MODEL_DIR / 'lstm_withdrawal_scaler.pkl')
        scaler, y_scale = scaler_art['scaler'], scaler_art['y_scale']
        seq = df[FEATURE_COLS].tail(LSTM_SEQUENCE_LENGTH)
        if len(seq) < LSTM_SEQUENCE_LENGTH:
            raise ValueError('Not enough history for the LSTM sequence window.')
        X = scaler.transform(seq.values)[None, :, :]
        model = keras.models.load_model(str(p))
        return float(model.predict(X, verbose=0).ravel()[0]) * y_scale
    model = joblib.load(p)
    if model_name in {'arima_withdrawal.pkl', 'sarima_withdrawal.pkl'}:
        if hasattr(model, 'get_forecast'):
            return float(np.asarray(model.get_forecast(steps=1).predicted_mean).ravel()[0])
        return float(np.asarray(model.forecast(steps=1)).ravel()[0])
    return float(model.predict(features)[0])

model_name=st.selectbox('Withdrawal model', ['rf_withdrawal.pkl','xgb_withdrawal.pkl','lgbm_withdrawal.pkl','arima_withdrawal.pkl','sarima_withdrawal.pkl','lstm_withdrawal.keras'])
try:
    pred=predict_withdrawal(model_name)
except Exception as exc:
    st.warning(f'Model schema mismatch or missing; using last-30-day baseline. {exc}'); pred=float(df['Withdrawal_Amount'].tail(30).mean())
st.metric('Predicted Withdrawal Demand (Next Day)', f'₦{pred:,.0f}'); st.metric('Recommended Cash Reserve', f'₦{pred*(1+SAFETY_BUFFER_WITHDRAWAL):,.0f}')
st.subheader('Deposit Forecast')
try:
    clf=load('rf_deposit_classifier.pkl'); reg=load('rf_deposit_regressor.pkl'); dep=predict_deposit(features, clf, reg) if clf and reg else 0.0
except Exception as exc:
    st.warning(f'Deposit model fallback to baseline. {exc}')
    _recent_deposits = df.loc[df['Deposit_Amount']>0,'Deposit_Amount'].tail(30)
    dep = float(_recent_deposits.mean()) if not _recent_deposits.empty else 0.0
st.metric('Predicted Deposit', f'₦{dep:,.0f}' if dep else 'No deposit expected tomorrow'); st.metric('Recommended Float Top-Up', f'₦{dep*(1+SAFETY_BUFFER_DEPOSIT):,.0f}')

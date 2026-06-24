from __future__ import annotations
import joblib, streamlit as st
from src.models.predict_deposit import predict_deposit
from src.utils.config import FEATURE_COLS, MODEL_DIR, SAFETY_BUFFER_DEPOSIT, SAFETY_BUFFER_WITHDRAWAL
st.title('Forecasting')
df=st.session_state.get('clean_dataset')
if df is None: st.warning('Load a dataset from the main page.'); st.stop()
st.info(f"Dataset loaded: {st.session_state.get('dataset_source','canonical')} | {st.session_state.get('dataset_type','FEATURE_ENGINEERED')}")
features=df[FEATURE_COLS].tail(1)
def load(name):
    p=MODEL_DIR/name; return joblib.load(p) if p.exists() else None
model_name=st.selectbox('Withdrawal model', ['rf_withdrawal.pkl','xgb_withdrawal.pkl','lgbm_withdrawal.pkl','arima_withdrawal.pkl','sarima_withdrawal.pkl','lstm_withdrawal.keras'])
try:
    m=load(model_name); pred=float(m.predict(features)[0]) if m and hasattr(m,'predict') else float(df['Withdrawal_Amount'].tail(30).mean())
except Exception as exc:
    st.warning(f'Model schema mismatch or missing; using last-30-day baseline. {exc}'); pred=float(df['Withdrawal_Amount'].tail(30).mean())
st.metric('Predicted Withdrawal Demand (Next Day)', f'₦{pred:,.0f}'); st.metric('Recommended Cash Reserve', f'₦{pred*(1+SAFETY_BUFFER_WITHDRAWAL):,.0f}')
st.subheader('Deposit Forecast')
try:
    clf=load('rf_deposit_classifier.pkl'); reg=load('rf_deposit_regressor.pkl'); dep=predict_deposit(features, clf, reg) if clf and reg else 0.0
except Exception as exc:
    st.warning(f'Deposit model fallback to baseline. {exc}'); dep=float(df.loc[df['Deposit_Amount']>0,'Deposit_Amount'].tail(30).mean() or 0.0)
st.metric('Predicted Deposit', f'₦{dep:,.0f}' if dep else 'No deposit expected tomorrow'); st.metric('Recommended Float Top-Up', f'₦{dep*(1+SAFETY_BUFFER_DEPOSIT):,.0f}')

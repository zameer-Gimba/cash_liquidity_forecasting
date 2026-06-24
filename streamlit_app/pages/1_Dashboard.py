from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import streamlit as st
from src.models.predict_deposit import predict_deposit
from src.utils.config import FEATURE_COLS, MODEL_DIR, SAFETY_BUFFER_DEPOSIT, SAFETY_BUFFER_WITHDRAWAL

st.title('Dashboard')
df = st.session_state.get('clean_dataset')
if df is None:
    st.warning('Load a dataset from the main page.'); st.stop()
features = df[FEATURE_COLS].tail(1)
def _baseline(col): return float(df[col].tail(30).mean())
def _load(name):
    try:
        import joblib
        p = MODEL_DIR / name
        return joblib.load(p) if p.exists() else None
    except Exception as exc:
        st.caption(f'Model loader unavailable for {name}: {exc}')
        return None
try:
    rf=_load('rf_withdrawal.pkl'); pred_w=float(rf.predict(features)[0]) if rf else _baseline('Withdrawal_Amount')
except Exception as exc:
    st.warning(f'Withdrawal model incompatible; using historical baseline. {exc}'); pred_w=_baseline('Withdrawal_Amount')
try:
    clf=_load('rf_deposit_classifier.pkl'); reg=_load('rf_deposit_regressor.pkl')
    pred_d=predict_deposit(features, clf, reg) if clf and reg else 0.0
except Exception as exc:
    st.warning(f'Deposit models unavailable/incompatible; using baseline. {exc}'); pred_d=float(df.loc[df['Deposit_Amount']>0,'Deposit_Amount'].tail(30).mean() or 0.0)
q33,q66=df['Withdrawal_Amount'].quantile([.33,.66]); risk='LOW' if pred_w < q33 else 'MEDIUM' if pred_w <= q66 else 'HIGH'
c=st.columns(4); c[0].metric('Predicted Withdrawal Demand (Next Day)', f'₦{pred_w:,.0f}'); c[1].metric('Recommended Cash Reserve', f'₦{pred_w*(1+SAFETY_BUFFER_WITHDRAWAL):,.0f}'); c[2].metric('Liquidity Risk', risk); c[3].metric('Recommended Float Top-Up', f'₦{pred_d*(1+SAFETY_BUFFER_DEPOSIT):,.0f}' if pred_d else 'No deposit expected tomorrow')
st.line_chart(df.set_index('TransactionDate')[['Withdrawal_Amount','Deposit_Amount']])

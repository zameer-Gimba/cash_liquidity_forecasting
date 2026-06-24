"""Streamlit entrypoint with canonical dataset loading and upload processing."""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.pipeline import process_dataset, validate_schema
from src.utils.config import CANONICAL_DATASET_PATH, FEATURE_COLS, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT, TARGET_WITHDRAWAL
from src.utils.dataset_detector import attempt_auto_repair, detect_dataset_type

st.set_page_config(page_title="Agent Liquidity Prediction", page_icon="💧", layout="wide")
for k, v in {'clean_dataset': None, 'dataset_type': None, 'repair_log': [], 'dataset_validated': False, 'dataset_source': 'canonical'}.items():
    st.session_state.setdefault(k, v)
if st.session_state['clean_dataset'] is None:
    df = pd.read_csv(PROJECT_ROOT / CANONICAL_DATASET_PATH)
    st.session_state.update(clean_dataset=df, dataset_type='FEATURE_ENGINEERED', dataset_validated=True, dataset_source='canonical')

st.title("Predictive Liquidity Model for Agent Banking in Nigeria")
st.caption("Decision support for POS cash reserve and float planning")
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload bank statement or engineered CSV", type=['csv','xlsx'])
    if uploaded is not None:
        try:
            raw = pd.read_excel(uploaded) if uploaded.name.lower().endswith(('xlsx','xls')) else pd.read_csv(uploaded)
            kind = detect_dataset_type(raw); st.session_state['dataset_type'] = kind; st.info(f"Detected: {kind}")
            if kind == 'RAW_STATEMENT':
                st.write('Raw transaction statement detected. Preparing dataset...'); clean = process_dataset(raw); log=[]
            elif kind == 'FEATURE_ENGINEERED':
                clean = validate_schema(raw); log=[]; st.success('Dataset ready.')
            elif kind == 'PARTIALLY_ENGINEERED':
                repaired, log = attempt_auto_repair(raw); clean = validate_schema(repaired); st.write(log)
            else:
                raise ValueError('Unrecognised file format. Please upload a Moniepoint bank statement (.xlsx) or a fully engineered liquidity_dataset.csv.')
            st.session_state.update(clean_dataset=clean, repair_log=log, dataset_validated=True, dataset_source='uploaded')
            st.success(f"Dataset successfully processed and validated. Rows: {len(clean)} | Features: {len(FEATURE_COLS)} | Targets: 3 | Status: ✓ READY")
        except Exception as exc:
            st.session_state['dataset_validated'] = False; st.error(str(exc))
    df = st.session_state['clean_dataset']
    st.download_button('Download Processed Dataset', df.to_csv(index=False).encode(), 'processed_liquidity_dataset.csv', 'text/csv')

_df = st.session_state['clean_dataset']
last30 = _df.tail(30)
pred_w = float(last30['Withdrawal_Amount'].mean())
pred_d = float(last30.loc[last30['Deposit_Amount'] > 0, 'Deposit_Amount'].mean() or 0.0)
st.session_state.setdefault('model_prediction', pred_w)
cols = st.columns(4)
cols[0].metric('Predicted Withdrawal Demand (Next Day)', f'₦{pred_w:,.0f}')
cols[1].metric('Recommended Cash Reserve', f'₦{pred_w*1.15:,.0f}')
cols[2].metric('Predicted Deposit (Next Day)', f'₦{pred_d:,.0f}' if pred_d else 'No deposit expected tomorrow')
cols[3].metric('Dataset Source', st.session_state['dataset_source'])
st.info('Use the sidebar pages for Dashboard, Forecasting, Analytics, and Model Comparison.')

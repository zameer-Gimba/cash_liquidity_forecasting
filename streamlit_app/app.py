"""Streamlit entrypoint with canonical dataset loading and upload processing."""
from __future__ import annotations

# 1. MUST BE FIRST: Setup paths before any other imports happen
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party packages
import pandas as pd
import streamlit as st

# 3. Local repository imports
from src.preprocessing.pipeline import process_dataset, validate_schema
from src.utils.config import CANONICAL_DATASET_PATH, FEATURE_COLS
from src.utils.dataset_detector import attempt_auto_repair, detect_dataset_type

st.set_page_config(page_title="Agent Liquidity Prediction", page_icon="💧", layout="wide")

for k, v in {
    'clean_dataset': None, 'dataset_type': None, 'repair_log': [], 'dataset_validated': False, 'dataset_source': 'canonical',
    'trained_model': None, 'trained_model_key': None, 'trained_model_extra': None, 'trained_model_rows': None,
    'current_prediction': None,
}.items():
    st.session_state.setdefault(k, v)

if st.session_state['clean_dataset'] is None:
    # FIX: CANONICAL_DATASET_PATH is already an absolute path from config.py
    df = pd.read_csv(CANONICAL_DATASET_PATH)
    st.session_state.update(clean_dataset=df, dataset_type='FEATURE_ENGINEERED', dataset_validated=True, dataset_source='canonical')

st.title("Predictive Liquidity Model for Agent Banking in Nigeria")
st.caption("Decision support for POS cash reserve and float planning")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload bank statement or engineered CSV", type=['csv', 'xlsx'])
    if uploaded is not None:
        try:
            raw = pd.read_excel(uploaded) if uploaded.name.lower().endswith(('xlsx', 'xls')) else pd.read_csv(uploaded)
            kind = detect_dataset_type(raw)
            st.info(f"Detected: {kind}")
            if st.button('Save dataset'):
                if kind == 'RAW_STATEMENT':
                    st.write('Raw transaction statement detected. Preparing dataset...')
                    clean = process_dataset(raw); log = []
                elif kind == 'FEATURE_ENGINEERED':
                    clean = validate_schema(raw); log = []
                elif kind == 'PARTIALLY_ENGINEERED':
                    repaired, log = attempt_auto_repair(raw); clean = validate_schema(repaired)
                else:
                    raise ValueError('Unrecognised file format. Please upload a Moniepoint bank statement (.xlsx) or a fully engineered liquidity_dataset.csv.')
                # A new dataset invalidates any model/prediction trained on the previous one.
                st.session_state.update(
                    clean_dataset=clean, dataset_type=kind, repair_log=log, dataset_validated=True, dataset_source='uploaded',
                    trained_model=None, trained_model_key=None, trained_model_extra=None, trained_model_rows=None, current_prediction=None,
                )
                st.success(f"Dataset successfully processed and validated. Rows: {len(clean)} | Features: {len(FEATURE_COLS)} | Targets: 3 | Status: ✓ READY")
        except Exception as exc:
            st.error(str(exc))
    st.caption(f"Active dataset: {st.session_state['dataset_source']} ({len(st.session_state['clean_dataset'])} rows)")
    df = st.session_state['clean_dataset']
    st.download_button('Download Processed Dataset', df.to_csv(index=False).encode(), 'processed_liquidity_dataset.csv', 'text/csv')

st.info('Use the sidebar pages for Dashboard, Forecasting, Analytics, and Model Comparison.')

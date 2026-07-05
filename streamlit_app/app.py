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
from src.utils.config import CANONICAL_DATASET_PATH, CANONICAL_TRAINING_END_DATE, FEATURE_COLS
from src.utils.dataset_detector import attempt_auto_repair, detect_dataset_type

st.set_page_config(page_title="Agent Liquidity Prediction", page_icon="💧", layout="wide")

for k, v in {
    'clean_dataset': None, 'dataset_type': None, 'repair_log': [], 'dataset_validated': False, 'dataset_source': 'canonical',
    'trained_model': None, 'trained_model_key': None, 'trained_model_extra': None, 'trained_model_rows': None,
    'current_prediction': None,
}.items():
    st.session_state.setdefault(k, v)


def _load_canonical() -> pd.DataFrame:
    # FIX: CANONICAL_DATASET_PATH is already an absolute path from config.py
    df = pd.read_csv(CANONICAL_DATASET_PATH)
    # The canonical file's trailing window (from 2026-01-17) is a sparse, gapped data-collection
    # artifact rather than real zero-business days (see config.CANONICAL_TRAINING_END_DATE) - excluded
    # here so models and date-aware predictions aren't anchored to it. The CSV on disk is untouched.
    return df[df['TransactionDate'] <= CANONICAL_TRAINING_END_DATE].reset_index(drop=True)


if st.session_state['clean_dataset'] is None:
    st.session_state.update(clean_dataset=_load_canonical(), dataset_type='FEATURE_ENGINEERED', dataset_validated=True, dataset_source='canonical')

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
            save_col, clear_col = st.columns(2)
            if save_col.button('Save dataset', width='stretch'):
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
            if clear_col.button('Clear dataset', width='stretch', disabled=st.session_state['dataset_source'] != 'uploaded'):
                st.session_state.update(
                    clean_dataset=_load_canonical(), dataset_type='FEATURE_ENGINEERED', dataset_validated=True, dataset_source='canonical',
                    trained_model=None, trained_model_key=None, trained_model_extra=None, trained_model_rows=None, current_prediction=None,
                )
                st.success('Cleared. Reverted to the canonical dataset.')
        except Exception as exc:
            st.error(str(exc))
    elif st.session_state['dataset_source'] == 'uploaded':
        # No file currently selected in the uploader, but a previously saved upload is still active.
        if st.button('Clear dataset', width='stretch'):
            st.session_state.update(
                clean_dataset=_load_canonical(), dataset_type='FEATURE_ENGINEERED', dataset_validated=True, dataset_source='canonical',
                trained_model=None, trained_model_key=None, trained_model_extra=None, trained_model_rows=None, current_prediction=None,
            )
            st.success('Cleared. Reverted to the canonical dataset.')
    st.caption(f"Active dataset: {st.session_state['dataset_source']} ({len(st.session_state['clean_dataset'])} rows)")
    df = st.session_state['clean_dataset']
    st.download_button('Download Processed Dataset', df.to_csv(index=False).encode(), 'processed_liquidity_dataset.csv', 'text/csv')

st.info('Use the sidebar pages for Dashboard, Forecasting, Analytics, and Model Comparison.')

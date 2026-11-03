from __future__ import annotations
from pathlib import Path
import pandas as pd, streamlit as st
from src.utils.config import FEATURE_COLS
st.title('Model Comparison')
t1,t2=st.tabs(['Withdrawal Models','Deposit Models'])
with t1:
    st.caption('Rows starting with "Baseline" are simple rules every model must beat. Test period: the last 15% of days (Jun 2025 – Jan 2026). Lower MAE is better.')
    p=Path('reports/results/withdrawal_model_comparison.csv')
    if p.exists(): st.dataframe(pd.read_csv(p), use_container_width=True)
    else: st.warning('Run python -m src.models.evaluate_models to generate withdrawal comparisons.')
    st.write(f'Feature importance is based on the canonical {len(FEATURE_COLS)}-feature schema.')
with t2:
    st.info('Deposit forecasting uses a two-stage architecture: classifiers predict whether a deposit occurs, then regressors estimate the amount for days with a deposit. Since 2024 this agent records deposits on almost every active day, so Stage 1 is near-trivial.')
    for title, file in [('Stage 1 classifier comparison','reports/results/deposit_classifier_comparison.csv'),('Stage 2 regressor comparison','reports/results/deposit_regressor_comparison.csv')]:
        st.subheader(title); p=Path(file)
        if p.exists(): st.dataframe(pd.read_csv(p), use_container_width=True)
        else: st.warning(f'Missing {file}. Run model evaluation after training.')

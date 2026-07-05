from __future__ import annotations
import datetime as dt

import joblib
import streamlit as st

from src.models.predict_deposit import predict_deposit
from src.models.session_forecast import MODEL_OPTIONS, build_future_feature_row, predict_withdrawal_for_date, train_withdrawal_model
from src.utils.config import MODEL_DIR, SAFETY_BUFFER_DEPOSIT, SAFETY_BUFFER_WITHDRAWAL

st.title('Forecasting')
df = st.session_state.get('clean_dataset')
if df is None:
    st.warning('Load a dataset from the main page.'); st.stop()
st.info(f"Dataset loaded: {st.session_state.get('dataset_source', 'canonical')} | {st.session_state.get('dataset_type', 'FEATURE_ENGINEERED')}")

tomorrow = dt.date.today() + dt.timedelta(days=1)
col1, col2 = st.columns(2)
target_date = col1.date_input('Prediction date', min_value=tomorrow, max_value=tomorrow + dt.timedelta(days=365), value=tomorrow)
model_label = col2.selectbox('Model', list(MODEL_OPTIONS.keys()))
model_key = MODEL_OPTIONS[model_label]

b_train, b_predict, b_clear = st.columns(3)
train_clicked = b_train.button('Train Model', width='stretch')
predict_clicked = b_predict.button('Generate Prediction', width='stretch')
clear_clicked = b_clear.button('Clear Trained Model', width='stretch')

if train_clicked:
    with st.spinner(f'Training {model_label} on {len(df)} rows...'):
        try:
            model, extra = train_withdrawal_model(model_key, df)
            st.session_state.update(trained_model=model, trained_model_key=model_key, trained_model_extra=extra, trained_model_rows=len(df))
            st.success(f'{model_label} trained on {len(df)} rows and is now active for this session.')
        except Exception as exc:
            st.error(f'Training failed: {exc}')

if clear_clicked:
    st.session_state.update(trained_model=None, trained_model_key=None, trained_model_extra=None, trained_model_rows=None, current_prediction=None)
    st.success('Trained model and prediction cleared.')

active_key = st.session_state.get('trained_model_key')
if active_key:
    active_label = next(label for label, key in MODEL_OPTIONS.items() if key == active_key)
    st.caption(f'Active model: {active_label} (trained on {st.session_state["trained_model_rows"]} rows) — persists across pages until cleared.')
else:
    st.caption('No model trained yet. Select a model above and click Train Model.')

if predict_clicked:
    if st.session_state.get('trained_model') is None:
        st.warning('Train a model first, then generate a prediction.')
    else:
        try:
            active_key = st.session_state['trained_model_key']
            active_label = next(label for label, key in MODEL_OPTIONS.items() if key == active_key)
            pred_w = predict_withdrawal_for_date(active_key, st.session_state['trained_model'], st.session_state['trained_model_extra'], df, target_date)

            future_row = build_future_feature_row(df, target_date)
            try:
                clf = joblib.load(MODEL_DIR / 'rf_deposit_classifier.pkl')
                reg = joblib.load(MODEL_DIR / 'rf_deposit_regressor.pkl')
                pred_d = predict_deposit(future_row, clf, reg)
            except Exception:
                pred_d = 0.0

            q33, q66 = df['Withdrawal_Amount'].quantile([.33, .66])
            risk_level = 'LOW' if pred_w < q33 else 'MEDIUM' if pred_w <= q66 else 'HIGH'

            st.session_state['current_prediction'] = {
                'date': target_date,
                'model_label': active_label,
                'predicted_withdrawal': pred_w,
                'recommended_reserve': pred_w * (1 + SAFETY_BUFFER_WITHDRAWAL),
                'risk_level': risk_level,
                'predicted_deposit': pred_d,
                'recommended_float_topup': pred_d * (1 + SAFETY_BUFFER_DEPOSIT) if pred_d else 0.0,
            }
        except Exception as exc:
            st.error(f'Prediction failed: {exc}')

pred = st.session_state.get('current_prediction')
if pred:
    st.subheader(f"Prediction for {pred['date']} — {pred['model_label']}")
    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted Withdrawal Demand', f"₦{pred['predicted_withdrawal']:,.0f}")
    c2.metric('Recommended Cash Reserve', f"₦{pred['recommended_reserve']:,.0f}")
    c3.metric('Liquidity Risk', pred['risk_level'])
    st.subheader('Deposit Forecast')
    c3, c4 = st.columns(2)
    c3.metric('Predicted Deposit', f"₦{pred['predicted_deposit']:,.0f}" if pred['predicted_deposit'] else 'No deposit expected')
    c4.metric('Recommended Float Top-Up', f"₦{pred['recommended_float_topup']:,.0f}" if pred['predicted_deposit'] else '₦0')

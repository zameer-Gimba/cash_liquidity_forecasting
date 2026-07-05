from __future__ import annotations
import streamlit as st

st.title('Dashboard')
df = st.session_state.get('clean_dataset')
if df is None:
    st.warning('Load a dataset from the main page.'); st.stop()

pred = st.session_state.get('current_prediction')
if pred is None:
    st.info(
        'No prediction yet. Go to the **Forecasting** page, pick a date and a model, '
        'click **Train Model**, then **Generate Prediction**. The result will appear here '
        'until you clear it.'
    )
else:
    st.caption(f"Prediction for {pred['date']} using {pred['model_label']} (from the Forecasting page)")
    c = st.columns(5)
    c[0].metric('Predicted Withdrawal Demand', f"₦{pred['predicted_withdrawal']:,.0f}")
    c[1].metric('Recommended Cash Reserve', f"₦{pred['recommended_reserve']:,.0f}")
    c[2].metric('Liquidity Risk', pred['risk_level'])
    c[3].metric('Predicted Deposit', f"₦{pred['predicted_deposit']:,.0f}" if pred['predicted_deposit'] else 'No deposit expected')
    c[4].metric('Recommended Float Top-Up', f"₦{pred['recommended_float_topup']:,.0f}" if pred['predicted_deposit'] else '₦0')
    st.line_chart(df.set_index('TransactionDate')[['Withdrawal_Amount', 'Deposit_Amount']])

"""Forecasting page."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.models.predict import MODEL_ARTIFACTS, available_model_artifacts, load_feature_history, predict_for_date
from src.utils.config import SAFETY_BUFFER


@st.cache_data(show_spinner=False)
def _load_history_from_path(path: str | None) -> pd.DataFrame:
    return load_feature_history(path)


st.title("Forecasting")
st.caption("Generate date-aware withdrawal demand forecasts from saved models or a safe historical baseline.")

uploaded = st.file_uploader("Optional: upload feature-engineered CSV", type=["csv"])
try:
    if uploaded is not None:
        history = pd.read_csv(uploaded, parse_dates=["TransactionDate"]).sort_values("TransactionDate")
    else:
        history = _load_history_from_path(None)
except Exception as exc:
    st.error(str(exc))
    st.info("Run preprocessing first or upload a feature-engineered CSV containing TransactionDate and Target_Cash_Demand_Next_Day.")
    st.stop()

latest_date = pd.Timestamp(history["TransactionDate"].max()).date()
default_forecast_date = max(date.today() + timedelta(days=1), latest_date + timedelta(days=1))
future_date = st.date_input("Select forecast date", value=default_forecast_date, min_value=latest_date + timedelta(days=1))
buffer = st.slider("Safety buffer", min_value=0.0, max_value=0.5, value=SAFETY_BUFFER, step=0.01)

artifacts = available_model_artifacts()
model_options = ["Best Available", *MODEL_ARTIFACTS.keys(), "Historical Baseline"]
model_name = st.selectbox("Model", options=model_options, index=0)

with st.expander("Available saved model artifacts", expanded=False):
    if artifacts:
        st.write({name: str(path) for name, path in artifacts.items()})
    else:
        st.warning("No saved model artifacts found. Forecasting will use the historical baseline.")

if st.button("Generate prediction", type="primary"):
    result = predict_for_date(model_name, future_date, history, safety_buffer=buffer)
    st.session_state["predicted_tomorrow_demand"] = result.predicted_withdrawal_demand
    st.session_state["model_used"] = result.model_name
    st.session_state["confidence_score"] = result.confidence_score

    cols = st.columns(4)
    cols[0].metric("Forecast Date", result.forecast_date.date().isoformat())
    cols[1].metric("Predicted Withdrawal Demand", f"₦{result.predicted_withdrawal_demand:,.0f}")
    cols[2].metric("Recommended Cash Reserve", f"₦{result.recommended_cash_reserve:,.0f}")
    cols[3].metric("Risk Level", result.risk_level)
    st.write(f"Safety buffer: **{result.safety_buffer:.0%}**")
    st.write(f"Model used: **{result.model_name}**")
    if result.source != "trained_artifact":
        st.warning("A trained artifact was unavailable or incompatible for this selection, so the app used the historical baseline instead of failing.")

st.subheader("Recent history")
st.dataframe(history.tail(10), use_container_width=True)

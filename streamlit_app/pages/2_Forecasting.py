"""Forecasting page."""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.models.predict import recommended_cash_reserve
from src.utils.config import SAFETY_BUFFER

st.title("Forecasting")
future_date = st.date_input("Select future date", value=date.today() + timedelta(days=1), min_value=date.today())
predicted = st.number_input("Predicted withdrawal demand (₦)", min_value=0.0, value=float(st.session_state.get("predicted_tomorrow_demand", 0.0)), step=1000.0)
buffer = st.slider("Safety buffer", min_value=0.0, max_value=0.5, value=SAFETY_BUFFER, step=0.01)
if st.button("Generate prediction"):
    reserve = recommended_cash_reserve(predicted, buffer)
    st.metric("Forecast date", future_date.isoformat())
    st.metric("Predicted Withdrawal Demand", f"₦{predicted:,.0f}")
    st.metric("Safety Buffer", f"{buffer:.0%}")
    st.metric("Recommended Cash Reserve", f"₦{reserve:,.0f}")

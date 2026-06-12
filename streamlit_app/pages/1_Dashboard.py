"""Streamlit entrypoint for the liquidity decision support dashboard."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import recommended_cash_reserve
from src.utils.config import SAFETY_BUFFER

st.set_page_config(page_title="Agent Liquidity Prediction", page_icon="💧", layout="wide")
st.title("Predictive Liquidity Model for Agent Banking in Nigeria")
st.caption("Decision support for POS cash reserve planning")

predicted = st.session_state.get("model_prediction", st.session_state.get("predicted_tomorrow_demand", 0.0))
model_used = st.session_state.get("model_used", st.session_state.get("selected_model", "Best saved model"))
reserve = recommended_cash_reserve(float(predicted), SAFETY_BUFFER)
risk = "High" if predicted > 0 and reserve > predicted * 1.1 else "Low"

cols = st.columns(6)
cols[0].metric("Current Date", date.today().isoformat())
cols[1].metric("Predicted Tomorrow Demand", f"₦{predicted:,.0f}")
cols[2].metric("Recommended Cash Reserve", f"₦{reserve:,.0f}")
cols[3].metric("Risk Level", risk)
cols[4].metric("Model Used", model_used)
cols[5].metric("Confidence Score", st.session_state.get("confidence_score", "N/A"))

st.info("Use the sidebar pages for forecasting, analytics, and model comparison. Train models first to populate live predictions and metrics.")

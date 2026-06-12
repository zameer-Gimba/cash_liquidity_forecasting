"""Analytics page for historical liquidity trends."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Analytics")
uploaded = st.file_uploader("Upload feature-engineered CSV", type=["csv"])
# prefer the project's feature_engineered_dataset file if present
default_path = Path("data/feature_engineered_dataset/feature_engineered_dataset.csv")
alt_path = Path("data/feature_engineered_dataset/features.csv")
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["TransactionDate"])
elif default_path.exists():
    df = pd.read_csv(default_path, parse_dates=["TransactionDate"])
elif alt_path.exists():
    df = pd.read_csv(alt_path, parse_dates=["TransactionDate"])
else:
    st.warning("No feature dataset found. Run preprocessing or upload a CSV.")
    st.stop()

st.line_chart(df.set_index("TransactionDate")[["Total_Debit", "Total_Credit"]])
st.subheader("Monthly trends")
st.bar_chart(df.groupby("Month")[["Total_Debit", "Total_Credit", "Transaction_Count"]].mean())
st.subheader("Rolling averages")
st.line_chart(df.set_index("TransactionDate")[["Rolling_7_Day_Average", "Rolling_30_Day_Average"]])

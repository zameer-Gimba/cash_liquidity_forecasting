"""Model comparison dashboard page."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Model Comparison")
comparison_path = Path("reports/tables/model_comparison.csv")
if not comparison_path.exists():
    # attempt to generate the table automatically
    try:
        from src.models.evaluate_models import save_comparison_tables

        save_comparison_tables()
    except Exception:
        st.warning("Model comparison table not found. Run src/models/evaluate_models.py after training.")

if comparison_path.exists():
    table = pd.read_csv(comparison_path)
    st.dataframe(table, use_container_width=True)
    if not table.empty:
        st.success(f"Best model: {table.iloc[0]['Model']}")
        st.bar_chart(table.set_index("Model")[["MAE", "RMSE", "MAPE"]])
figure = Path("reports/figures/model_comparison.png")
if figure.exists():
    st.image(str(figure), caption="Visual model comparison")

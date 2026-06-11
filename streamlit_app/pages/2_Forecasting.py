"""Forecasting page."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import recommended_cash_reserve
from src.utils.config import SAFETY_BUFFER
from src.models.predict import LiquidityPredictor
import src.models.train_random_forest as train_rf_module
from src.models.common import load_feature_data
from src.utils.config import MODEL_DIR
import importlib
import src.utils.config as config_module
import src.models.common as common_module

st.title("Forecasting")
future_date = st.date_input("Select future date", value=date.today() + timedelta(days=1), min_value=date.today())
predicted = st.number_input("Predicted withdrawal demand (₦)", min_value=0.0, value=float(st.session_state.get("predicted_tomorrow_demand", 0.0)), step=1000.0)
buffer = st.slider("Safety buffer", min_value=0.0, max_value=0.5, value=SAFETY_BUFFER, step=0.01)

# Training controls
st.markdown("### Train and Use Saved Model")
cols = st.columns([1, 1, 1, 1])
with cols[0]:
    if st.button("Train Random Forest (tuned)"):
        with st.spinner("Training Random Forest (this may take a while)..."):
            data_path = Path("data/feature_engineered_dataset/feature_engineered_dataset.csv")
            if not data_path.exists():
                st.error(f"Feature dataset not found: {data_path}")
            else:
                try:
                    # reload config, common, and training modules to pick up on-disk changes without a server restart
                    importlib.reload(config_module)
                    importlib.reload(common_module)
                    importlib.reload(train_rf_module)
                    metrics = train_rf_module.main(str(data_path), n_iter=10)
                    st.success("Training complete")
                    st.json(metrics)
                    st.write(f"Saved model artifacts to {MODEL_DIR}")
                except KeyError as e:
                    st.error("Training failed: required target column not found in dataset.")
                    st.error(str(e))
                    # concise diagnostics without printing full dataset
                    try:
                        df = load_feature_data(str(data_path))
                        expected = config_module.TARGET_COLUMN
                        available = list(df.columns)
                        if expected not in available:
                            st.warning(f"Expected target column '{expected}' not present. Available columns: {len(available)}")
                            st.info("Ensure your feature dataset contains the configured target column or update src/utils/config.py to match the dataset.")
                    except Exception:
                        st.info("Could not load dataset for diagnostics.")
                except Exception as e:
                    st.error("Training failed — see exception details.")
                    st.exception(e)
with cols[1]:
    if st.button("Run Saved Model Predictions"):
        model_path = MODEL_DIR / "random_forest.joblib"
        if not model_path.exists():
            st.error("Saved model not found. Train a model first.")
        else:
            df = load_feature_data("data/feature_engineered_dataset/feature_engineered_dataset.csv")
            predictor = LiquidityPredictor(model_path)
            preds = predictor.predict(df, history=df)
            # show the most recent prediction and set session state for dashboard
            last_pred = preds.sort_values("TransactionDate").iloc[-1]
            st.session_state["predicted_tomorrow_demand"] = float(last_pred["Predicted_Withdrawal_Demand"])
            st.session_state["model_used"] = model_path.name
            st.session_state["confidence_score"] = "N/A"
            st.success("Predictions generated and dashboard updated.")
            st.dataframe(preds.tail(50), use_container_width=True)
with cols[2]:
    if st.button("Clear Saved Models"):
        for p in MODEL_DIR.glob("*.joblib"):
            try:
                p.unlink()
            except Exception:
                pass
        st.warning("Deleted saved models from MODEL_DIR")

if st.button("Generate prediction"):
    reserve = recommended_cash_reserve(predicted, buffer)
    st.metric("Forecast date", future_date.isoformat())
    st.metric("Predicted Withdrawal Demand", f"₦{predicted:,.0f}")
    st.metric("Safety Buffer", f"{buffer:.0%}")
    st.metric("Recommended Cash Reserve", f"₦{reserve:,.0f}")

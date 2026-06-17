"""Forecasting page."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import recommended_cash_reserve
from src.utils.config import SAFETY_BUFFER
from src.models.predict import LiquidityPredictor
from src.models.common import load_feature_data
from src.utils.config import MODEL_DIR
import importlib
import src.utils.config as config_module
import src.models.common as common_module
import src.models.evaluate_models as evaluate_module
import src.models.predict as predict_module
import joblib
import numpy as np
import math
from pathlib import Path as _Path

st.title("Forecasting")

# Initialize session state for persistent model and data
if "trained_predictor" not in st.session_state:
    st.session_state["trained_predictor"] = None
if "trained_predictor_model_name" not in st.session_state:
    st.session_state["trained_predictor_model_name"] = None
if "historical_data" not in st.session_state:
    st.session_state["historical_data"] = None
if "future_date" not in st.session_state:
    st.session_state["future_date"] = date.today() + timedelta(days=1)
if "predicted_tomorrow_demand" not in st.session_state:
    st.session_state["predicted_tomorrow_demand"] = 0.0
if "model_prediction" not in st.session_state:
    st.session_state["model_prediction"] = st.session_state["predicted_tomorrow_demand"]
if "buffer" not in st.session_state:
    st.session_state["buffer"] = SAFETY_BUFFER
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "Random Forest"
if "confidence_score" not in st.session_state:
    st.session_state["confidence_score"] = "N/A"
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# Display current model status
st.sidebar.markdown("### Model Status")
if st.session_state["trained_predictor"] is not None:
    st.sidebar.success(f"✓ {st.session_state['trained_predictor_model_name']} model loaded in memory")
else:
    st.sidebar.info("No model currently loaded")

future_date = st.date_input(
    "Select future date",
    value=st.session_state["future_date"],
    min_value=date.today(),
)
buffer = st.slider(
    "Safety buffer",
    min_value=0.0,
    max_value=0.5,
    value=st.session_state["buffer"],
    step=0.01,
)

# Training controls
st.markdown("### Train and Use Saved Model")
model_options = {
    "Random Forest": "src.models.train_random_forest",
    "XGBoost": "src.models.train_xgboost",
    "LightGBM": "src.models.train_lightgbm",
    "LSTM": "src.models.train_lstm",
    "ARIMA": "src.models.train_arima",
    "SARIMA": "src.models.train_sarima",
}
selected_model = st.selectbox(
    "Model to train",
    list(model_options.keys()),
    index=list(model_options.keys()).index(st.session_state["selected_model"]),
)
st.session_state["selected_model"] = selected_model
cols = st.columns([1, 1, 1, 1])
with cols[0]:
    if st.button("Train Selected Model"):
        with st.spinner("Training selected model (this may take a while)..."):
            data_path = Path("data/feature_engineered_dataset/feature_engineered_dataset.csv")
            if not data_path.exists():
                st.error(f"Feature dataset not found: {data_path}")
            else:
                try:
                    # reload modules
                    importlib.reload(config_module)
                    importlib.reload(common_module)
                    importlib.reload(predict_module)
                    train_module = importlib.import_module(model_options[selected_model])
                    importlib.reload(train_module)
                    metrics = train_module.main(str(data_path))
                    st.success("Training complete")
                    st.json(metrics)
                    st.write(f"Saved model artifacts to {MODEL_DIR}")
                    
                    # Load the trained model into session state for persistence
                    artifact_map = {
                        "Random Forest": "random_forest.joblib",
                        "XGBoost": "xgboost.joblib",
                        "LightGBM": "lightgbm.joblib",
                        "LSTM": "lstm_fallback_mlp.joblib",
                        "ARIMA": "arima.joblib",
                        "SARIMA": "sarima.joblib",
                    }
                    model_file = artifact_map.get(selected_model)
                    model_path = MODEL_DIR / model_file if model_file else None
                    if selected_model == "LSTM":
                        fallback_path = MODEL_DIR / "lstm.keras"
                        if (model_path is None or not model_path.exists()) and fallback_path.exists():
                            model_path = fallback_path
                    
                    if model_path and model_path.exists():
                        try:
                            # Use the reloaded predict module to get the updated LiquidityPredictor
                            Predictor = predict_module.LiquidityPredictor
                            predictor = Predictor(model_path)
                            st.session_state["trained_predictor"] = predictor
                            st.session_state["trained_predictor_model_name"] = selected_model
                            # Load and store historical data
                            df = load_feature_data(str(data_path))
                            st.session_state["historical_data"] = df
                            st.success(f"✓ Model loaded into memory for persistent use")
                        except Exception as e:
                            st.warning(f"Could not load model into memory: {str(e)}")
                    
                    # update model comparison table
                    try:
                        evaluate_module.save_comparison_tables()
                        st.success("Updated model comparison table")
                    except Exception:
                        st.info("Could not update model comparison table automatically.")
                except KeyError as e:
                    st.error("Training failed: required target column not found in dataset.")
                    st.error(str(e))
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
    if st.button("Load Saved Model to Memory"):
        artifact_map = {
            "Random Forest": "random_forest.joblib",
            "XGBoost": "xgboost.joblib",
            "LightGBM": "lightgbm.joblib",
            "LSTM": "lstm_fallback_mlp.joblib",
            "ARIMA": "arima.joblib",
            "SARIMA": "sarima.joblib",
        }
        model_file = artifact_map.get(selected_model, "random_forest.joblib")
        model_path = MODEL_DIR / model_file
        if selected_model == "LSTM" and not model_path.exists() and (MODEL_DIR / "lstm.keras").exists():
            model_path = MODEL_DIR / "lstm.keras"
        if not model_path.exists():
            st.error("Saved model not found. Train a model first.")
        else:
            try:
                # Reload the predict module to ensure latest LiquidityPredictor
                importlib.reload(predict_module)
                Predictor = predict_module.LiquidityPredictor
                predictor = Predictor(model_path)
                st.session_state["trained_predictor"] = predictor
                st.session_state["trained_predictor_model_name"] = selected_model
                df = load_feature_data("data/feature_engineered_dataset/feature_engineered_dataset.csv")
                st.session_state["historical_data"] = df
                st.success(f"✓ {selected_model} model loaded into memory for persistent use")
                st.info(f"You can now generate predictions for multiple dates without reloading the model")
            except Exception as e:
                st.error("Unable to load the selected model.")
                st.exception(e)
with cols[2]:
    if st.button("Clear Session Memory"):
        st.session_state["trained_predictor"] = None
        st.session_state["trained_predictor_model_name"] = None
        st.session_state["historical_data"] = None
        st.session_state["prediction_history"] = []
        st.warning("Cleared model from session memory")
with cols[3]:
    if st.button("Clear Saved Models"):
        for p in MODEL_DIR.glob("*.joblib"):
            try:
                p.unlink()
            except Exception:
                pass
        for p in MODEL_DIR.glob("*.keras"):
            try:
                p.unlink()
            except Exception:
                pass
        st.warning("Deleted saved models from MODEL_DIR")

st.markdown("---")
st.markdown("### Generate Predictions (Persistent Model)")
st.info(f"Once a model is loaded into memory, you can generate predictions for multiple dates without retraining.")

if st.button("Generate Prediction for Selected Date"):
    if st.session_state["trained_predictor"] is None:
        st.error("No model loaded in memory. Train or load a model first.")
    elif st.session_state["historical_data"] is None:
        st.error("No historical data available. Load a model first.")
    else:
        try:
            predictor = st.session_state["trained_predictor"]
            historical = st.session_state["historical_data"]
            
            # Generate prediction for the selected date
            result = predictor.predict_for_date(pd.Timestamp(future_date), historical)
            
            # Store in session state
            st.session_state["model_prediction"] = result["predicted_value"]
            st.session_state["predicted_tomorrow_demand"] = result["predicted_value"]
            st.session_state["model_used"] = st.session_state["trained_predictor_model_name"]
            st.session_state["confidence_score"] = "N/A"
            st.session_state["future_date"] = future_date
            st.session_state["buffer"] = buffer
            
            # Add to prediction history
            st.session_state["prediction_history"].append({
                "date": future_date,
                "predicted_value": result["predicted_value"],
                "model": st.session_state["trained_predictor_model_name"],
            })
            
            # Display results
            reserve = recommended_cash_reserve(result["predicted_value"], buffer)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Forecast Date", future_date.isoformat())
            with col2:
                st.metric("Predicted Withdrawal Demand", f"₦{result['predicted_value']:,.0f}")
            with col3:
                st.metric("Safety Buffer", f"{buffer:.0%}")
            with col4:
                st.metric("Recommended Cash Reserve", f"₦{reserve:,.0f}")
            
            st.success("✓ Prediction generated successfully from persistent model")
            
        except Exception as e:
            st.error("Unable to generate prediction from the model.")
            st.exception(e)

# Display prediction history
if st.session_state["prediction_history"]:
    st.markdown("### Prediction History")
    history_df = pd.DataFrame(st.session_state["prediction_history"])
    history_df["date"] = pd.to_datetime(history_df["date"]).dt.date
    st.dataframe(history_df, use_container_width=True, hide_index=True)


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
from src.models.common import load_feature_data
from src.utils.config import MODEL_DIR
import importlib
import src.utils.config as config_module
import src.models.common as common_module
import src.models.evaluate_models as evaluate_module
import joblib
import numpy as np
import math
from pathlib import Path as _Path

st.title("Forecasting")
future_date = st.date_input("Select future date", value=date.today() + timedelta(days=1), min_value=date.today())
predicted = st.number_input("Predicted withdrawal demand (₦)", min_value=0.0, value=float(st.session_state.get("predicted_tomorrow_demand", 0.0)), step=1000.0)
buffer = st.slider("Safety buffer", min_value=0.0, max_value=0.5, value=SAFETY_BUFFER, step=0.01)

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
selected_model = st.selectbox("Model to train", list(model_options.keys()), index=0)
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
                    train_module = importlib.import_module(model_options[selected_model])
                    importlib.reload(train_module)
                    metrics = train_module.main(str(data_path))
                    st.success("Training complete")
                    st.json(metrics)
                    st.write(f"Saved model artifacts to {MODEL_DIR}")
                    # update model comparison table
                    try:
                        evaluate_module.save_comparison_tables()
                        st.success("Updated model comparison table")
                    except Exception:
                        st.info("Could not update model comparison table automatically.")
                    # attempt quick prediction to update dashboard
                    try:
                        df = load_feature_data(str(data_path))
                        def try_quick_predict(model_name: str, df: pd.DataFrame) -> float | None:
                            artifact_map = {
                                "Random Forest": "random_forest.joblib",
                                "XGBoost": "xgboost.joblib",
                                "LightGBM": "lightgbm.joblib",
                                "LSTM": "lstm_fallback_mlp.joblib",
                                "ARIMA": "arima.joblib",
                                "SARIMA": "sarima.joblib",
                            }
                            model_file = artifact_map.get(model_name)
                            if model_file is None:
                                return None
                            model_path = _Path(MODEL_DIR) / model_file
                            if not model_path.exists():
                                # special case: keras lstm model
                                if model_name == "LSTM" and (_Path(MODEL_DIR) / "lstm.keras").exists():
                                    model_path = _Path(MODEL_DIR) / "lstm.keras"
                                else:
                                    return None
                            # try joblib load first
                            try:
                                artifact = joblib.load(model_path)
                            except Exception:
                                artifact = None
                            # sklearn-like predictor
                            if artifact is not None and hasattr(artifact, "predict"):
                                try:
                                    features = common_module.feature_columns(df)
                                    preds = artifact.predict(df[features])
                                    return float(preds[-1])
                                except Exception:
                                    return None
                            # dict fallback (arima/sarima)
                            if isinstance(artifact, dict):
                                if artifact.get("strategy") == "last_observation":
                                    return float(artifact.get("value"))
                                if artifact.get("strategy") == "weekly_seasonal_naive":
                                    pattern = artifact.get("pattern", [])
                                    if pattern:
                                        return float(pattern[-1])
                            # keras LSTM model handling
                            if model_name == "LSTM":
                                scaler_file = _Path(MODEL_DIR) / "lstm_scaler.joblib"
                                keras_file = _Path(MODEL_DIR) / "lstm.keras"
                                if scaler_file.exists() and keras_file.exists():
                                    try:
                                        scaler_art = joblib.load(scaler_file)
                                        from tensorflow import keras

                                        keras_model = keras.models.load_model(str(keras_file))
                                        features = scaler_art["features"]
                                        scaler = scaler_art["scaler"]
                                        X = df[features].fillna(0.0).values
                                        Xs = scaler.transform(X)
                                        x_input = Xs[-1].reshape(1, 1, Xs.shape[1])
                                        pred = keras_model.predict(x_input, verbose=0).ravel()[-1]
                                        return float(pred)
                                    except Exception:
                                        return None
                            return None

                        pred_value = try_quick_predict(selected_model, df)
                        if pred_value is not None and not (math.isnan(pred_value)):
                            st.session_state["predicted_tomorrow_demand"] = float(pred_value)
                            st.session_state["model_used"] = f"{selected_model}"
                            st.session_state["confidence_score"] = "N/A"
                            st.success("Dashboard updated with quick prediction from trained model.")
                    except Exception:
                        pass
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
    if st.button("Run Saved Model Predictions"):
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
        if not model_path.exists():
            st.error("Saved model not found. Train a model first.")
        else:
            try:
                df = load_feature_data("data/feature_engineered_dataset/feature_engineered_dataset.csv")
                # attempt to use LiquidityPredictor; some model artifacts may not be compatible
                predictor = LiquidityPredictor(model_path)
                preds = predictor.predict(df, history=df)
                last_pred = preds.sort_values("TransactionDate").iloc[-1]
                st.session_state["predicted_tomorrow_demand"] = float(last_pred["Predicted_Withdrawal_Demand"])
                st.session_state["model_used"] = model_path.name
                st.session_state["confidence_score"] = "N/A"
                st.success("Predictions generated and dashboard updated.")
                st.dataframe(preds.tail(50), use_container_width=True)
            except Exception as e:
                st.error("Saved model loaded but is incompatible with quick prediction UI.")
                st.exception(e)
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

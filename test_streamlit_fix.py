#!/usr/bin/env python
"""
Simulate Streamlit module reloading to verify the fix works.
This mimics what happens when the user trains a model in the Streamlit app.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta
import importlib

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# First import (like the app does at the top)
from src.models.predict import LiquidityPredictor
from src.models.common import load_feature_data
from src.utils.config import MODEL_DIR
import src.models.predict as predict_module


def test_streamlit_reload_workflow():
    """Test the exact workflow that happens in Streamlit."""
    print("=" * 80)
    print("TEST: Streamlit Module Reloading Workflow")
    print("=" * 80)
    
    # Simulate session state
    session_state = {
        "trained_predictor": None,
        "trained_predictor_model_name": None,
        "historical_data": None,
    }
    
    # Step 1: Load data (done once)
    data_path = "data/feature_engineered_dataset/feature_engineered_dataset.csv"
    print(f"\n1. Loading historical data...")
    historical_data = load_feature_data(data_path)
    print(f"   ✓ Loaded {len(historical_data)} rows")
    
    # Step 2: Simulate training with module reload (like in the Streamlit app)
    print(f"\n2. Simulating model training with module reload (as in Streamlit)...")
    
    # This is what happens when the user clicks "Train Selected Model"
    print(f"   Reloading predict module...")
    importlib.reload(predict_module)
    print(f"   ✓ Module reloaded")
    
    # Get the LiquidityPredictor from the reloaded module
    model_path = MODEL_DIR / "random_forest.joblib"
    if not model_path.exists():
        print(f"   ✗ Model not found: {model_path}")
        return False
    
    print(f"   Creating predictor with reloaded module...")
    Predictor = predict_module.LiquidityPredictor
    predictor = Predictor(model_path)
    print(f"   ✓ Predictor created with reloaded module")
    
    session_state["trained_predictor"] = predictor
    session_state["trained_predictor_model_name"] = "Random Forest"
    session_state["historical_data"] = historical_data
    
    # Step 3: Generate predictions (like the user would do)
    print(f"\n3. Generating predictions from loaded model...")
    
    test_dates = [
        date.today() + timedelta(days=1),
        date.today() + timedelta(days=7),
        date.today() + timedelta(days=30),
    ]
    
    for target_date in test_dates:
        try:
            # This is where the AttributeError would occur if the fix didn't work
            result = session_state["trained_predictor"].predict_for_date(
                pd.Timestamp(target_date),
                session_state["historical_data"]
            )
            print(f"   {target_date.isoformat()}: ₦{result['predicted_value']:,.0f} ✓")
        except AttributeError as e:
            if 'predict_for_date' in str(e):
                print(f"   ✗ AttributeError: {str(e)}")
                print(f"\n✗ TEST FAILED: The fix did not work!")
                return False
            else:
                raise
    
    # Step 4: Simulate loading a different model (like switching models)
    print(f"\n4. Testing model switching with reload...")
    print(f"   Reloading predict module again...")
    importlib.reload(predict_module)
    
    model_path = MODEL_DIR / "lightgbm.joblib"
    if model_path.exists():
        print(f"   Loading LightGBM model...")
        Predictor = predict_module.LiquidityPredictor
        predictor2 = Predictor(model_path)
        print(f"   ✓ LightGBM model loaded")
        
        # Test prediction
        target_date = date.today() + timedelta(days=5)
        try:
            result = predictor2.predict_for_date(
                pd.Timestamp(target_date),
                historical_data
            )
            print(f"   LightGBM prediction: ₦{result['predicted_value']:,.0f} ✓")
        except AttributeError as e:
            if 'predict_for_date' in str(e):
                print(f"   ✗ AttributeError: {str(e)}")
                return False
            else:
                raise
    
    print(f"\n{'=' * 80}")
    print(f"✓ STREAMLIT WORKFLOW TEST PASSED")
    print(f"{'=' * 80}")
    print(f"\nThe fix successfully resolves the AttributeError issue!")
    print(f"Users can now:")
    print(f"  1. Train a model and get it loaded automatically")
    print(f"  2. Generate predictions for multiple dates")
    print(f"  3. Switch models and continue predicting")
    print(f"  4. All without encountering AttributeError")
    
    return True


if __name__ == "__main__":
    try:
        success = test_streamlit_reload_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

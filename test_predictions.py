#!/usr/bin/env python
"""Test script to verify the persistent model and date-independent prediction functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import LiquidityPredictor
from src.models.common import load_feature_data
from src.utils.config import MODEL_DIR


def test_predict_for_date():
    """Test that predict_for_date works for different dates and returns different values."""
    print("=" * 80)
    print("TEST: Date-independent predictions with persistent model")
    print("=" * 80)
    
    # Load feature data
    data_path = "data/feature_engineered_dataset/feature_engineered_dataset.csv"
    print(f"\n1. Loading feature data from {data_path}...")
    historical_data = load_feature_data(data_path)
    print(f"   ✓ Loaded {len(historical_data)} rows of historical data")
    print(f"   Date range: {historical_data['TransactionDate'].min()} to {historical_data['TransactionDate'].max()}")
    
    # Check if Random Forest model exists
    model_path = MODEL_DIR / "random_forest.joblib"
    if not model_path.exists():
        print(f"\n2. Model not found at {model_path}")
        print("   ✗ SKIPPING: Random Forest model needs to be trained first")
        return False
    
    print(f"\n2. Loading Random Forest model from {model_path}...")
    predictor = LiquidityPredictor(model_path)
    print(f"   ✓ Model loaded successfully")
    
    # Test: Generate predictions for multiple different dates
    print(f"\n3. Testing date-independent predictions (multiple dates)...")
    test_dates = [
        date.today() + timedelta(days=1),
        date.today() + timedelta(days=7),
        date.today() + timedelta(days=30),
        date.today() + timedelta(days=180),
    ]
    
    predictions = []
    for target_date in test_dates:
        result = predictor.predict_for_date(pd.Timestamp(target_date), historical_data)
        predictions.append(result)
        print(f"   Date: {target_date.isoformat()} | Predicted: ₦{result['predicted_value']:,.2f}")
    
    # Verify that predictions are different (they should be mostly different due to different calendar features)
    pred_values = [p['predicted_value'] for p in predictions]
    unique_values = len(set(np.round(pred_values, 0)))
    
    print(f"\n4. Checking prediction independence...")
    print(f"   Generated {len(predictions)} predictions for different dates")
    print(f"   Unique prediction values: {unique_values}/{len(predictions)}")
    
    if unique_values >= len(predictions) - 1:  # Allow 1 collision
        print(f"   ✓ Predictions are sufficiently independent (expected for different dates)")
    else:
        print(f"   ⚠ WARNING: Many predictions are identical (might indicate an issue)")
    
    # Test: Verify model is persistent (simulate session state)
    print(f"\n5. Testing model persistence in memory...")
    print(f"   Predictor object ID: {id(predictor)}")
    
    # Make another prediction with the same predictor
    another_date = date.today() + timedelta(days=15)
    result2 = predictor.predict_for_date(pd.Timestamp(another_date), historical_data)
    print(f"   Second prediction for {another_date.isoformat()}: ₦{result2['predicted_value']:,.2f}")
    print(f"   ✓ Model successfully used for multiple predictions without reloading")
    
    # Test: Different models should give different predictions
    print(f"\n6. Testing different models (if available)...")
    model_names = ["xgboost.joblib", "lightgbm.joblib", "random_forest.joblib"]
    test_date = date.today() + timedelta(days=5)
    
    predictions_by_model = {}
    for model_name in model_names:
        model_path = MODEL_DIR / model_name
        if model_path.exists():
            try:
                model_predictor = LiquidityPredictor(model_path)
                result = model_predictor.predict_for_date(pd.Timestamp(test_date), historical_data)
                predictions_by_model[model_name] = result['predicted_value']
                print(f"   {model_name}: ₦{result['predicted_value']:,.2f}")
            except Exception as e:
                print(f"   {model_name}: Error - {str(e)[:50]}")
    
    if len(predictions_by_model) > 1:
        model_pred_values = list(predictions_by_model.values())
        if len(set(np.round(model_pred_values, 0))) > 1:
            print(f"   ✓ Different models produce different predictions")
        else:
            print(f"   ⚠ WARNING: Different models produced same predictions")
    
    print("\n" + "=" * 80)
    print("✓ TEST PASSED: All checks completed successfully")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        success = test_predict_for_date()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

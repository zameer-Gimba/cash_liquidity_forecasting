#!/usr/bin/env python
"""Test all available models to ensure predict_for_date works correctly."""

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


def test_all_models():
    """Test predict_for_date method for all available models."""
    print("=" * 80)
    print("TEST: Verify predict_for_date works for all model types")
    print("=" * 80)
    
    # Load feature data
    data_path = "data/feature_engineered_dataset/feature_engineered_dataset.csv"
    historical_data = load_feature_data(data_path)
    print(f"\nLoaded historical data: {len(historical_data)} rows")
    
    # Define all models to test
    models_to_test = [
        ("Random Forest", "random_forest.joblib"),
        ("LightGBM", "lightgbm.joblib"),
        ("LSTM (Keras)", "lstm.keras"),
        ("ARIMA", "arima.joblib"),
        ("SARIMA", "sarima.joblib"),
        ("XGBoost", "xgboost.joblib"),  # This one doesn't exist
    ]
    
    # Test dates
    test_date_tomorrow = date.today() + timedelta(days=1)
    test_date_future = date.today() + timedelta(days=30)
    
    results = []
    
    for model_name, model_file in models_to_test:
        model_path = MODEL_DIR / model_file
        
        print(f"\n{'─' * 80}")
        print(f"Testing: {model_name}")
        print(f"{'─' * 80}")
        
        if not model_path.exists():
            print(f"✗ Model file not found: {model_path}")
            results.append((model_name, "NOT_FOUND", None))
            continue
        
        try:
            # Load model
            print(f"  Loading model...")
            predictor = LiquidityPredictor(model_path)
            print(f"  ✓ Model loaded")
            
            # Check if predict_for_date method exists
            if not hasattr(predictor, 'predict_for_date'):
                print(f"  ✗ ERROR: predict_for_date method not found!")
                results.append((model_name, "METHOD_ERROR", "Missing predict_for_date"))
                continue
            
            # Test prediction for tomorrow
            print(f"  Generating prediction for tomorrow ({test_date_tomorrow.isoformat()})...")
            result_tomorrow = predictor.predict_for_date(
                pd.Timestamp(test_date_tomorrow),
                historical_data
            )
            print(f"  ✓ Tomorrow prediction: ₦{result_tomorrow['predicted_value']:,.0f}")
            
            # Test prediction for future date
            print(f"  Generating prediction for future date ({test_date_future.isoformat()})...")
            result_future = predictor.predict_for_date(
                pd.Timestamp(test_date_future),
                historical_data
            )
            print(f"  ✓ Future prediction: ₦{result_future['predicted_value']:,.0f}")
            
            # Verify predictions have required fields
            required_fields = ['date', 'predicted_value', 'recommended_reserve']
            for field in required_fields:
                if field not in result_tomorrow:
                    raise ValueError(f"Missing field in prediction: {field}")
            
            # Check if predictions are different (should be in most cases)
            if result_tomorrow['predicted_value'] != result_future['predicted_value']:
                independence = "✓ Date-independent"
            else:
                independence = "⚠ Same prediction for different dates"
            
            print(f"  {independence}")
            results.append((model_name, "SUCCESS", f"Tomorrow: ₦{result_tomorrow['predicted_value']:,.0f}, Future: ₦{result_future['predicted_value']:,.0f}"))
            
        except AttributeError as e:
            if 'predict_for_date' in str(e):
                print(f"  ✗ ERROR: {str(e)}")
                results.append((model_name, "ATTRIBUTE_ERROR", str(e)))
            else:
                raise
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:100]}")
            results.append((model_name, "EXCEPTION", f"{type(e).__name__}: {str(e)[:50]}"))
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")
    
    print(f"{'Model':<20} {'Status':<20} {'Details'}")
    print(f"{'-' * 80}")
    
    success_count = 0
    for model_name, status, details in results:
        status_display = status
        if status == "SUCCESS":
            status_display = "✓ SUCCESS"
            success_count += 1
        elif status == "NOT_FOUND":
            status_display = "⚠ NOT_FOUND"
        else:
            status_display = f"✗ {status}"
        
        print(f"{model_name:<20} {status_display:<20} {details or ''}")
    
    print(f"\n{'─' * 80}")
    print(f"✓ Successful: {success_count}/{len([r for r in results if r[1] != 'NOT_FOUND'])}")
    print(f"Total available models: {len([r for r in results if r[1] != 'NOT_FOUND'])}")
    
    # Determine overall result
    available_models = [r for r in results if r[1] != 'NOT_FOUND']
    failures = [r for r in available_models if r[1] != 'SUCCESS']
    
    if failures:
        print(f"\n✗ FAILURES DETECTED:")
        for model_name, status, details in failures:
            print(f"  - {model_name}: {status}")
        return False
    else:
        print(f"\n✓ ALL AVAILABLE MODELS SUPPORT predict_for_date METHOD")
        return True


if __name__ == "__main__":
    try:
        success = test_all_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

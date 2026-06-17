#!/usr/bin/env python
"""Test if predictions vary across different dates for all models."""

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


def test_date_variation_for_model(model_name: str, model_file: str, historical_data: pd.DataFrame) -> dict:
    """Test if a model produces varying predictions for different dates."""
    model_path = MODEL_DIR / model_file
    
    if not model_path.exists():
        return {"model": model_name, "status": "NOT_FOUND", "details": f"Model file not found"}
    
    try:
        predictor = LiquidityPredictor(model_path)
        
        # Generate predictions for many different dates
        predictions = []
        test_dates = []
        
        # Test dates spread across months
        for days_offset in [1, 5, 10, 15, 20, 25, 30, 60, 90, 180]:
            target_date = date.today() + timedelta(days=days_offset)
            test_dates.append(target_date)
            
            result = predictor.predict_for_date(pd.Timestamp(target_date), historical_data)
            predictions.append(result['predicted_value'])
        
        # Analyze variation
        pred_array = np.array(predictions)
        unique_values = len(set(np.round(pred_array, 0)))  # Round to nearest whole number
        
        std_dev = np.std(pred_array)
        coefficient_of_variation = std_dev / np.mean(pred_array) if np.mean(pred_array) != 0 else 0
        
        min_val = np.min(pred_array)
        max_val = np.max(pred_array)
        range_val = max_val - min_val
        
        # Determine if predictions vary
        has_variation = unique_values > 1  # At least 2 different values
        sufficient_variation = coefficient_of_variation > 0.01  # More than 1% variation
        
        return {
            "model": model_name,
            "status": "OK" if has_variation and sufficient_variation else "FIXED" if unique_values == 1 else "LIMITED",
            "unique_values": unique_values,
            "total_tests": len(predictions),
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "mean": np.mean(pred_array),
            "std_dev": std_dev,
            "cv": coefficient_of_variation,
            "samples": list(zip([d.isoformat() for d in test_dates[:3]], predictions[:3])),
        }
    
    except Exception as e:
        return {"model": model_name, "status": "ERROR", "details": f"{type(e).__name__}: {str(e)[:80]}"}


def main():
    print("=" * 100)
    print("TEST: Date Variation Across All Models")
    print("=" * 100)
    
    # Load historical data
    data_path = "data/feature_engineered_dataset/feature_engineered_dataset.csv"
    historical_data = load_feature_data(data_path)
    print(f"\nLoaded historical data: {len(historical_data)} rows\n")
    
    # Define models to test
    models_to_test = [
        ("Random Forest", "random_forest.joblib"),
        ("LightGBM", "lightgbm.joblib"),
        ("LSTM (Keras)", "lstm.keras"),
        ("ARIMA", "arima.joblib"),
        ("SARIMA", "sarima.joblib"),
        ("XGBoost", "xgboost.joblib"),
    ]
    
    results = []
    fixed_models = []
    
    # Test each model
    for model_name, model_file in models_to_test:
        print(f"Testing {model_name}...", end=" ")
        result = test_date_variation_for_model(model_name, model_file, historical_data)
        results.append(result)
        
        if result["status"] == "FIXED":
            fixed_models.append(model_name)
            print("⚠ FIXED PREDICTIONS")
        elif result["status"] == "NOT_FOUND":
            print("⚠ NOT FOUND")
        elif result["status"] == "ERROR":
            print(f"✗ ERROR")
        elif result["status"] == "OK":
            print(f"✓ VARIES")
        else:
            print(f"⚠ LIMITED VARIATION")
    
    # Display results
    print(f"\n{'=' * 100}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 100}\n")
    
    print(f"{'Model':<20} {'Status':<15} {'Unique':<10} {'Min':<15} {'Max':<15} {'Coef.Var':<12}")
    print(f"{'-' * 100}")
    
    for result in results:
        if result["status"] in ["NOT_FOUND", "ERROR"]:
            print(f"{result['model']:<20} {result['status']:<15}")
            if "details" in result:
                print(f"  {result['details']}")
        else:
            cv = result.get("cv", 0)
            print(f"{result['model']:<20} {result['status']:<15} {result.get('unique_values', 'N/A'):<10} "
                  f"₦{result.get('min', 0):>13,.0f} ₦{result.get('max', 0):>13,.0f} {cv:>10.2%}")
    
    # Show details for fixed models
    if fixed_models:
        print(f"\n{'=' * 100}")
        print("MODELS WITH FIXED PREDICTIONS (NEED FIX)")
        print(f"{'=' * 100}\n")
        
        for result in results:
            if result["status"] == "FIXED":
                print(f"\n{result['model']}:")
                print(f"  Unique values: {result.get('unique_values', 'N/A')} out of {result.get('total_tests', 'N/A')} tests")
                print(f"  Sample predictions:")
                for date_str, pred_val in result.get('samples', []):
                    print(f"    {date_str}: ₦{pred_val:,.0f}")
    
    # Final verdict
    print(f"\n{'=' * 100}")
    if fixed_models:
        print(f"✗ ISSUE DETECTED: {len(fixed_models)} model(s) produce fixed predictions")
        print(f"   Models: {', '.join(fixed_models)}")
        return False
    else:
        print(f"✓ ALL MODELS produce varying predictions for different dates")
        return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

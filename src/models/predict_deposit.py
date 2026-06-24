from __future__ import annotations
import pandas as pd

def predict_deposit(features: pd.DataFrame, classifier_model, regressor_model) -> float:
    """Run two-stage deposit inference and return 0.0 when no deposit is predicted."""
    will_deposit = classifier_model.predict(features)[0]
    if int(will_deposit) == 1:
        return max(0.0, float(regressor_model.predict(features)[0]))
    return 0.0

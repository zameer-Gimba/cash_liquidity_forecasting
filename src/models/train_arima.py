from __future__ import annotations
import numpy as np
from src.models.common import chronological_split, load_canonical_data, regression_metrics, save_pickle
from src.utils.config import DATE_COL, TARGET_WITHDRAWAL

def main() -> dict[str, float]:
    from statsmodels.tsa.arima.model import ARIMA
    train, _, test = chronological_split(load_canonical_data())
    train_series = train.set_index(DATE_COL)[TARGET_WITHDRAWAL].asfreq('D').ffill()
    model = ARIMA(train_series, order=(1, 1, 1)).fit()
    save_pickle(model, 'arima_withdrawal.pkl')
    if test.empty:
        return {}
    steps = len(test)
    preds = np.asarray(model.get_forecast(steps=steps).predicted_mean).ravel()
    return regression_metrics(test[TARGET_WITHDRAWAL].to_numpy(), preds)
if __name__ == '__main__': print(main())

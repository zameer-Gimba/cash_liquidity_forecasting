from __future__ import annotations
from src.models.common import load_canonical_data, save_pickle
from src.utils.config import DATE_COL, TARGET_WITHDRAWAL

def main() -> dict[str, float]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    df = load_canonical_data().set_index(DATE_COL)
    series = df[TARGET_WITHDRAWAL].asfreq('D').ffill()
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    save_pickle(model, 'sarima_withdrawal.pkl'); return {'AIC': float(model.aic)}
if __name__ == '__main__': print(main())

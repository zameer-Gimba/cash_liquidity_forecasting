from __future__ import annotations
from src.models._statespace import evaluate


def main() -> dict[str, float]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # Weekly seasonality: POS cash demand follows the day of the week.
    return evaluate(lambda s: SARIMAX(s, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)), 'sarima_withdrawal.pkl', disp=False)


if __name__ == '__main__': print(main())

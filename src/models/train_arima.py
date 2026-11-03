from __future__ import annotations
from src.models._statespace import evaluate


def main() -> dict[str, float]:
    from statsmodels.tsa.arima.model import ARIMA
    return evaluate(lambda s: ARIMA(s, order=(1, 1, 1)), 'arima_withdrawal.pkl')


if __name__ == '__main__': print(main())

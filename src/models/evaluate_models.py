from __future__ import annotations
from src.models import (
    train_arima,
    train_deposit_classifier,
    train_deposit_regressor,
    train_lightgbm,
    train_lstm,
    train_random_forest,
    train_sarima,
    train_xgboost,
)
from src.models.common import save_results
from src.utils.config import TARGET_WITHDRAWAL

WITHDRAWAL_MODELS = [
    ('Random Forest', train_random_forest),
    ('XGBoost', train_xgboost),
    ('LightGBM', train_lightgbm),
    ('ARIMA', train_arima),
    ('SARIMA', train_sarima),
    ('LSTM', train_lstm),
]


def main() -> None:
    rows = []
    for label, module in WITHDRAWAL_MODELS:
        try:
            metrics = module.main()
            rows.append({'Model': label, 'Target': TARGET_WITHDRAWAL, **metrics})
        except Exception as exc:
            rows.append({'Model': label, 'Target': TARGET_WITHDRAWAL, 'Error': str(exc)})
    save_results(rows, 'reports/results/withdrawal_model_comparison.csv')

    try:
        clf_rows = train_deposit_classifier.main()
    except Exception as exc:
        clf_rows = [{'Model': 'ALL', 'Error': str(exc)}]
    save_results(clf_rows, 'reports/results/deposit_classifier_comparison.csv')

    try:
        reg_rows = train_deposit_regressor.main()
    except Exception as exc:
        reg_rows = [{'Model': 'ALL', 'Error': str(exc)}]
    save_results(reg_rows, 'reports/results/deposit_regressor_comparison.csv')

    print('Evaluation reports written to reports/results')


if __name__ == '__main__':
    main()

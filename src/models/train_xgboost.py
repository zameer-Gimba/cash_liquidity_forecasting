from __future__ import annotations
from src.models.common import chronological_split, load_canonical_data, regression_metrics, save_pickle, x_y
from src.utils.config import RANDOM_STATE, TARGET_WITHDRAWAL

def main() -> dict[str, float]:
    from xgboost import XGBRegressor
    train, _, test = chronological_split(load_canonical_data())
    model = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=2)
    model.fit(*x_y(train, TARGET_WITHDRAWAL)); pred = model.predict(x_y(test)[0])
    save_pickle(model, 'xgb_withdrawal.pkl'); return regression_metrics(test[TARGET_WITHDRAWAL], pred)
if __name__ == '__main__': print(main())

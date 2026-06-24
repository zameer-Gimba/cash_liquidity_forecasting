from __future__ import annotations
from sklearn.ensemble import RandomForestRegressor
from src.models.common import chronological_split, load_canonical_data, regression_metrics, save_pickle, x_y
from src.utils.config import RANDOM_STATE, TARGET_WITHDRAWAL

def main() -> dict[str, float]:
    train, _, test = chronological_split(load_canonical_data())
    model = RandomForestRegressor(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10)
    model.fit(*x_y(train, TARGET_WITHDRAWAL)); pred = model.predict(test[x_y(test)[0].columns])
    save_pickle(model, 'rf_withdrawal.pkl'); return regression_metrics(test[TARGET_WITHDRAWAL], pred)
if __name__ == '__main__': print(main())

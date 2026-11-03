from __future__ import annotations
from src.models.common import fit_eval_split, load_canonical_data, regression_metrics, save_pickle, x_y
from src.models.level_scaled import tree_regressor
from src.utils.config import TARGET_WITHDRAWAL


def main() -> dict[str, float]:
    train, test = fit_eval_split(load_canonical_data())
    model = tree_regressor('rf', 'Withdrawal').fit(*x_y(train, TARGET_WITHDRAWAL))
    pred = model.predict(x_y(test)[0])
    save_pickle(model, 'rf_withdrawal.pkl')
    return regression_metrics(test[TARGET_WITHDRAWAL], pred)


if __name__ == '__main__': print(main())

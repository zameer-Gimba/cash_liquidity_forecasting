from __future__ import annotations
import joblib
from pathlib import Path
from src.models.common import chronological_split, classifier_metrics, load_canonical_data, regression_metrics, save_results, x_y
from src.utils.config import MODEL_DIR, ROOT_DIR, TARGET_DEPOSIT, TARGET_HAS_DEPOSIT, TARGET_WITHDRAWAL

def _load(name):
    p = MODEL_DIR / name
    return joblib.load(p) if p.exists() else None

def main() -> None:
    df = load_canonical_data(); _, _, test = chronological_split(df)
    rows=[]
    for file, label in [('rf_withdrawal.pkl','Random Forest'),('xgb_withdrawal.pkl','XGBoost'),('lgbm_withdrawal.pkl','LightGBM')]:
        m=_load(file)
        if m is not None: rows.append({'Model': label, 'Target': TARGET_WITHDRAWAL, **regression_metrics(test[TARGET_WITHDRAWAL], m.predict(x_y(test)[0]))})
    save_results(rows, 'reports/results/withdrawal_model_comparison.csv')
    clf_rows=[]
    Xte, yte = x_y(test, TARGET_HAS_DEPOSIT)
    for file, label in [('rf_deposit_classifier.pkl','RF Classifier'),('xgb_deposit_classifier.pkl','XGB Classifier'),('lgbm_deposit_classifier.pkl','LGBM Classifier')]:
        m=_load(file)
        if m is not None:
            pred=m.predict(Xte); score=m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else pred
            clf_rows.append({'Model': label, **classifier_metrics(yte, pred, score)})
    save_results(clf_rows, 'reports/results/deposit_classifier_comparison.csv')
    dep = df[df['Deposit_Amount'] > 0].copy(); _, _, dep_test = chronological_split(dep)
    reg_rows=[]
    for file, label in [('rf_deposit_regressor.pkl','RF Regressor'),('xgb_deposit_regressor.pkl','XGB Regressor'),('lgbm_deposit_regressor.pkl','LGBM Regressor')]:
        m=_load(file)
        if m is not None: reg_rows.append({'Model': label, **regression_metrics(dep_test[TARGET_DEPOSIT], m.predict(x_y(dep_test, TARGET_DEPOSIT)[0]))})
    save_results(reg_rows, 'reports/results/deposit_regressor_comparison.csv')
    print('Evaluation reports written to reports/results')
if __name__ == '__main__': main()

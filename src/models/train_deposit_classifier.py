from __future__ import annotations
from src.models.common import chronological_split, classifier_metrics, load_canonical_data, save_pickle, x_y
from src.utils.config import RANDOM_STATE, TARGET_HAS_DEPOSIT

def _spw(y):
    pos = max(int((y==1).sum()), 1); neg = max(int((y==0).sum()), 1); return neg/pos

def main() -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    train, _, test = chronological_split(load_canonical_data())
    Xtr, ytr = x_y(train, TARGET_HAS_DEPOSIT); Xte, yte = x_y(test, TARGET_HAS_DEPOSIT)
    models = {
        'rf': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'xgb': XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, eval_metric='logloss', scale_pos_weight=_spw(ytr), random_state=RANDOM_STATE, n_jobs=2),
        'lgbm': LGBMClassifier(n_estimators=80, learning_rate=0.08, scale_pos_weight=_spw(ytr), random_state=RANDOM_STATE, verbose=-1),
    }
    rows=[]
    for name, model in models.items():
        model.fit(Xtr, ytr); pred=model.predict(Xte); score=model.predict_proba(Xte)[:,1] if hasattr(model,'predict_proba') else pred
        save_pickle(model, f'{name}_deposit_classifier.pkl'); rows.append({'Model': name, **classifier_metrics(yte, pred, score)})
    return rows
if __name__ == '__main__': print(main())

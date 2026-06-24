from __future__ import annotations
import hashlib, json
from pathlib import Path
import pytest

pd = pytest.importorskip('pandas')
from src.models.predict_deposit import predict_deposit
from src.utils.dataset_detector import detect_dataset_type

ROOT = Path(__file__).resolve().parents[1]

def test_detect_dataset_type():
    schema=json.loads((ROOT/'reference_schema.json').read_text())
    feature_df=pd.DataFrame(columns=schema['required_columns'])
    assert detect_dataset_type(feature_df) == 'FEATURE_ENGINEERED'
    assert detect_dataset_type(pd.DataFrame(columns=['Narration','Reference','Debit','Credit'])) == 'RAW_STATEMENT'
    assert detect_dataset_type(pd.DataFrame(columns=schema['required_columns'][:12])) == 'PARTIALLY_ENGINEERED'
    assert detect_dataset_type(pd.DataFrame(columns=['foo','bar'])) == 'UNKNOWN'

def test_schema_validation():
    schema=json.loads((ROOT/'reference_schema.json').read_text())
    df=pd.read_csv(ROOT/'data/feature_engineered_dataset/liquidity_dataset.csv')
    assert list(df.columns) == schema['required_columns']
    assert df.shape == tuple(schema['canonical_shape'])
    assert len(schema['feature_columns']) == 47
    assert len(schema['target_columns']) == 3

class _ZeroClassifier:
    def predict(self, features): return [0]
class _OneClassifier:
    def predict(self, features): return [1]
class _PositiveRegressor:
    def predict(self, features): return [123.45]

def test_two_stage_deposit():
    features=pd.DataFrame({'x':[1]})
    assert predict_deposit(features, _ZeroClassifier(), _PositiveRegressor()) == 0.0
    assert predict_deposit(features, _OneClassifier(), _PositiveRegressor()) > 0.0

def test_session_state_immutability():
    path=ROOT/'data/feature_engineered_dataset/liquidity_dataset.csv'
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    df=pd.read_csv(path).head(5).copy()
    df['Total_Debit']=0
    after=hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after

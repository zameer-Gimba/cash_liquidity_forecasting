"""Central, import-safe configuration for the liquidity forecasting project.

This module intentionally has no third-party dependencies so Streamlit pages can
import constants even when optional ML packages are unavailable.
"""
from __future__ import annotations
from pathlib import Path
from typing import Union

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
FEATURE_DATA_DIR = DATA_DIR / "feature_engineered_dataset"
MODEL_DIR = ROOT_DIR / "models" / "saved_models"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
RANDOM_STATE = 42

CANONICAL_DATASET_PATH = "data/feature_engineered_dataset/liquidity_dataset.csv"
SCHEMA_PATH = "reference_schema.json"
DEPRECATED_DATASET_PATH = "data/feature_engineered_dataset/features_DEPRECATED.csv"
FEATURE_COLS = [
    "Total_Debit", "Total_Credit", "Transaction_Count", "Closing_Balance",
    "Net_Flow", "Withdrawal_Amount", "Withdrawal_Count", "Deposit_Amount",
    "Deposit_Count", "Airtime_Amount", "Airtime_Count", "BillPayment_Amount",
    "BillPayment_Count", "WalletTransfer_Amount", "WalletTransfer_Count",
    "Transfer_Amount", "Transfer_Count", "Levy_Amount", "Levy_Count",
    "Commission_Amount", "Commission_Count", "Reversal_Amount", "Reversal_Count",
    "Insurance_Amount", "Insurance_Count", "Other_Amount", "Other_Count",
    "DayOfWeek", "Month", "WeekOfMonth", "Quarter", "IsWeekend",
    "Average_Withdrawal_Size", "Average_Deposit_Size", "Credit_Debit_Ratio",
    "Lag_1_Withdrawal_Amount", "Lag_1_Deposit_Amount",
    "Lag_7_Withdrawal_Amount", "Lag_7_Deposit_Amount",
    "Lag_30_Withdrawal_Amount", "Lag_30_Deposit_Amount",
    "Rolling_7_Day_Withdrawal_Amount", "Rolling_7_Day_Deposit_Amount",
    "Rolling_30_Day_Withdrawal_Amount", "Rolling_30_Day_Deposit_Amount",
    "Has_Deposit",
]
TARGET_WITHDRAWAL = "Target_Next_Day_Withdrawal_Amount"
TARGET_DEPOSIT = "Target_Next_Day_Deposit_Amount"
TARGET_HAS_DEPOSIT = "Target_Has_Deposit_Tomorrow"
DATE_COL = "TransactionDate"
TARGET_COLUMN = TARGET_WITHDRAWAL
DATE_COLUMN = DATE_COL
SAFETY_BUFFER_WITHDRAWAL = 0.15
SAFETY_BUFFER_DEPOSIT = 0.10
SAFETY_BUFFER = SAFETY_BUFFER_WITHDRAWAL
LSTM_SEQUENCE_LENGTH = 30
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MODEL_SUFFIX_WITHDRAWAL = "_withdrawal"
MODEL_SUFFIX_DEP_CLF = "_deposit_classifier"
MODEL_SUFFIX_DEP_REG = "_deposit_regressor"

class ProjectConfig:
    """Small compatibility container for legacy code expecting config attributes."""
    target_column = TARGET_COLUMN
    date_column = DATE_COLUMN
    safety_buffer = SAFETY_BUFFER_WITHDRAWAL
    random_state = RANDOM_STATE
    model_dir = MODEL_DIR
    results_dir = RESULTS_DIR
    figures_dir = FIGURES_DIR

def project_path(relative: Union[str, Path]) -> Path:
    """Resolve a repository-relative path from the project root."""
    return ROOT_DIR / relative

def ensure_directories() -> None:
    """Create output directories without touching the immutable dataset file."""
    for directory in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, FEATURE_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

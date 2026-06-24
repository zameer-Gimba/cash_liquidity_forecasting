# src/utils/config.py
# Canonical configuration for Predictive Liquidity Model for Agent Banking
# This file is the single source of truth for all constants used across the project.

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR              = Path(__file__).resolve().parents[2]
DATA_DIR              = ROOT_DIR / "data" / "feature_engineered_dataset"
MODEL_DIR             = ROOT_DIR / "models" / "saved_models"
REPORTS_DIR           = ROOT_DIR / "reports" / "results"
SCHEMA_PATH           = ROOT_DIR / "reference_schema.json"
CANONICAL_DATASET_PATH = DATA_DIR / "liquidity_dataset.csv"
DEPRECATED_DATASET_PATH = DATA_DIR / "features_DEPRECATED.csv"

# ── Column names ─────────────────────────────────────────────────────────────
DATE_COL           = "TransactionDate"
TARGET_WITHDRAWAL  = "Target_Next_Day_Withdrawal_Amount"
TARGET_DEPOSIT     = "Target_Next_Day_Deposit_Amount"
TARGET_HAS_DEPOSIT = "Target_Has_Deposit_Tomorrow"

# ── Feature columns (47) — exact order matches liquidity_dataset.csv ─────────
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

# ── Model settings ───────────────────────────────────────────────────────────
SAFETY_BUFFER_WITHDRAWAL = 0.15   # 15% buffer on predicted withdrawal
SAFETY_BUFFER_DEPOSIT    = 0.10   # 10% buffer on predicted deposit top-up
LSTM_SEQUENCE_LENGTH     = 30
TRAIN_SPLIT              = 0.70
VAL_SPLIT                = 0.15
TEST_SPLIT               = 0.15

# ── Model artefact naming ────────────────────────────────────────────────────
MODEL_SUFFIX_WITHDRAWAL  = "_withdrawal"
MODEL_SUFFIX_DEP_CLF     = "_deposit_classifier"
MODEL_SUFFIX_DEP_REG     = "_deposit_regressor"

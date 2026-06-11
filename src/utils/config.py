"""Central configuration for the liquidity forecasting project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
FEATURE_DATA_DIR = DATA_DIR / "feature_engineered_dataset"
MODEL_DIR = ROOT_DIR / "models" / "saved_models"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
RANDOM_STATE = 42
TARGET_COLUMN = "Target_Cash_Demand_Next_Day"
DATE_COLUMN = "TransactionDate"
SAFETY_BUFFER = 0.15


@dataclass(frozen=True)
class ProjectConfig:
    """Runtime configuration for training, prediction, and dashboards."""

    target_column: str = TARGET_COLUMN
    date_column: str = DATE_COLUMN
    safety_buffer: float = SAFETY_BUFFER
    random_state: int = RANDOM_STATE
    model_dir: Path = MODEL_DIR
    results_dir: Path = RESULTS_DIR
    figures_dir: Path = FIGURES_DIR


def ensure_directories() -> None:
    """Create project output directories when they do not already exist."""
    for directory in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, FEATURE_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

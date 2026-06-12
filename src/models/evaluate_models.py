"""Aggregate model evaluation results into comparison tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.config import RESULTS_DIR, TABLES_DIR, ensure_directories


def collect_regression_metrics(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Collect regression metric JSON files into a sorted comparison table."""
    rows: list[dict[str, float | str]] = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        metrics = json.loads(path.read_text())
        if {"MAE", "RMSE", "MAPE", "R2"}.issubset(metrics):
            rows.append({"Model": path.name.replace("_metrics.json", ""), "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": metrics["MAPE"], "R2": metrics["R2"]})
    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"], ascending=[True, True]) if rows else pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE", "R2"])


def save_comparison_tables() -> Path:
    """Save model comparison table and best-model metadata."""
    ensure_directories()
    table = collect_regression_metrics()
    output = TABLES_DIR / "model_comparison.csv"
    table.to_csv(output, index=False)
    if not table.empty:
        (RESULTS_DIR / "best_model.json").write_text(json.dumps(table.iloc[0].to_dict(), indent=2))
    return output


if __name__ == "__main__":
    print(save_comparison_tables())

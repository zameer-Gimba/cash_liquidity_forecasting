"""Model comparison visualizations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import FIGURES_DIR, TABLES_DIR


def plot_model_comparison(table_path: Path = TABLES_DIR / "model_comparison.csv", output_dir: Path = FIGURES_DIR) -> Path:
    """Create a visual RMSE/MAE comparison chart."""
    table = pd.read_csv(table_path)
    melted = table.melt(id_vars="Model", value_vars=["MAE", "RMSE", "MAPE"], var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="Model", y="Value", hue="Metric", ax=ax)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Model Comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

"""End-to-end preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path

from src.preprocessing.clean_data import clean_transactions, load_dataset, outlier_report
from src.preprocessing.feature_engineering import build_feature_dataset
from src.utils.config import FEATURE_DATA_DIR, ensure_directories


def run_preprocessing(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Clean raw data, engineer features, save the feature dataset, and return its path."""
    ensure_directories()
    raw = load_dataset(input_path)
    cleaned = clean_transactions(raw)
    features = build_feature_dataset(cleaned)
    output = Path(output_path) if output_path else FEATURE_DATA_DIR / "features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    outlier_report(cleaned).to_csv(FEATURE_DATA_DIR / "outlier_report.csv", index=False)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build feature-engineered liquidity dataset")
    parser.add_argument("input_path")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()
    print(run_preprocessing(args.input_path, args.output_path))

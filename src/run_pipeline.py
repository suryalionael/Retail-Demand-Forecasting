#!/usr/bin/env python3
import argparse
import sys

from src.pipelines.training import main as train_main
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Retail Demand Forecasting Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument("--download-data", action="store_true", help="Download dataset")
    parser.add_argument("--run-eda", action="store_true", help="Run EDA notebook")
    return parser.parse_args()


def main():
    setup_logger()
    args = parse_args()
    if args.download_data:
        from src.data.ingestion import DataIngestion

        ing = DataIngestion()
        success = ing.download_favorita()
        if success:
            print("Dataset downloaded successfully")
        else:
            print("Dataset download failed. See data/raw/ for manual instructions.")
            return 1
    if args.run_eda:
        import subprocess

        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "notebooks/eda.ipynb"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("EDA notebook executed successfully")
        else:
            print(f"EDA failed: {result.stderr}")
            return 1
    results = train_main()
    return 0 if results.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

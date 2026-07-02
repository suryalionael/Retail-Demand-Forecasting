import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    pass


class DataIngestion:
    def __init__(self, raw_path: str = "data/raw"):
        self.raw_path = Path(raw_path)
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def download_favorita(self, force: bool = False) -> bool:
        try:
            import kagglehub

            dataset_path = kagglehub.competition_download("favorita-grocery-sales-forecasting")
            if dataset_path:
                logger.info(f"Dataset downloaded to {dataset_path}")
                return True
            return False
        except ImportError:
            logger.warning("kagglehub not installed. Attempting Kaggle API...")
            return self._download_via_kaggle_api()
        except Exception as e:
            logger.error(f"Failed to download Favorita dataset: {e}")
            return False

    def _download_via_kaggle_api(self) -> bool:
        try:
            import subprocess
            import zipfile

            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", "favorita-grocery-sales-forecasting"],
                capture_output=True,
                text=True,
                cwd=str(self.raw_path),
            )
            if result.returncode != 0:
                logger.error(f"Kaggle API failed: {result.stderr}")
                return False
            zip_path = self.raw_path / "favorita-grocery-sales-forecasting.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.raw_path)
                zip_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Kaggle API download failed: {e}")
            return False

    def load_favorita_data(self) -> dict[str, pd.DataFrame]:
        files = {
            "train": self.raw_path / "train.csv",
            "test": self.raw_path / "test.csv",
            "stores": self.raw_path / "stores.csv",
            "items": self.raw_path / "items.csv",
            "transactions": self.raw_path / "transactions.csv",
            "oil": self.raw_path / "oil.csv",
            "holidays_events": self.raw_path / "holidays_events.csv",
        }
        data = {}
        for name, path in files.items():
            if path.exists():
                data[name] = pd.read_csv(path)
                logger.info(f"Loaded {name}: {len(data[name])} rows")
            else:
                logger.warning(f"File not found: {path}")
        return data

    def load_m5_data(self) -> dict[str, pd.DataFrame]:
        files = {
            "calendar": self.raw_path / "calendar.csv",
            "sell_prices": self.raw_path / "sell_prices.csv",
            "sales_train_validation": self.raw_path / "sales_train_validation.csv",
            "sales_train_evaluation": self.raw_path / "sales_train_evaluation.csv",
            "sample_submission": self.raw_path / "sample_submission.csv",
        }
        data = {}
        for name, path in files.items():
            if path.exists():
                data[name] = pd.read_csv(path)
                logger.info(f"Loaded {name}: {len(data[name])} rows")
        return data

    def load_rossmann_data(self) -> dict[str, pd.DataFrame]:
        files = {
            "train": self.raw_path / "train.csv",
            "test": self.raw_path / "test.csv",
            "store": self.raw_path / "store.csv",
        }
        data = {}
        for name, path in files.items():
            if path.exists():
                data[name] = pd.read_csv(path)
                logger.info(f"Loaded {name}: {len(data[name])} rows")
        return data

    def load_data(self, dataset: str = "favorita") -> dict[str, pd.DataFrame]:
        dataset = dataset.lower()
        if dataset == "favorita":
            return self.load_favorita_data()
        elif dataset == "m5":
            return self.load_m5_data()
        elif dataset == "rossmann":
            return self.load_rossmann_data()
        else:
            raise DataIngestionError(f"Unknown dataset: {dataset}")

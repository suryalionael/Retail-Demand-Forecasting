import logging

import mlflow
import numpy as np
import pandas as pd

from src.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self, config: dict, model_uri: str | None = None):
        self.config = config
        self.model_uri = model_uri
        self.feature_engineer = FeatureEngineer(
            lags=config["features"]["lags"],
            rolling_windows=config["features"]["rolling_windows"],
            rolling_stats=config["features"]["rolling_stats"],
        )

    def load_model(self, run_id: str | None = None):
        if run_id:
            self.model_uri = f"runs:/{run_id}/model"
        if self.model_uri:
            self.model = mlflow.pyfunc.load_model(self.model_uri)
        return self

    def predict(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        logger.info(f"Generating predictions for horizon={horizon}")
        df = df.copy()
        df = self.feature_engineer.create_calendar_features(df)
        predictions = []
        for _ in range(horizon):
            df = self.feature_engineer.create_lag_features(df, ["item_nbr"], "sales")
            df = self.feature_engineer.create_rolling_features(df, ["item_nbr"], "sales")
            feature_cols = [
                c for c in df.columns if df[c].dtype in ["int64", "float64"] and c not in ["sales"]
            ]
            df = df.dropna(subset=feature_cols)
            if len(df) > 0:
                X = df[feature_cols].iloc[-1:].values
                pred = self.model.predict(X)[0]
                predictions.append(pred)
        return pd.DataFrame({"prediction": predictions, "horizon": range(1, horizon + 1)})

    def batch_predict(self, df: pd.DataFrame, model: object, feature_cols: list[str]) -> np.ndarray:
        X = df[feature_cols].dropna()
        return model.predict(X)

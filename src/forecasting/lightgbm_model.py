import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LightGBMForecaster:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model = None
        self.fitted = False
        self.feature_names: list[str] = []

    def _build_model(self, **kwargs):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("LightGBM is not installed. pip install lightgbm") from e

        params = {
            "n_estimators": self.config.get("n_estimators", 1000),
            "max_depth": self.config.get("max_depth", -1),
            "learning_rate": self.config.get("learning_rate", 0.01),
            "num_leaves": self.config.get("num_leaves", 31),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "reg_alpha": self.config.get("reg_alpha", 0),
            "reg_lambda": self.config.get("reg_lambda", 0),
            "random_state": 42,
            "verbose": -1,
        }
        params.update(kwargs)
        return lgb.LGBMRegressor(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "LightGBMForecaster":
        self.feature_names = list(X_train.columns)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, y_val)]
        self.model = self._build_model()
        self.model.fit(
            X_train.values,
            y_train,
            eval_set=eval_set,
        )
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted yet.")
        return self.model.predict(X.values)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted yet.")
        importance = self.model.feature_importances_
        return pd.DataFrame({"feature": self.feature_names, "importance": importance}).sort_values(
            "importance", ascending=False
        )

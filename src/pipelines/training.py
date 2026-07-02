import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.data.cleaning import DataCleaner
from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.evaluation.backtesting import WalkForwardValidator
from src.evaluation.metrics import calculate_all_metrics, create_comparison_table
from src.features.engineering import FeatureEngineer
from src.forecasting.naive import NaiveSeasonalForecaster
from src.forecasting.prophet_model import ProphetForecaster
from src.forecasting.xgboost_model import XGBoostForecaster
from src.visualization.plots import Visualizer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ingestion = DataIngestion(config["data"]["raw_path"])
        self.cleaner = DataCleaner()
        self.validator = DataValidation()
        self.feature_engineer = FeatureEngineer(
            lags=config["features"]["lags"],
            rolling_windows=config["features"]["rolling_windows"],
            rolling_stats=config["features"]["rolling_stats"],
        )
        self.visualizer = Visualizer(
            style=config["visualization"]["style"],
            dpi=config["visualization"]["dpi"],
            figsize=tuple(config["visualization"]["figsize"]),
        )
        self.results: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        logger.info("Starting training pipeline")
        data = self.ingestion.load_data(self.config["data"]["dataset"])
        if "train" not in data:
            logger.error("No training data found. Attempting download...")
            success = self.ingestion.download_favorita()
            if not success:
                logger.error("Cannot download dataset. Place files in data/raw/ manually.")
                return {"status": "failed", "error": "No training data"}
            data = self.ingestion.load_data(self.config["data"]["dataset"])
        df = data["train"].copy()
        df = self.cleaner.clean_column_names(df)
        df = self.cleaner.clean_sales_data(df)
        holiday_df = data.get("holidays_events", None)
        if holiday_df is not None:
            holiday_df = self.cleaner.clean_column_names(holiday_df)
        logger.info(f"Training data shape: {df.shape}")
        date_col = "date"
        target = "sales"
        group_cols = ["item_nbr"]
        df[date_col] = pd.to_datetime(df[date_col])
        promo_col = "onpromotion" if "onpromotion" in df.columns else None
        df = self.feature_engineer.create_all_features(
            df,
            group_cols=group_cols,
            target=target,
            date_col=date_col,
            holiday_df=holiday_df,
            promo_col=promo_col,
        )
        df = self._prepare_for_prophet(df, group_cols)
        self._train_and_evaluate_all_models(df, date_col, target, group_cols)
        self._generate_figures(df)
        logger.info("Training pipeline completed successfully")
        return {"status": "success", "results": self.results}

    def _prepare_for_prophet(self, df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if "store_nbr" in df.columns and "item_nbr" in df.columns:
            df["sku_id"] = df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)
        return df

    def _train_and_evaluate_all_models(self, df, date_col, target, group_cols):
        logger.info("Training and evaluating all models...")
        top_skus = df.groupby("item_nbr")[target].sum().sort_values(ascending=False).head(10).index
        sample_df = df[df["item_nbr"].isin(top_skus)].copy()
        sample_df = sample_df.sort_values([date_col, "item_nbr"]).dropna(subset=[target])
        logger.info(f"Sample data shape for modeling: {sample_df.shape}")
        with mlflow.start_run(run_name="full_pipeline"):
            mlflow.log_params(self.config)
            for model_name in ["naive", "prophet", "xgboost"]:
                if not self.config["models"][model_name]["enabled"]:
                    continue
                logger.info(f"Training {model_name}...")
                try:
                    self._train_model(model_name, sample_df, date_col, target, group_cols)
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
            all_metrics = {
                name: res["metrics"] for name, res in self.results.items() if "metrics" in res
            }
            if all_metrics:
                comparison = create_comparison_table(all_metrics)
                logger.info(f"\nModel Comparison:\n{comparison}")
                for model_name, metrics in all_metrics.items():
                    mlflow.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items()})

    def _train_model(self, model_name, df, date_col, target, group_cols):
        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.log_params(self.config["models"][model_name])
            if model_name == "naive":
                self._train_naive(df, date_col, target)
            elif model_name == "prophet":
                self._train_prophet(df, date_col, target)
            elif model_name == "xgboost":
                self._train_xgboost(df, date_col, target, group_cols)

    def _train_naive(self, df, date_col, target):
        model = NaiveSeasonalForecaster(
            seasonality=self.config["models"]["naive"]["seasonality"],
            period=self.config["models"]["naive"]["period"],
        )
        wfv = WalkForwardValidator(
            n_folds=self.config["evaluation"]["cv_folds"],
            initial_days=self.config["evaluation"]["cv_initial"],
            horizon_days=self.config["evaluation"]["cv_horizon"],
            step_days=self.config["evaluation"]["cv_step"],
        )
        daily = df.groupby(date_col)[target].sum().reset_index()
        folds = wfv.split_dataframe(daily, date_col)
        all_preds = []
        all_trues = []
        for train_df, test_df, _fold_info in folds:
            model.fit(train_df, date_col=date_col, target=target)
            preds = model.predict_with_series(test_df)
            all_preds.extend(preds)
            all_trues.extend(test_df[target].values)
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        metrics = calculate_all_metrics(all_trues, all_preds)
        self.results["naive"] = {"metrics": metrics, "predictions": all_preds}
        mlflow.log_metrics(metrics)
        logger.info(f"Naive metrics: {metrics}")

    def _train_prophet(self, df, date_col, target):
        model = ProphetForecaster(self.config["models"]["prophet"])
        wfv = WalkForwardValidator(
            n_folds=self.config["evaluation"]["cv_folds"],
            initial_days=self.config["evaluation"]["cv_initial"],
            horizon_days=self.config["evaluation"]["cv_horizon"],
            step_days=self.config["evaluation"]["cv_step"],
        )
        top_sku = df.groupby("item_nbr")[target].sum().idxmax()
        sku_df = df[df["item_nbr"] == top_sku].sort_values(date_col).dropna(subset=[target])
        if len(sku_df) < 100:
            logger.warning(f"Not enough data for Prophet: {len(sku_df)} rows")
            self.results["prophet"] = {
                "metrics": {
                    k: float("nan")
                    for k in ["mae", "rmse", "mape", "smape", "wape", "bias", "forecast_accuracy"]
                },
                "predictions": np.array([]),
            }
            return
        folds = wfv.split_dataframe(sku_df, date_col)
        all_preds = []
        all_trues = []
        for train_df, test_df, _fold_info in folds[:2]:
            try:
                model.fit(train_df, date_col=date_col, target=target)
                preds = model.predict_with_series(test_df, date_col=date_col)
                all_preds.extend(preds)
                all_trues.extend(test_df[target].values)
            except Exception as e:
                logger.error(f"Prophet fold failed: {e}")
                continue
        if len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_trues = np.array(all_trues)
            metrics = calculate_all_metrics(all_trues, all_preds)
            self.results["prophet"] = {"metrics": metrics, "predictions": all_preds}
            mlflow.log_metrics(metrics)
            logger.info(f"Prophet metrics: {metrics}")

    def _train_xgboost(self, df, date_col, target, group_cols):
        model = XGBoostForecaster(self.config["models"]["xgboost"])
        wfv = WalkForwardValidator(
            n_folds=self.config["evaluation"]["cv_folds"],
            initial_days=self.config["evaluation"]["cv_initial"],
            horizon_days=self.config["evaluation"]["cv_horizon"],
            step_days=self.config["evaluation"]["cv_step"],
        )
        top_sku = df.groupby("item_nbr")[target].sum().idxmax()
        sku_df = df[df["item_nbr"] == top_sku].sort_values(date_col).dropna(subset=[target])
        exclude_cols = [date_col, "sku_id", "item_nbr", "store_nbr"]
        feature_cols = [
            c
            for c in sku_df.columns
            if c not in exclude_cols and sku_df[c].dtype in ["int64", "float64"]
        ]
        cols_for_xgb = feature_cols + [target]
        sku_df = sku_df.dropna(subset=cols_for_xgb)
        if len(sku_df) < 100:
            logger.warning(f"Not enough data for XGBoost: {len(sku_df)} rows")
            self.results["xgboost"] = {
                "metrics": {
                    k: float("nan")
                    for k in ["mae", "rmse", "mape", "smape", "wape", "bias", "forecast_accuracy"]
                },
                "predictions": np.array([]),
            }
            return
        folds = wfv.split_dataframe(sku_df, date_col)
        all_preds = []
        all_trues = []
        for train_df, test_df, _fold_info in folds[:2]:
            try:
                train_clean = train_df[cols_for_xgb].dropna()
                test_clean = test_df[cols_for_xgb].dropna()
                if len(train_clean) < 50 or len(test_clean) < 10:
                    continue
                y_train = train_clean[target].values
                y_test = test_clean[target].values
                X_train = train_clean[feature_cols]
                X_test = test_clean[feature_cols]
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                all_preds.extend(preds)
                all_trues.extend(y_test)
            except Exception as e:
                logger.error(f"XGBoost fold failed: {e}")
                continue
        if len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_trues = np.array(all_trues)
            metrics = calculate_all_metrics(all_trues, all_preds)
            imp = model.get_feature_importance()
            self.results["xgboost"] = {
                "metrics": metrics,
                "predictions": all_preds,
                "feature_importance": imp,
            }
            mlflow.log_metrics(metrics)
            logger.info(f"XGBoost metrics: {metrics}")

    def _generate_figures(self, df):
        logger.info("Generating figures...")
        date_col = "date"
        target = "sales"
        fig = self.visualizer.plot_time_series(df, date_col, target)
        self.visualizer.save_figure(fig, "time_series.png")
        fig = self.visualizer.plot_sales_distribution(df, target)
        self.visualizer.save_figure(fig, "sales_distribution.png")
        fig = self.visualizer.plot_weekly_seasonality(df, date_col, target)
        self.visualizer.save_figure(fig, "weekly_seasonality.png")
        fig = self.visualizer.plot_monthly_seasonality(df, date_col, target)
        self.visualizer.save_figure(fig, "monthly_seasonality.png")
        if "is_promotion" in df.columns or "promo" in df.columns:
            fig = self.visualizer.plot_promotion_analysis(df, target)
            self.visualizer.save_figure(fig, "promotion_analysis.png")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            fig = self.visualizer.plot_heatmap(numeric_df, "Feature Correlation Heatmap")
            self.visualizer.save_figure(fig, "correlation_heatmap.png")
        if "family" in df.columns:
            fig = self.visualizer.plot_category_analysis(df, "family", target)
            self.visualizer.save_figure(fig, "category_analysis.png")
        if "naive" in self.results:
            fig = self.visualizer.plot_actual_vs_forecast(
                np.array(self.results["naive"]["predictions"]),
                np.array(self.results["naive"]["predictions"]),
                "Naive: Actual vs Forecast",
            )
            self.visualizer.save_figure(fig, "naive_actual_vs_forecast.png")
        logger.info("Figures generated successfully")


def main():
    from src.utils.config import load_config
    from src.utils.logger import setup_logger

    setup_logger()
    config = load_config()
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    print(f"Pipeline status: {results.get('status')}")
    if "results" in results and results["results"]:
        comparison = create_comparison_table(
            {name: res["metrics"] for name, res in results["results"].items() if "metrics" in res}
        )
        print("\nFinal Model Comparison:")
        print(comparison)
    return results


if __name__ == "__main__":
    main()

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9_]", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
        )
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "auto",
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if strategy == "auto":
                if df[col].dtype in ["int64", "float64"]:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna("unknown")
            elif strategy == "zero":
                df[col] = df[col].fillna(fill_value)
            elif strategy == "ffill":
                df[col] = df[col].ffill()
            elif strategy == "bfill":
                df[col] = df[col].bfill()
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
        return df

    def detect_outliers_iqr(
        self, df: pd.DataFrame, column: str, multiplier: float = 1.5
    ) -> pd.Series:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        return (df[column] < lower) | (df[column] > upper)

    def detect_outliers_zscore(
        self, df: pd.DataFrame, column: str, threshold: float = 3.0
    ) -> pd.Series:
        mean = df[column].mean()
        std = df[column].std()
        if std == 0:
            return pd.Series(False, index=df.index)
        z_scores = np.abs((df[column] - mean) / std)
        return z_scores > threshold

    def cap_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "iqr",
        multiplier: float = 1.5,
    ) -> pd.DataFrame:
        df = df.copy()
        df[column] = df[column].astype(float)
        if method == "iqr":
            outliers = self.detect_outliers_iqr(df, column, multiplier)
        else:
            outliers = self.detect_outliers_zscore(df, column, multiplier)
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
        else:
            mean = df[column].mean()
            std = df[column].std()
            lower = mean - multiplier * std
            upper = mean + multiplier * std
        df.loc[df[column] < lower, column] = lower
        df.loc[df[column] > upper, column] = upper
        n_capped = outliers.sum()
        if n_capped > 0:
            logger.info(f"Capped {n_capped} outliers in '{column}'")
        return df

    def clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "unit_sales" in df.columns and "sales" not in df.columns:
            df = df.rename(columns={"unit_sales": "sales"})
        if "sales" in df.columns:
            negative_mask = df["sales"] < 0
            if negative_mask.any():
                logger.warning(f"Found {negative_mask.sum()} negative sales values, setting to 0")
                df.loc[negative_mask, "sales"] = 0
        return df

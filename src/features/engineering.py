import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        rolling_stats: list[str] | None = None,
    ):
        self.lags = lags or [1, 7, 14, 28, 56]
        self.rolling_windows = rolling_windows or [7, 14, 28, 56]
        self.rolling_stats = rolling_stats or ["mean", "median", "std", "min", "max"]

    def create_lag_features(
        self, df: pd.DataFrame, group_cols: list[str], target: str = "sales"
    ) -> pd.DataFrame:
        df = df.copy()
        for lag in self.lags:
            df[f"lag_{lag}"] = df.groupby(group_cols, observed=True)[target].shift(lag)
        return df

    def create_rolling_features(
        self, df: pd.DataFrame, group_cols: list[str], target: str = "sales"
    ) -> pd.DataFrame:
        df = df.copy()
        for window in self.rolling_windows:
            group = df.groupby(group_cols, observed=True)[target]
            if "mean" in self.rolling_stats:
                df[f"rolling_mean_{window}"] = group.transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
            if "median" in self.rolling_stats:
                df[f"rolling_median_{window}"] = group.transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).median()
                )
            if "std" in self.rolling_stats:
                df[f"rolling_std_{window}"] = group.transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).std()
                )
            if "min" in self.rolling_stats:
                df[f"rolling_min_{window}"] = group.transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).min()
                )
            if "max" in self.rolling_stats:
                df[f"rolling_max_{window}"] = group.transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).max()
                )
        return df

    def create_ema_features(
        self, df: pd.DataFrame, group_cols: list[str], target: str = "sales"
    ) -> pd.DataFrame:
        df = df.copy()
        for span in [7, 14, 28]:
            df[f"ema_{span}"] = df.groupby(group_cols, observed=True)[target].transform(
                lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
            )
        return df

    def create_calendar_features(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["day_of_week"] = df[date_col].dt.dayofweek
        df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
        df["month"] = df[date_col].dt.month
        df["quarter"] = df[date_col].dt.quarter
        df["year"] = df[date_col].dt.year
        df["day_of_month"] = df[date_col].dt.day
        df["day_of_year"] = df[date_col].dt.dayofyear
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
        df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
        df["weekend_saturday"] = (df["day_of_week"] == 5).astype(int)
        df["weekend_sunday"] = (df["day_of_week"] == 6).astype(int)
        return df

    def create_promotion_features(
        self,
        df: pd.DataFrame,
        promo_col: str | None = None,
        price_col: str | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        if promo_col and promo_col in df.columns:
            df["is_promotion"] = df[promo_col].astype(int)
        if price_col and price_col in df.columns:
            price_orig = f"{price_col}_original"
            if price_orig not in df.columns:
                df["price_rank"] = df.groupby("item_nbr", observed=True)[price_col].rank(
                    method="dense"
                )
        return df

    def create_holiday_features(
        self,
        df: pd.DataFrame,
        holiday_df: pd.DataFrame | None = None,
        date_col: str = "date",
    ) -> pd.DataFrame:
        df = df.copy()
        df["is_holiday"] = 0
        if holiday_df is not None and "date" in holiday_df.columns:
            holiday_df["date"] = pd.to_datetime(holiday_df["date"])
            holiday_dates = holiday_df["date"].unique()
            df.loc[df[date_col].isin(holiday_dates), "is_holiday"] = 1
        return df

    def create_price_features(
        self,
        df: pd.DataFrame,
        price_col: str = "sell_price",
    ) -> pd.DataFrame:
        df = df.copy()
        if price_col in df.columns:
            df["price_change_1"] = df.groupby("item_nbr", observed=True)[price_col].diff(1)
            df["price_change_7"] = df.groupby("item_nbr", observed=True)[price_col].diff(7)
            df["price_rolling_mean_7"] = df.groupby("item_nbr", observed=True)[price_col].transform(
                lambda x: x.shift(1).rolling(7).mean()
            )
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "is_promotion" in df.columns and "is_weekend" in df.columns:
            df["promo_weekend"] = df["is_promotion"] * df["is_weekend"]
        if "is_promotion" in df.columns and "is_holiday" in df.columns:
            df["promo_holiday"] = df["is_promotion"] * df["is_holiday"]
        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        group_cols: list[str] | None = None,
        target: str = "sales",
        date_col: str = "date",
        holiday_df: pd.DataFrame | None = None,
        promo_col: str | None = None,
        price_col: str | None = None,
    ) -> pd.DataFrame:
        if group_cols is None:
            group_cols = ["item_nbr"]
        df = df.copy()
        df = self.create_calendar_features(df, date_col)
        df = self.create_lag_features(df, group_cols, target)
        df = self.create_rolling_features(df, group_cols, target)
        df = self.create_ema_features(df, group_cols, target)
        df = self.create_holiday_features(df, holiday_df, date_col)
        if promo_col or price_col:
            df = self.create_promotion_features(df, promo_col, price_col)
        if price_col:
            df = self.create_price_features(df, price_col)
        df = self.create_interaction_features(df)
        return df

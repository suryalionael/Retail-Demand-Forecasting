import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidation:
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        result = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
        result = result[result["missing_count"] > 0].sort_values("missing_count", ascending=False)
        return result

    def check_duplicates(self, df: pd.DataFrame, subset: list[str] | None = None) -> int:
        return df.duplicated(subset=subset).sum()

    def check_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"dtype": df.dtypes, "nunique": df.nunique(), "null_count": df.isnull().sum()}
        )

    def validate_sales_data(self, df: pd.DataFrame) -> dict:
        issues = {}
        if "sales" in df.columns:
            negative = (df["sales"] < 0).sum()
            zero = (df["sales"] == 0).sum()
            issues["negative_sales"] = int(negative)
            issues["zero_sales"] = int(zero)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            date_range = (df["date"].max() - df["date"].min()).days
            issues["date_range_days"] = date_range
            issues["missing_dates"] = self._find_missing_dates(df)
        return issues

    def _find_missing_dates(self, df: pd.DataFrame) -> int:
        full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
        observed = pd.to_datetime(df["date"].unique())
        missing = full_range.difference(observed)
        return len(missing)

    def generate_validation_report(self, df: pd.DataFrame, name: str = "dataset") -> str:
        lines = [f"=== Data Validation Report: {name} ==="]
        lines.append(f"Shape: {df.shape}")
        lines.append(f"Columns: {list(df.columns)}")
        missing = self.check_missing_values(df)
        if len(missing) > 0:
            lines.append("\nMissing values:")
            for col, row in missing.iterrows():
                lines.append(f"  {col}: {row['missing_count']} ({row['missing_pct']:.1f}%)")
        else:
            lines.append("\nNo missing values found.")
        dupes = self.check_duplicates(df)
        lines.append(f"Duplicate rows: {dupes}")
        if "sales" in df.columns:
            lines.append("\nSales stats:")
            lines.append(f"  Mean: {df['sales'].mean():.2f}")
            lines.append(f"  Median: {df['sales'].median():.2f}")
            lines.append(f"  Std: {df['sales'].std():.2f}")
            lines.append(f"  Min: {df['sales'].min():.2f}")
            lines.append(f"  Max: {df['sales'].max():.2f}")
        return "\n".join(lines)

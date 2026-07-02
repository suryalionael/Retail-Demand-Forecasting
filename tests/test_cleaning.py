import numpy as np
import pandas as pd

from src.data.cleaning import DataCleaner


def test_clean_column_names():
    df = pd.DataFrame({"Sales Amount ": [1], "  Store-ID": [2]})
    cleaner = DataCleaner()
    result = cleaner.clean_column_names(df)
    assert "sales_amount" in result.columns
    assert "store_id" in result.columns


def test_handle_missing_values_zero():
    df = pd.DataFrame({"sales": [1.0, np.nan, 3.0], "category": ["a", None, "c"]})
    cleaner = DataCleaner()
    result = cleaner.handle_missing_values(df, strategy="auto")
    assert result["sales"].iloc[1] == 0.0
    assert result["category"].iloc[1] == "unknown"


def test_handle_missing_values_ffill():
    df = pd.DataFrame({"sales": [1.0, np.nan, 3.0]})
    cleaner = DataCleaner()
    result = cleaner.handle_missing_values(df, strategy="ffill")
    assert result["sales"].iloc[1] == 1.0


def test_detect_outliers_iqr():
    df = pd.DataFrame({"sales": [1, 2, 3, 4, 5, 100]})
    cleaner = DataCleaner()
    outliers = cleaner.detect_outliers_iqr(df, "sales")
    assert outliers.iloc[5]  # 100 should be outlier


def test_detect_outliers_zscore():
    df = pd.DataFrame({"sales": [1, 2, 3, 4, 5, 100]})
    cleaner = DataCleaner()
    outliers = cleaner.detect_outliers_zscore(df, "sales", threshold=2)
    assert outliers.iloc[5]  # 100 should be outlier


def test_cap_outliers():
    df = pd.DataFrame({"sales": [1, 2, 3, 4, 5, 100]})
    cleaner = DataCleaner()
    result = cleaner.cap_outliers(df, "sales", method="iqr")
    assert result["sales"].max() < 100


def test_clean_sales_data():
    df = pd.DataFrame({"date": ["2023-01-01", "bad_date", "2023-01-03"], "sales": [10, -5, 15]})
    cleaner = DataCleaner()
    result = cleaner.clean_sales_data(df)
    assert result["sales"].iloc[1] == 0

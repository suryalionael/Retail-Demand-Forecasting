import pandas as pd

from src.features.engineering import FeatureEngineer


def test_create_calendar_features():
    df = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=10, freq="D"), "sales": range(10)}
    )
    fe = FeatureEngineer()
    result = fe.create_calendar_features(df, "date")
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "quarter" in result.columns
    assert "year" in result.columns
    assert "is_weekend" in result.columns
    assert result["day_of_week"][0] == 6
    assert result["month"][0] == 1


def test_create_lag_features():
    df = pd.DataFrame(
        {"item_nbr": [1] * 10, "date": pd.date_range("2023-01-01", periods=10), "sales": range(10)}
    )
    fe = FeatureEngineer(lags=[1, 3])
    result = fe.create_lag_features(df, ["item_nbr"], "sales")
    assert "lag_1" in result.columns
    assert "lag_3" in result.columns
    assert pd.isna(result["lag_1"].iloc[0])
    assert result["lag_1"].iloc[1] == 0


def test_create_rolling_features():
    df = pd.DataFrame(
        {"item_nbr": [1] * 10, "date": pd.date_range("2023-01-01", periods=10), "sales": range(10)}
    )
    fe = FeatureEngineer(rolling_windows=[3], rolling_stats=["mean"])
    result = fe.create_rolling_features(df, ["item_nbr"], "sales")
    assert "rolling_mean_3" in result.columns


def test_ema_features():
    df = pd.DataFrame(
        {"item_nbr": [1] * 10, "date": pd.date_range("2023-01-01", periods=10), "sales": range(10)}
    )
    fe = FeatureEngineer()
    result = fe.create_ema_features(df, ["item_nbr"], "sales")
    assert "ema_7" in result.columns


def test_holiday_features():
    df = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=10, freq="D"), "sales": range(10)}
    )
    holidays = pd.DataFrame({"date": ["2023-01-01", "2023-01-07"]})
    fe = FeatureEngineer()
    result = fe.create_holiday_features(df, holidays, "date")
    assert result["is_holiday"].iloc[0] == 1
    assert result["is_holiday"].iloc[3] == 0


def test_interaction_features():
    df = pd.DataFrame(
        {
            "is_promotion": [0, 1, 0, 1],
            "is_weekend": [0, 0, 1, 1],
            "is_holiday": [0, 1, 0, 1],
            "sales": [1, 2, 3, 4],
        }
    )
    fe = FeatureEngineer()
    result = fe.create_interaction_features(df)
    assert "promo_weekend" in result.columns
    assert "promo_holiday" in result.columns
    assert result["promo_weekend"].iloc[3] == 1
    assert result["promo_holiday"].iloc[1] == 1

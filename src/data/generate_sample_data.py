from pathlib import Path

import numpy as np
import pandas as pd


def generate_favorita_like_data(
    output_dir: str = "data/raw",
    n_stores: int = 5,
    n_items: int = 20,
    n_days: int = 730,
    start_date: str = "2015-01-01",
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    np.random.seed(random_state)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    stores = pd.DataFrame(
        {
            "store_nbr": range(1, n_stores + 1),
            "city": np.random.choice(["Quito", "Guayaquil", "Cuenca", "Santo Domingo"], n_stores),
            "state": np.random.choice(["Pichincha", "Guayas", "Azuay", "Los Rios"], n_stores),
            "type": np.random.choice(["A", "B", "C", "D"], n_stores),
            "cluster": np.random.randint(1, 10, n_stores),
        }
    )
    stores.to_csv(output_path / "stores.csv", index=False)

    families = [
        "GROCERY I",
        "BEVERAGES",
        "DAIRY",
        "BREAD/BAKERY",
        "FROZEN FOODS",
        "MEATS",
        "PREPARED FOODS",
        "DELI",
        "PRODUCE",
        "EGGS",
        "CLEANING",
        "HOME CARE",
        "BABY CARE",
        "PERSONAL CARE",
        "MAGAZINES",
        "LAWN AND GARDEN",
        "PET SUPPLIES",
        "SEAFOOD",
        "LIQUOR,WINE,BEER",
        "POULTRY",
    ]
    items = pd.DataFrame(
        {
            "item_nbr": range(1, n_items + 1),
            "family": np.random.choice(families, n_items),
            "class": np.random.randint(1, 20, n_items),
            "perishable": np.random.choice([0, 1], n_items, p=[0.6, 0.4]),
        }
    )
    items.to_csv(output_path / "items.csv", index=False)

    oil = pd.DataFrame(
        {
            "date": dates,
            "dcoilwtico": 50
            + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
            + np.random.normal(0, 2, len(dates)),
        }
    )
    oil.to_csv(output_path / "oil.csv", index=False)

    holiday_types = ["Holiday", "Event", "Transfer", "Additional", "Bridge"]
    n_holidays = 30
    holiday_dates = np.random.choice(dates, n_holidays, replace=False)
    holidays = pd.DataFrame(
        {
            "date": holiday_dates,
            "type": np.random.choice(holiday_types, n_holidays),
            "locale": np.random.choice(["National", "Local", "Regional"], n_holidays),
            "description": [f"Holiday_{i}" for i in range(n_holidays)],
            "transferred": np.random.choice([True, False], n_holidays),
        }
    ).sort_values("date")
    holidays.to_csv(output_path / "holidays_events.csv", index=False)

    transactions = pd.DataFrame(
        {
            "date": np.repeat(dates, n_stores),
            "store_nbr": np.tile(range(1, n_stores + 1), len(dates)),
            "transactions": np.random.poisson(500, len(dates) * n_stores),
        }
    )
    transactions.to_csv(output_path / "transactions.csv", index=False)

    records = []
    for date in dates:
        for store in range(1, n_stores + 1):
            for item in range(1, n_items + 1):
                base_sales = 5 + 2 * np.sin(date.dayofyear * 2 * np.pi / 365)
                base_sales += 3 * np.sin(date.dayofyear * 2 * np.pi / 7)
                base_sales += np.random.poisson(3)
                is_promo = np.random.random() < 0.15
                if is_promo:
                    base_sales *= 1.5 + np.random.random() * 0.5
                onpromotion = int(is_promo) * np.random.randint(1, 10)
                records.append(
                    {
                        "date": date,
                        "store_nbr": store,
                        "item_nbr": item,
                        "unit_sales": max(0, int(base_sales)),
                        "onpromotion": onpromotion,
                    }
                )
    train = pd.DataFrame(records)
    train.columns = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    train.to_csv(output_path / "train.csv", index=False)

    test_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=30, freq="D")
    test_records = []
    for date in test_dates:
        for store in range(1, n_stores + 1):
            for item in range(1, n_items + 1):
                test_records.append(
                    {
                        "date": date,
                        "store_nbr": store,
                        "item_nbr": item,
                        "onpromotion": np.random.choice(
                            [0, 1, 2, 3, 4, 5], p=[0.8, 0.05, 0.05, 0.04, 0.03, 0.03]
                        ),
                    }
                )
    test = pd.DataFrame(test_records)
    test.to_csv(output_path / "test.csv", index=False)

    sample_submission = pd.DataFrame(
        {
            "id": [
                f"{d.date()}_{s}_{i}"
                for d in test_dates
                for s in range(1, n_stores + 1)
                for i in range(1, n_items + 1)
            ],
            "unit_sales": 0.0,
        }
    )
    sample_submission.to_csv(output_path / "sample_submission.csv", index=False)

    return {
        "train": train,
        "test": test,
        "stores": stores,
        "items": items,
        "oil": oil,
        "holidays_events": holidays,
        "transactions": transactions,
    }


if __name__ == "__main__":
    data = generate_favorita_like_data()
    print("Generated synthetic dataset:")
    for name, df in data.items():
        print(f"  {name}: {df.shape}")

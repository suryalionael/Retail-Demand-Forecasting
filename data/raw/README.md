# Dataset: Favorita Grocery Sales Forecasting

This project uses the **Corporación Favorita Grocery Sales Forecasting** dataset from Kaggle.

## Automatic Download

Run:

```bash
python src/run_pipeline.py --download-data
```

This uses `kagglehub` to download the dataset automatically.

## Manual Download

If automatic download fails:

1. Go to https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
2. Accept the competition rules
3. Download and extract the zip file
4. Place the following CSV files in `data/raw/`:

   - `train.csv`
   - `test.csv`
   - `stores.csv`
   - `items.csv`
   - `transactions.csv`
   - `oil.csv`
   - `holidays_events.csv`
   - `sample_submission.csv`

## Dataset Description

- **train.csv**: Daily sales data by store and item
- **test.csv**: Test data for predictions
- **stores.csv**: Store information (city, state, type, cluster)
- **items.csv**: Item information (family, class, perishable)
- **transactions.csv**: Transaction counts by store and date
- **oil.csv**: Daily oil prices (economic indicator)
- **holidays_events.csv**: Holidays and events data

## Alternative Datasets

If you prefer a different dataset:

1. **M5 Forecasting**: https://www.kaggle.com/competitions/m5-forecasting-accuracy
2. **Rossmann Store Sales**: https://www.kaggle.com/competitions/rossmann-store-sales
3. **Walmart Recruiting**: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

# Retail Demand Forecasting

Production-ready SKU-level demand forecasting system that improves inventory planning by outperforming naive seasonal forecasting baselines.

## Business Problem

Retail inventory planners currently estimate demand using:
- Manual adjustments
- Historical averages
- Human intuition

This causes overstocking, understocking, lost sales, and high markdown costs.

This system builds forecasting models that consistently outperform the naive seasonal approach using rigorous walk-forward validation.

## Dataset

### UCI Online Retail II

Transaction-level retail data from a UK-based online retailer (2009-2011):
- **~1 million transactions** across two sheets
- **4,000-5,000 unique SKUs**
- **4,300+ unique customers** across 38-40 countries
- **Daily transaction log** with Invoice, StockCode, Quantity, Price, Customer ID, Country

Place `online_retail_II.xlsx` in `data/raw/`. See `data/raw/README.md` for details.

### Cleaning Methodology

1. **Cancelled invoices**: Rows with invoices starting with "C" are removed
2. **Returns**: Negative quantities (returns) are removed
3. **Invalid prices**: Rows with Price <= 0 are removed
4. **Invalid quantities**: Rows with Quantity <= 0 are removed
5. **Duplicates**: Exact duplicate rows are dropped
6. **Missing values**: Descriptions and customer IDs are filled with "unknown"

### Aggregation

Transactions are aggregated to **daily SKU-level demand**:
- `(Date, StockCode)` — one row per SKU per day
- `DailyDemand` — sum of quantities sold
- `Revenue` — sum of Quantity x Price
- `NumberOfTransactions` — unique invoices per day
- `AvgPrice` — average unit price

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Data       │────▶│  Feature     │────▶│  Forecasting│
│  Ingestion  │     │  Engineering │     │  Models     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Reporting   │◀────│  Evaluation  │◀────│  Walk-Forward│
│  & Viz      │     │  & Metrics   │     │  Validation  │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Folder Structure

```
retail-demand-forecasting/
├── README.md
├── LICENSE
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── configs/              # YAML/Hydra configuration
├── data/
│   ├── raw/              # online_retail_II.xlsx
│   ├── interim/
│   └── processed/
├── models/               # Saved model artifacts
├── notebooks/            # EDA notebook
├── reports/
│   ├── figures/          # Generated visualizations
│   └── business_report.md
├── src/
│   ├── data/             # Ingestion, cleaning, validation, aggregation
│   ├── features/         # Feature engineering
│   ├── forecasting/      # Naive, Prophet, XGBoost, LightGBM
│   ├── evaluation/       # Metrics, walk-forward validation
│   ├── visualization/    # Plotting utilities
│   ├── pipelines/        # Training and prediction pipelines
│   └── utils/            # Config, logging helpers
├── tests/                # Pytest unit tests
└── .github/workflows/    # GitHub Actions CI
```

## Installation

### Prerequisites

- Python 3.12+
- The `online_retail_II.xlsx` file in `data/raw/`

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd retail-demand-forecasting

# Install with pip
pip install -r requirements.txt
pip install -e .
```

### Dataset

Place the `online_retail_II.xlsx` file in `data/raw/`. The file can be downloaded from:
- UCI ML Repository: https://archive.ics.uci.edu/dataset/502/online+retail+ii

## Docker Usage

```bash
# Build and run the pipeline
docker compose up forecasting

# Start MLflow tracking server
docker compose --profile mlflow up

# Start Jupyter notebook server
docker compose --profile jupyter up
```

## Local Development

```bash
# Run the full pipeline
python src/run_pipeline.py

# Run EDA notebook
jupyter notebook notebooks/eda.ipynb

# Run tests
pytest tests/ -v

# Run linting
ruff check src/ tests/
black --check src/ tests/
```

## MLflow

Experiment tracking is automatically configured:

```bash
mlflow ui
```

Or with Docker: `docker compose --profile mlflow up`

Access MLflow UI at `http://localhost:5001`

## EDA

The EDA notebook (`notebooks/eda.ipynb`) includes:

- Transaction overview and quality checks
- Data cleaning impact analysis
- Daily sales and revenue trends
- Weekly/monthly seasonality
- Top-selling SKUs analysis
- Country distribution
- Price distribution
- Rolling averages
- Correlation analysis
- Active SKU counts

## Feature Engineering

For each SKU, the system generates:

- **Lag features**: 1, 7, 14, 28, 56 days of past demand
- **Rolling statistics**: mean, median, std, min, max over 7/14/28/56 day windows
- **EMA**: Exponential moving averages at 7/14/28 day spans
- **Calendar**: day of week, week of year, month, quarter, year, weekend flags

All features are computed using **only historical data** to prevent leakage.

## Forecasting Models

### Naive Seasonal Baseline

- Same-day-last-week or last-year forecast
- Used as minimum performance threshold

### Prophet

- Automatic changepoint detection
- Weekly/monthly/yearly seasonality
- Hyperparameter tuning (optional)

### XGBoost

- Trained on engineered lag/rolling/calendar features
- Feature importance analysis
- Early stopping to prevent overfitting

## Walk-Forward Validation

Uses rolling-origin cross-validation instead of single train/test split:
- Configurable number of folds
- Configurable initial training window
- Configurable forecast horizon
- Configurable step size between folds

## Evaluation Metrics

- MAE, RMSE, MAPE, SMAPE, WAPE
- Bias (mean forecast error)
- Forecast Accuracy (100 - MAPE)

## Results

Results are tracked in MLflow and summarized in `reports/business_report.md`.

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Code Quality

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks
ruff check src/ tests/
black --check src/ tests/
```

## CI/CD

GitHub Actions automatically:
1. Runs Ruff linting
2. Checks Black formatting
3. Runs all Pytest tests
4. Verifies pipeline imports and configuration

## Business Insights

See the full business report at `reports/business_report.md` for detailed analysis.

## Limitations

- Intermittent demand inflates MAPE for many SKUs
- No external regressors (holidays, promotions, weather)
- Two-year dataset limits yearly seasonality estimation
- Cold-start problem for new products

## Future Improvements

- Hierarchical forecasting (category → SKU)
- Intermittent demand models (Croston's, TSB)
- Deep learning (LSTM, Transformer)
- External regressors
- SKU clustering for grouped modeling
- Automated retraining pipeline

## License

MIT

# Retail Demand Forecasting

Production-ready SKU-level demand forecasting system that improves inventory planning by outperforming naive seasonal forecasting baselines.

## Business Problem

Retail inventory planners currently estimate demand using:
- Last year's sales
- Manual adjustments
- Human intuition

This causes overstocking, understocking, lost sales, and high markdown costs.

This system builds forecasting models that consistently outperform the naive seasonal approach using rigorous historical backtesting.

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

## Dataset

Uses the **Corporación Favorita Grocery Sales Forecasting** dataset from Kaggle. See `data/raw/README.md` for download instructions.

Alternative supported datasets: M5 Forecasting, Rossmann Store Sales.

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
│   ├── raw/              # Raw dataset files
│   ├── interim/          # Intermediate data
│   └── processed/        # Feature-engineered data
├── models/               # Saved model artifacts
├── notebooks/            # EDA and analysis notebooks
├── reports/
│   ├── figures/          # Generated visualizations
│   └── business_report.md
├── src/
│   ├── data/             # Ingestion, validation, cleaning
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
- uv (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd retail-demand-forecasting

# Install with pip
pip install -r requirements.txt
pip install -e .

# Or install with uv
uv sync
```

### Download Dataset

```bash
python src/run_pipeline.py --download-data
```

If automatic download fails, see `data/raw/README.md` for manual instructions.

## Docker Usage

```bash
# Build and run the pipeline
docker compose up forecasting

# Start MLflow tracking server
docker compose --profile mlflow up

# Start Jupyter notebook server
docker compose --profile jupyter up

# Run everything
docker compose --profile "*" up
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
isort --check src/ tests/
```

## MLflow

Experiment tracking is automatically configured:

```bash
# Start MLflow UI
mlflow ui

# Or with Docker
docker compose --profile mlflow up
```

Access MLflow UI at `http://localhost:5001`

## EDA

The EDA notebook (`notebooks/eda.ipynb`) includes:

- Sales distribution analysis
- SKU popularity and store analysis
- Product category analysis
- Promotion and holiday effects
- Trend decomposition
- Weekly/monthly/yearly seasonality
- Year-over-year comparison
- Rolling averages
- Missing value analysis
- Correlation analysis and heatmaps

## Feature Engineering

The system generates:

- **Lag features**: 1, 7, 14, 28, 56 days
- **Rolling statistics**: mean, median, std, min, max over 7/14/28/56 day windows
- **EMA**: Exponential moving averages
- **Calendar**: day of week, month, quarter, year, weekend flags
- **Holiday/Promotion**: indicators and interactions
- **Price**: changes, rolling price features

## Forecasting Models

### Naive Seasonal Baseline

- Same-day-last-year or same-week-last-year forecast
- Used as minimum performance threshold

### Prophet

- Automatic changepoint detection
- Weekly/monthly/yearly seasonality
- Holiday and promotion regressors
- Hyperparameter tuning with Optuna

### XGBoost

- Trained on engineered lag/rolling features
- Feature importance analysis
- Hyperparameter tuning with Optuna
- Early stopping to prevent overfitting

## Walk-Forward Validation

Uses rolling-origin cross-validation instead of single train/test split:

- Configurable number of folds
- Configurable initial training window
- Configurable forecast horizon
- Configurable step size between folds
- Generates fold visualization diagrams

## Evaluation Metrics

- MAE, RMSE, MAPE, SMAPE, WAPE
- Bias (mean forecast error)
- Forecast Accuracy (100 - MAPE)
- Per-category and per-store breakdowns

## Results

Results are tracked in MLflow and summarized in `reports/business_report.md`.

## Business Insights

See the full business report at `reports/business_report.md` for detailed analysis including:

- Category-level performance analysis
- Inventory planning recommendations
- Promotion effectiveness analysis
- Forecast reliability assessment

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Code Quality

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks
ruff check src/ tests/
black --check src/ tests/
isort --check src/ tests/
```

## CI/CD

GitHub Actions automatically:
1. Runs Ruff linting
2. Checks Black formatting
3. Runs all Pytest tests
4. Verifies pipeline imports and configuration

## Limitations

- Requires 1+ years of historical data for reliable forecasts
- Cold-start problem for new products
- Does not account for supply chain disruptions
- Single SKU forecasts may have high uncertainty

## Future Improvements

- Deep learning (LSTM, Transformer architectures)
- Hierarchical forecasting
- Real-time forecast updates
- Causal inference for promotion optimization
- Automated model retraining

## License

MIT

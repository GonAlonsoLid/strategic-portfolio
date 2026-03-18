# Technology Stack

**Analysis Date:** 2026-03-18

## Languages

**Primary:**
- Python 3.11 - All core analysis, modeling, and backtesting code

## Runtime

**Environment:**
- Python 3.11.15
- Virtual environment (`venv/`) included in repository

**Package Manager:**
- pip
- Lockfile: implicit (requirements.txt maintained, no lock file generated)

## Frameworks

**Core Data & Analytics:**
- pandas >= 2.0 - Data manipulation, panel construction, time series handling
- numpy >= 1.24 - Numerical computations for feature engineering and portfolio calculations
- scikit-learn >= 1.3 - Logistic regression, random forest, and gradient boosting classifiers

**ML & Ensemble:**
- xgboost >= 2.0 - XGBoost classifier for advanced prediction models
- statsmodels >= 0.14 - OLS regression for Fama-French factor analysis

**Visualization:**
- matplotlib >= 3.7 - Generate figures for performance charts, drawdowns, factor loadings
- seaborn >= 0.12 - Statistical data visualization

**Utilities:**
- pyyaml >= 6.0 - YAML configuration parsing (`config/config.yaml`)
- openpyxl >= 3.1 - Excel file reading (S&P 500 events data)
- pyarrow >= 12.0 - Parquet serialization for efficient data storage

## Key Dependencies

**Critical:**
- pandas - Panel construction, time-series alignment, data preprocessing
- scikit-learn - Classification models (logistic, RF, GB)
- numpy - Numerical operations for feature and portfolio calculations

**Data I/O:**
- openpyxl - Parse S&P 500 joiner/leaver Excel file (`data/raw/SPX_index leavers & joiners_17-Feb-2026.xlsx`)
- pyarrow - Parquet storage for large datasets (daily prices, processed features)

**Statistical & ML:**
- statsmodels - Fama-French factor regression analysis
- xgboost - Optional; included for advanced classification but can be disabled if unavailable

## Configuration

**Environment:**
- Configuration driven via `config/config.yaml` (YAML)
- No environment variables required for core functionality
- Paths are relative to project root:
  - Raw data: `data/raw/`
  - Interim processing: `data/interim/`
  - Processed output: `data/processed/`
  - Results: `results/figures/`, `results/tables/`

**Build:**
- No build system (pure Python)
- No compilation step

## Platform Requirements

**Development:**
- macOS, Linux, or Windows with Python 3.11+
- Virtual environment recommended

**Production:**
- Any Python 3.11+ environment
- Local filesystem for data (no cloud dependencies required)
- Estimated memory: ~2-4 GB for full dataset processing (daily.csv is large)

---

*Stack analysis: 2026-03-18*

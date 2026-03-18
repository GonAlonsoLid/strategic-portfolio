# Architecture

**Analysis Date:** 2026-03-18

## Pattern Overview

**Overall:** Multi-stage quantitative research pipeline with layered separation of concerns

**Key Characteristics:**
- **Config-driven**: All parameters (event windows, feature lookbacks, model types, backtest settings) loaded from `config/config.yaml`
- **Sequential data processing**: Events → Daily panel → Features → Model predictions → Portfolio weights → Backtest results
- **Rolling time-series validation**: Time-aware train/test splits with no data leakage (train strictly before test)
- **Modular handlers**: Each domain (events, features, models, portfolio, backtesting) is a standalone module with clear input/output contracts

## Layers

**Data Loading Layer:**
- Purpose: Ingest raw Excel events and CSV price data; normalize and validate
- Location: `src/data/load_data.py`
- Contains: Event parser (Excel sheet "L&J" with ticker normalization), chunked price reader (handles large daily.csv via iterator pattern)
- Depends on: File system, pandas for I/O
- Used by: `src/data/preprocess_data.py` (builds daily panel)

**Data Preparation Layer:**
- Purpose: Build daily panel with S&P 500 membership tracking, return calculations, and market cap
- Location: `src/data/preprocess_data.py`
- Contains: Ticker-PERMNO bridge builder, daily panel assembly with index membership state machine (tracks ADD/DEL events)
- Depends on: Data loading layer, event dates to toggle membership status
- Used by: Feature engineering and event study layers

**Feature Engineering Layer:**
- Purpose: Compute rolling statistics (momentum, volatility, liquidity), rank market cap, and build forward-looking labels
- Location: `src/features/feature_engineering.py`, `src/features/rolling_features.py`
- Contains: Momentum windows [21, 63, 126, 252 trading days], volatility windows, skip-month momentum, quality proxy, joiner/leaver forward labels (63 days ahead)
- Depends on: Daily panel with returns and market cap
- Used by: Model training layer

**Event Study Layer:**
- Purpose: Compute abnormal returns, cumulative abnormal returns (CAR), aggregate by event type and relative day
- Location: `src/events/event_study.py`, `src/events/event_windows.py`
- Contains: Event window generation (pre/post event windows), market-relative abnormal return calculation, CAR aggregation by event type
- Depends on: Daily panel and event definitions
- Used by: Standalone analysis; produces plots and tables for research documentation

**Model Training Layer:**
- Purpose: Train binary classifiers (logistic, random forest, gradient boosting, XGBoost) with rolling time-series splits
- Location: `src/models/join_prediction.py`, `src/models/leave_prediction.py`, `src/models/model_utils.py`
- Contains: Model instantiation from config, rolling splits (5 years train, 1 year test), metric computation (ROC-AUC, precision, recall, F1)
- Depends on: Feature datasets with labels
- Used by: Portfolio construction layer (consumes probability predictions)

**Portfolio Construction Layer:**
- Purpose: Build target weights (long top joiners, short top leavers) with configurable weighting schemes
- Location: `src/portfolio/portfolio_construction.py`, `src/portfolio/weighting_schemes.py`
- Contains: Long/short builder, weighting schemes (equal, probability-weighted, risk-parity, rank), monthly rebalance dates
- Depends on: Model scores (join/leave probabilities), daily panel for volatility (risk-parity)
- Used by: Backtester

**Backtesting Layer:**
- Purpose: Simulate daily portfolio returns from target weights with turnover and transaction costs
- Location: `src/backtesting/backtester.py`, `src/backtesting/transaction_costs.py`
- Contains: Weight forward-fill (rebalance dates to daily), daily return calculation, turnover tracking, cost deduction
- Depends on: Target weights and daily panel with returns
- Used by: Evaluation layer

**Evaluation & Attribution Layer:**
- Purpose: Compute performance metrics (Sharpe, max DD, VaR, skewness), subperiod analysis, Fama-French factor attribution
- Location: `src/evaluation/performance_metrics.py`, `src/evaluation/factor_analysis.py`
- Contains: Metric compilers (requires return time series), optional factor regression
- Depends on: Portfolio returns and optional factor data
- Used by: Output generation (tables, plots)

**Utility Layer:**
- Purpose: Shared configuration loading, plotting, and utility functions
- Location: `src/utils/config_loader.py`, `src/utils/plotting.py`
- Contains: YAML config parser, nested dict traversal, plotting functions (CAR curves, cumulative returns, drawdowns)
- Depends on: File system (config.yaml), matplotlib
- Used by: All layers

## Data Flow

**Full Pipeline (run_backtest.py):**

1. **Load Config** → `config/config.yaml` specifies all paths and parameters
2. **Build Daily Panel** (if not cached):
   - Load events from Excel: normalizes tickers, maps event_type (Joiner→ADD, Leaver→DEL), sets event dates
   - Load prices chunked from CSV: normalizes column names, filters dates
   - Build Ticker-PERMNO bridge: maps normalized ticker to permno (latest per permno)
   - Track S&P 500 membership: initial constituents (all firms in 1995 not in ADD list), then apply DEL/ADD events by effective_date
   - Output: `data/interim/daily_panel.parquet` with columns [date, permno, ticker, ret, market_cap, volume, is_sp500]

3. **Feature Engineering**:
   - Compute rolling momentum (4 windows) + skip-month momentum
   - Compute rolling volatility (3 windows)
   - Compute liquidity (turnover, volume averages)
   - Compute abnormal performance (cross-sectional return rank)
   - Compute quality proxy (low-volatility rank)
   - Build market cap rank and percentile (cross-sectional daily)
   - Build forward labels: label_join = 1 if firm joins S&P 500 in next 63 trading days (else 0), label_leave = opposite
   - Drop rows with insufficient history (<252 days)
   - Output: `data/processed/features_join.parquet`, `data/processed/features_leave.parquet`

4. **Train Models**:
   - Rolling time-series splits: for each year from 2000 onward, train on prior 5 years, test on next 1 year (no overlap)
   - For each model type (logistic, RF, GB, XGBoost):
     - Standardize features if logistic (scale=True), else raw
     - Fit on X_train, predict probabilities on X_test
     - Compute metrics: ROC-AUC, precision, recall, F1 (sklearn)
   - Save model scores per fold to `data/processed/join_scores.parquet` and `data/processed/leave_scores.parquet`
   - Output: columns [date, permno, p_join_<model>, p_leave_<model>]

5. **Portfolio Construction**:
   - Identify rebalance dates: first trading day of each month
   - For each rebalance date:
     - Rank firms by join probability; select top decile (10%)
     - Rank firms by leave probability; select top decile (10%)
     - Assign weights based on scheme (equal, probability, risk-parity, rank)
     - Enforce gross_exposure=2.0 (100% long + 100% short), net_exposure=0.0
   - Output: `target_weights` with columns [date, permno, weight] (weights sum to ±1 per date)

6. **Backtest**:
   - For each trading day in panel:
     - Get current weight from prior rebalance (forward-fill)
     - Calculate portfolio return: sum(weight[i] * ret[i])
     - Calculate turnover: sum(|weight_today[i] - weight_yesterday[i]|)
     - Deduct transaction costs: cost_bps / 10000 * turnover
     - Track gross_exposure (sum of absolute weights), net_exposure (sum of weights)
   - Output: `result` dict with [returns, gross_returns, turnover, transaction_costs, gross_exposure, net_exposure]

7. **Evaluation & Output**:
   - Compute metrics: annualized return, Sharpe, max drawdown, VaR (95%, 99%), skewness
   - Compute subperiod metrics (rolling 3-year windows)
   - Plot cumulative returns, drawdown curves
   - Optional: run Fama-French factor regression (if factors provided)
   - Save to `results/tables/` (CSV) and `results/figures/` (PNG)

**State Management:**
- Time-indexed state: all data keyed by (date, permno) pairs, sorted by date within groups for rolling calculations
- Caching: intermediate datasets saved as parquet (with CSV fallback) to avoid recomputation
- Date alignment: all operations normalize dates to midnight (dt.normalize()) for consistency
- Memory efficiency: large price CSV read in chunks (500k rows default) to avoid OOM

## Key Abstractions

**Event (ADD/DEL):**
- Purpose: Represents S&P 500 index membership change
- Examples: `src/data/load_data.py` (load_events), `src/data/preprocess_data.py` (build_daily_panel)
- Pattern: DataFrame row with [event_type, ticker, ticker_raw, event_date, announcement_date, effective_date]; effective_date triggers membership toggle

**Daily Panel:**
- Purpose: Time-indexed snapshot of each firm's daily return, market cap, volume, and index membership
- Examples: `src/data/preprocess_data.py` → `data/interim/daily_panel.parquet`
- Pattern: MultiIndex-like (date, permno) with columns [ret, market_cap, volume, is_sp500, market_ret, ...]

**Feature Matrix:**
- Purpose: Roll up daily panel to feature vectors per (date, permno) with forward label
- Examples: `src/features/feature_engineering.py` → `data/processed/features_join.parquet`
- Pattern: Rows are unique (date, permno) pairs; columns are rolling/cross-sectional statistics plus binary label (label_join or label_leave)

**Model Scores:**
- Purpose: Probability predictions per (date, permno) from trained classifier
- Examples: `src/models/join_prediction.py` → `data/processed/join_scores.parquet`
- Pattern: Columns [date, permno, p_join_logistic, p_join_random_forest, p_join_gradient_boosting, p_join_xgboost]

**Portfolio Weights:**
- Purpose: Target allocation (positive for long, negative for short) per (date, permno)
- Examples: `src/portfolio/portfolio_construction.py` → weights passed to Backtester
- Pattern: Updated monthly; weights sum to gross_exposure (typically 2.0) with net close to 0

**Backtest Result:**
- Purpose: Daily portfolio performance metrics
- Examples: returned by `Backtester.run_backtest()`
- Pattern: Dict with keys [returns, gross_returns, turnover, transaction_costs, gross_exposure, net_exposure]; all Series indexed by date

## Entry Points

**run_backtest.py:**
- Location: `scripts/run_backtest.py`
- Triggers: Manual execution (`python scripts/run_backtest.py`); orchestrates full pipeline
- Responsibilities: Load config, conditionally build each stage (check cache), execute pipeline stages sequentially, produce final results

**run_event_study.py:**
- Location: `scripts/run_event_study.py`
- Triggers: Standalone analysis execution
- Responsibilities: Load panel and events, compute CAR, aggregate by event type, generate plots and tables (pre/post announcement, inclusion reversal)

**train_models.py:**
- Location: `scripts/train_models.py`
- Triggers: Manual retraining of models
- Responsibilities: Load features, execute model training with rolling splits, save scores and metrics

**Notebooks:**
- `notebooks/01_event_study.ipynb`: Interactive exploration of abnormal returns and CAR
- `notebooks/02_feature_exploration.ipynb`: Exploratory analysis of feature distributions, forward label rates

## Error Handling

**Strategy:** Permissive data validation with informative warnings; missing data filled or dropped based on context

**Patterns:**
- File existence checks: `if not path.exists(): raise FileNotFoundError(...)` in data loaders
- Column mapping: Flexible Excel/CSV column matching (handle spacing, variant names like "Status" vs "event_type")
- NaN handling: Feature rows dropped if key features missing (`dropna(subset=key_feats)`); model inputs filled with 0 (`fillna(0)`)
- Iterator patterns: Chunked CSV reading yields only non-empty chunks, skips ahead if date filters exclude entire chunk
- Model compatibility: Check if XGBoost installed before attempting import; fall back to other models if missing
- Cache fallback: Check parquet, fall back to CSV; skip intermediate stage if output exists

## Cross-Cutting Concerns

**Logging:** Uses print() statements for milestone reporting (e.g., "Building daily panel", "Training models"); no centralized logger

**Validation:**
- Event data: Normalize event_type (Joiner→ADD, Leaver→DEL), strip ticker exchange suffixes (.N, .OQ, ^B26)
- Price data: Ensure date column exists and is datetime; rename columns to standard names
- Feature data: Drop rows with insufficient history; NaN handling varies by use (fill 0 for models, drop for key features)

**Authentication:** No authentication needed; all data files are local (Excel, CSV)

**Configuration:** Centralized in `config/config.yaml`; all paths and parameters loaded at runtime; no hardcoded values in code


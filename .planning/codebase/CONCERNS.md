# Codebase Concerns

**Analysis Date:** 2026-03-18

## Tech Debt

**Index Membership Logic - Incomplete Historical Data:**
- Issue: S&P 500 index membership is inferred from data gaps rather than from authoritative historical membership lists. Initial constituents (before 1995) are approximated by taking stocks that appear in 1995 data and subtracting those that joined in 1995.
- Files: `src/data/preprocess_data.py` (lines 45-53, 107-112)
- Impact: Survivorship bias in initial constituent identification. Stocks delisted or that never appeared in data are excluded, inflating apparent "initial" membership. This could bias event study results for early-period joiners and leavers.
- Fix approach: Either obtain authoritative S&P 500 constituent history (e.g., from Bloomberg, Refinitiv) or explicitly document the survivorship bias assumption. Consider validating against known large-cap constituents from 1995.

**Ticker-PERMNO Bridge Uses Limited Sample:**
- Issue: The ticker-PERMNO bridge is built from only the first 2M rows of daily.csv (line 163 in `load_data.py`). If the file is sorted by date, this samples only early periods and may miss tickers that appear later in history.
- Files: `src/data/load_data.py` (lines 148-169)
- Impact: Firms that joined S&P 500 after the sampled period may not be matched to their PERMNO, causing missing data in the panel for recent additions.
- Fix approach: Either sample uniformly across the entire file or sort/deduplicate by date before sampling. Validate bridge coverage against actual event dates.

**Date Column Duplication and Type Coercion:**
- Issue: `build_daily_panel()` checks for "date" column that may already exist, then renames DlyCalDt to "date" again (lines 73-77). Type coercion uses `pd.to_datetime(..., errors="coerce")` which silently converts unparseable dates to NaT.
- Files: `src/data/preprocess_data.py` (lines 73-77)
- Impact: Silent data loss if date parsing fails; duplicate column logic is fragile if called multiple times or if data loaders change.
- Fix approach: Validate date parsing strictly (raise on errors instead of coerce). Extract date handling to a single, well-tested function.

**Market Return Calculation Inconsistency:**
- Issue: Market return is calculated in multiple places: in `run_backtest.py` (line 42) and in `feature_engineering.py` (line 78), computed as simple mean of individual stock returns per date.
- Files: `scripts/run_backtest.py` (line 42), `src/features/feature_engineering.py` (line 78)
- Impact: If the panel is subsampled or filtered differently between calls, market returns will diverge, leading to inconsistent feature values. Equal-weight market return is not aligned with S&P 500 market-cap weighting.
- Fix approach: Compute and cache market returns once during panel building. Use VW (value-weighted) returns if available in raw data; otherwise, document that equal-weight is used.

**Rolling Feature Calculations - NaN Propagation:**
- Issue: Rolling features use `min_periods=1` or small thresholds (e.g., lines 64, 85, 88 in `rolling_features.py`), which can produce features from very short windows when data is sparse. This inflates feature values and may introduce look-ahead bias if data gaps align with events.
- Files: `src/features/rolling_features.py` (lines 64, 85, 88)
- Impact: Features may be unreliable during low-liquidity periods or for newly added stocks. Models trained on sparse-window features may not generalize.
- Fix approach: Set stricter `min_periods` thresholds (e.g., 50% of window size) and drop rows with insufficient history. Document minimum data requirements per feature.

**Quality Proxy Calculation - Division by Volatility:**
- Issue: Quality proxy divides return by volatility (line 129 in `rolling_features.py`), creating unbounded values when volatility is very low. Although `.replace(0, np.nan)` and `.replace(np.inf, np.nan)` are applied, extreme low-vol periods can still produce very large values.
- Files: `src/features/rolling_features.py` (lines 113-130)
- Impact: Outlier feature values can dominate model training, especially for illiquid stocks. May reduce model generalization.
- Fix approach: Winsorize or clip the quality proxy to a reasonable range (e.g., ±3 standard deviations). Explicitly handle the case where volatility is exactly zero.

## Known Bugs

**Portfolio Construction - Fallback Model Selection Bug:**
- Issue: In `run_backtest.py` (lines 94-99), if the requested model (e.g., "random_forest") is not found in join_scores, code falls back to any p_join_* column. However, if multiple models are present, there's no control over which one is selected (list comprehension just takes the first).
- Files: `scripts/run_backtest.py` (lines 94-99), `src/portfolio/portfolio_construction.py` (lines 40-45)
- Impact: Unintended model selection in edge cases; user may expect a specific model but get a different one silently.
- Workaround: Explicitly check which models are available before running backtest; ensure requested model is in scores.
- Fix approach: Raise an informative error if the requested model is not found, rather than silently falling back.

**Backtester Weight Alignment Issue:**
- Issue: In `backtester.py` (lines 67-68), daily returns are extracted from the panel and aligned with weights using `union` and `reindex`. If permno appears in panel but not in weights, it gets zero weight. This is correct but may mask missing data issues.
- Files: `src/backtesting/backtester.py` (lines 67-75)
- Impact: Silent partial position liquidation if a stock drops out of the score set due to NaN features or model failures.
- Workaround: Validate score coverage against panel before backtesting.
- Fix approach: Add logging to track "dropped" stocks per date and issue warnings if coverage drops below threshold.

**Rolling Window Label Construction - Lookahead Detection:**
- Issue: `build_joiner_label()` uses `shift(-1)` then `rolling().max()` (line 39 in `feature_engineering.py`). The `shift(-1)` moves data forward, and then rolling window looks ahead. Although `shift(1)` is applied later to align, the logic is convoluted and error-prone.
- Files: `src/features/feature_engineering.py` (lines 27-57)
- Impact: Risk of undetected lookahead bias if the logic is refactored.
- Workaround: Validate labels by comparing against known event dates (manual spot check).
- Fix approach: Rewrite to use explicit forward-looking windows with clear date semantics.

## Security Considerations

**Configuration File Loading - No Validation:**
- Risk: `config_loader.py` uses `yaml.safe_load()` which is safe from arbitrary code execution, but there's no schema validation. Users could accidentally introduce typos in config keys that silently get ignored (see `get_section()` line 36).
- Files: `src/utils/config_loader.py` (lines 8-25)
- Current mitigation: YAML parsing is sandboxed.
- Recommendations: Add a config schema validator (e.g., using pydantic or voluptuous). Warn on unknown keys. Consider adding required-key checks.

**Data File Paths - No Symlink or Path Traversal Check:**
- Risk: File paths are loaded from config and used directly with `Path(path)`. A malicious config could specify `../../sensitive_data.csv` or symlinks.
- Files: `src/data/load_data.py` (lines 45-51), `src/data/preprocess_data.py` (lines 32-35)
- Current mitigation: Code runs in local development environment; no untrusted user input expected.
- Recommendations: Validate that resolved paths stay within expected data directories. Use `Path.resolve().relative_to(base)` to enforce boundaries.

**Missing Data - Silent NaN Handling:**
- Risk: NaN values are filled with 0 or ffill without validation. In feature matrix (line 81 in `join_prediction.py`), missing features are filled with 0, which may violate model assumptions.
- Files: `src/models/join_prediction.py` (line 81), `src/features/feature_engineering.py` (line 103)
- Impact: Models may learn spurious patterns from zero-filled missing data.
- Recommendations: Explicitly track missing data fractions per feature. Use more sophisticated imputation (e.g., forward-fill + backward-fill) or drop rows with excessive missingness.

## Performance Bottlenecks

**Chunked Daily Price Loading - Memory vs. Speed Tradeoff:**
- Problem: `load_prices_chunked()` reads 500k rows per chunk (default), then concatenates into a list that is periodically re-concatenated (line 96 in `preprocess_data.py`). For a 1.7B row CSV, this requires ~3400 iterations.
- Files: `src/data/preprocess_data.py` (lines 10-100)
- Cause: Naive chunking without streaming aggregation or database.
- Improvement path: Use Parquet streaming, DuckDB, or Polars lazy evaluation to avoid re-concatenation. Pre-sort daily.csv by date/permno to enable efficient panel building.

**Panel Filtering - O(n) Linear Search:**
- Problem: `build_daily_panel()` filters panel by date range using boolean indexing on every chunk. If applied repeatedly (e.g., in tests), this scales linearly with data size.
- Files: `src/data/preprocess_data.py` (lines 66-97)
- Improvement path: Partition data by date range during initial load. Use parquet partitioning if output is parquetized.

**Feature Engineering - Redundant Groupby Operations:**
- Problem: Each feature-building function (momentum, volatility, liquidity) calls `groupby(permno)` independently. For a 500-stock x 10k-day panel, this is 4-5 separate groupby operations.
- Files: `src/features/rolling_features.py` (lines 34-50, 53-66, 69-90, 93-110)
- Improvement path: Batch multiple features in a single groupby-transform loop using custom aggregation functions.

**Backtester Alignment - Redundant Reindex:**
- Problem: `run_backtest()` reindexes weights and returns multiple times per day (line 68 in `backtester.py`). For 250 trading days x 500 stocks, this is 125k reindex calls.
- Files: `src/backtesting/backtester.py` (lines 29-88)
- Improvement path: Pre-align weights and returns to a common index once before the loop.

## Fragile Areas

**Event Study Window Calculations - Edge Cases:**
- Files: `src/events/event_study.py`, `src/events/event_windows.py`
- Why fragile: Event study windows (pre_window, post_window) are defined in trading days. If data has gaps (e.g., weekends, holidays), the window alignment may be off by 1-2 days. There's no explicit check for "trading day" vs. "calendar day".
- Safe modification: Always validate window alignment by checking that the pre-window start date is exactly `pre_window` trading days before the event. Flag misaligned events.
- Test coverage gaps: No tests for events that occur on Mondays (after 2-day weekend) or around holidays.

**Portfolio Construction Weighting Schemes - Unbounded Weights:**
- Files: `src/portfolio/weighting_schemes.py`
- Why fragile: Equal weight and rank weight functions allocate based on top decile cutoff, but don't enforce leverage or net-exposure constraints at the individual-stock level. A stock with very high probability could get allocated more than intended.
- Safe modification: Add explicit clipping for gross and net exposure per stock.
- Test coverage gaps: No tests for edge cases (e.g., all stocks have identical probabilities, top_decile=0.05 but only 3 stocks).

**Model Rolling Window Splits - Insufficient Test Data:**
- Files: `src/models/model_utils.py` (lines 10-30)
- Why fragile: `make_rolling_splits()` checks `len(train_idx) > 0` and `len(test_idx) > 0` but doesn't verify that test sets have sufficient positive labels. If a test year has zero joiners/leavers, metrics will be NaN or skewed.
- Safe modification: Add checks for minimum positive/negative sample sizes in test set. Skip folds with imbalanced classes.
- Test coverage gaps: No validation of class balance per fold.

**Pipeline Fallback Logic - Silent Cascades:**
- Files: `scripts/run_backtest.py` (lines 32-81)
- Why fragile: The main script checks for parquet then CSV files, and if both are missing, rebuilds data. If rebuild fails silently (e.g., due to file permissions), the script continues with stale cached data.
- Safe modification: Explicitly raise errors if critical data is missing or rebuild fails. Add cleanup logic to remove partial artifacts.
- Test coverage gaps: No tests for missing/corrupted intermediate files.

## Scaling Limits

**Daily Price CSV Size - File I/O Bottleneck:**
- Current capacity: daily.csv is 1.7B rows. Reading in 500k-row chunks requires ~3400 iterations.
- Limit: Beyond 5B rows, chunked reading becomes impractical on standard hardware.
- Scaling path: Migrate to Parquet or DuckDB for columnar storage and better compression. Implement distributed processing (e.g., Spark, Dask) if adding more securities or longer history.

**In-Memory Feature Matrix - RAM Requirement:**
- Current capacity: ~5M rows x 25 features x 8 bytes = 1GB for a single feature matrix.
- Limit: If expanding to full Russell 2000 or adding intraday data, memory could exceed 16GB.
- Scaling path: Use sparse matrices or lazy evaluation (Polars, DuckDB). Implement feature-on-demand loading.

**Model Training - Combinatorial Explosion:**
- Current capacity: 4 model types x ~23 rolling windows = 92 model variants.
- Limit: If adding hyperparameter grids, this explodes to thousands of models.
- Scaling path: Use automated hyperparameter tuning (e.g., Optuna) rather than manual config. Cache trained models to avoid retraining.

**Backtesting - Time Complexity:**
- Current capacity: 250 trading days x 500 stocks = 125k daily portfolio evaluations.
- Limit: Scaling to daily rebalancing (250 rebalances) or intraday simulation becomes expensive.
- Scaling path: Use vectorized portfolio operations (e.g., with NumPy broadcasting) instead of per-day loops.

## Dependencies at Risk

**XGBoost Import Failure - Graceful Fallback:**
- Risk: XGBoost is imported with a try-except (lines 13-17 in `join_prediction.py`), so missing XGBoost is handled. However, if XGBoost is present but incompatible (e.g., version mismatch), the failure is silent.
- Files: `src/models/join_prediction.py` (lines 13-17)
- Impact: User may not realize XGBoost is disabled if there's a subtle import error.
- Migration plan: Add explicit version pinning in requirements.txt. Consider removing XGBoost if not essential; logistic + RF + GB cover most use cases.

**Pandas & NumPy Version Sensitivity:**
- Risk: Code uses modern pandas APIs (e.g., `dt.normalize()`) that may not exist in older versions. Requirements specify `pandas>=2.0` and `numpy>=1.24`.
- Files: Project-wide usage of pandas API.
- Recommendation: Either strictly pin versions or add explicit compatibility checks at startup.

**statsmodels for Factor Regression:**
- Risk: `statsmodels` is used only in factor regression, which is optional. If not installed, `run_factor_regression()` will fail at import time.
- Files: `src/evaluation/factor_analysis.py` (line 7)
- Current mitigation: Factor regression is optional; backtest works without it.
- Recommendation: Add try-except around factor regression in the main pipeline.

## Missing Critical Features

**No Validation of Input Data Schemas:**
- Problem: Code assumes events Excel and daily CSV have expected columns, but there's no schema validation. If a user provides incorrectly formatted data, errors occur deep in the pipeline.
- Blocks: Reproducible error messages for data preparation issues.
- Fix approach: Implement a data validation layer with Cerberus or pydantic to check schemas upfront.

**No Logging or Debugging Hooks:**
- Problem: The pipeline has minimal logging beyond print() statements. If something fails midway (e.g., data corruption), it's hard to diagnose.
- Blocks: Production monitoring and debugging.
- Fix approach: Integrate Python logging module with configurable levels (DEBUG, INFO, WARNING, ERROR). Log key metrics (row counts, NaN percentages) at each pipeline stage.

**No Checkpointing or Resume Capability:**
- Problem: If the pipeline fails (e.g., out of memory), you must restart from scratch. Intermediate outputs are overwritten.
- Blocks: Efficient iteration on large datasets.
- Fix approach: Implement checkpoint save/load (e.g., Mlflow, DVC) so that failed runs can resume from the last successful stage.

**No Model Versioning or Experiment Tracking:**
- Problem: Trained models are saved as "join_scores.parquet" without version info. If you run the pipeline twice with different configs, the second run overwrites the first.
- Blocks: Reproducibility and A/B testing.
- Fix approach: Add experiment tracking (e.g., Mlflow, Weights & Biases) with git commit hashing and timestamp-based versioning.

## Test Coverage Gaps

**No Unit Tests for Data Loading:**
- What's not tested: Edge cases in ticker-PERMNO matching, date column type coercion, event parsing from Excel.
- Files: `src/data/load_data.py`, `src/data/preprocess_data.py`
- Risk: Subtle data issues (e.g., off-by-one date alignments, missing tickers) are discovered only in final results, not during development.
- Priority: High

**No Tests for Feature Engineering Edge Cases:**
- What's not tested: Behavior when a stock has < 252 days of history, when volatility is zero, when turnover is infinite.
- Files: `src/features/rolling_features.py`, `src/features/feature_engineering.py`
- Risk: NaN propagation or extreme values can destabilize model training.
- Priority: High

**No Tests for Model Selection and Cross-Validation:**
- What's not tested: Rolling window splits with imbalanced folds, fallback model selection, score alignment.
- Files: `src/models/join_prediction.py`, `src/models/leave_prediction.py`, `src/models/model_utils.py`
- Risk: Model performance metrics may be misleading if folds are constructed incorrectly.
- Priority: High

**No Backtesting Validation Tests:**
- What's not tested: Turnover calculations, transaction cost application, weight alignment across dates.
- Files: `src/backtesting/backtester.py`
- Risk: Backtest results may overstate performance if weights are misaligned or costs are miscalculated.
- Priority: Critical

**No Integration Tests for Full Pipeline:**
- What's not tested: End-to-end execution with real data; intermediate file formats (parquet vs. CSV fallback).
- Files: `scripts/run_backtest.py`, `scripts/run_event_study.py`, `scripts/train_models.py`
- Risk: Pipeline may fail on user's machine due to environment or data differences.
- Priority: Medium

---

*Concerns audit: 2026-03-18*

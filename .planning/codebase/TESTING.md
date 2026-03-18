# Testing Patterns

**Analysis Date:** 2026-03-18

## Test Framework

**Runner:**
- Not detected - No pytest.ini, setup.cfg, tox.ini, or conftest.py found
- No test files located in `tests/` directory or with `test_*.py` / `*_test.py` naming convention
- No test dependencies in `requirements.txt` (file contains only data science packages: pandas, numpy, scikit-learn, xgboost, statsmodels, matplotlib, seaborn, pyyaml, openpyxl, pyarrow)

**Assertion Library:**
- Not applicable - testing framework not in use

**Run Commands:**
- Not established - no Makefile, tox.ini, or pytest config
- Manual testing via notebook: `notebooks/01_event_study.ipynb`, `notebooks/02_feature_exploration.ipynb`

## Test File Organization

**Location:**
- No dedicated test directory exists
- Tests are performed via Jupyter notebooks in `notebooks/` directory

**Naming:**
- Notebooks use descriptive names: `01_event_study.ipynb`, `02_feature_exploration.ipynb`
- No separate unit or integration test files

**Structure:**
```
notebooks/
├── 01_event_study.ipynb       # Event study analysis and validation
├── 02_feature_exploration.ipynb  # Feature engineering exploration
└── [Analysis notebooks]
```

## Test Structure

**Manual Testing Approach:**
- Code validation occurs in Jupyter notebooks
- Scripts run end-to-end pipeline: `scripts/run_backtest.py`, `scripts/train_models.py`, `scripts/run_event_study.py`
- Pipeline scripts validate by:
  1. Loading data from config paths
  2. Executing all stages sequentially
  3. Writing outputs to `results/` and `data/` directories
  4. Printing status messages to console

**Example validation from `scripts/run_backtest.py` (lines 34-36):**
```python
if not panel_path.exists() and not panel_csv.exists():
    print("Building daily panel (max_chunks=20 for testing; set max_chunks=None for full run)...")
    build_daily_panel(config=cfg, max_chunks=20)
```

**Error Detection Strategy:**
- FileNotFoundError raised immediately if required files missing (e.g., `src/data/load_data.py:50-51`)
- ValueError raised if required columns missing (e.g., `src/data/load_data.py:65-70`)
- Pandas assertions via `.dropna(subset=)` for data quality checks
- Empty DataFrame checks: `if windows.empty: return {...}` (in `src/events/event_study.py:79-80`)

## Mocking

**Framework:**
- No mocking framework in use
- Data passed through actual file I/O and pandas operations

**Patterns:**
- Functions accept optional `config` parameter for test-friendly configuration paths (seen throughout, e.g., `src/data/load_data.py:27-29`)
- Default parameter fallback allows overriding: `path: str | Path | None = None` with `if path is None: path = default` pattern
- `max_chunks` parameter in `build_daily_panel()` enables testing on large CSV by limiting chunk processing (in `src/data/preprocess_data.py:18`)

**What to Mock:**
- External file paths can be provided as function parameters instead of mocking
- Config loading can be overridden with `config` dict parameter
- No network calls or external API integrations to mock

**What NOT to Mock:**
- Pandas DataFrame operations - rely on actual data transformations
- File I/O operations - use actual temporary files if testing
- NumPy/sklearn operations - test with real numeric data

## Fixtures and Factories

**Test Data:**
- No fixtures module detected
- Test data validation occurs in notebooks via direct data loading

**Location:**
- Sample notebooks in `notebooks/` directory serve as validation fixtures
- Raw data located in `data/raw/` (SPX index leavers & joiners Excel, daily prices CSV)
- Processed outputs in `data/interim/` and `data/processed/` serve as intermediate fixtures

**Example data shape assumptions from `src/models/model_utils.py`:**
```python
def make_rolling_splits(
    df: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
    min_start_year: int = 2000,
    date_col: str = "date",
) -> List[Tuple[pd.Index, pd.Index]]:
    """Splits (train_index, test_index) with train strictly before test. No overlap."""
```

## Coverage

**Requirements:**
- No enforced coverage requirements detected
- No `.coveragerc` or coverage configuration files

**View Coverage:**
- Not applicable - coverage tooling not in use
- Manual validation via notebook execution and script output inspection

## Test Types

**Unit Tests:**
- Not implemented formally
- Functions designed for composability allow manual unit testing in notebooks
- Example testable units: `load_events()`, `build_feature_panel()`, `compute_performance_metrics()`

**Integration Tests:**
- Pipeline scripts serve as de facto integration tests
- `scripts/run_backtest.py` (lines 21-125) exercises entire pipeline:
  1. Load events and prices
  2. Build daily panel
  3. Construct features
  4. Train models
  5. Generate portfolio weights
  6. Run backtest
  7. Compute metrics
  8. Plot results

**E2E Tests:**
- Pipeline validation via manual notebook execution
- Scripts produce outputs written to `results/figures/` and `results/tables/`
- Backtest script prints metrics: `print("Metrics:", metrics)` (line 125)

## Common Patterns

**Data Validation Pattern:**
```python
# From src/data/preprocess_data.py - checking for required columns
if "PERMNO" not in ch.columns:
    continue
if "date" not in ch.columns and "DlyCalDt" in ch.columns:
    ch["date"] = pd.to_datetime(ch["DlyCalDt"], errors="coerce").dt.normalize()
```

**Handling Missing Data:**
```python
# From src/models/model_utils.py
res["roc_auc"] = roc_auc_score(y_test, proba) if y_test.nunique() >= 2 else np.nan
res["precision"] = precision_score(y_test, pred, zero_division=0)
```

**Pipeline Idempotency:**
```python
# From scripts/run_backtest.py
if not (features_join_path.exists() or features_join_csv.exists()):
    print("Building features and labels...")
    features_join, features_leave = build_feature_panel(panel, config=cfg)
    save_feature_datasets(features_join, features_leave, config=cfg)
if features_join_path.exists():
    features_join = pd.read_parquet(features_join_path)
```

**Function-level Testing via Notebooks:**
Example from `notebooks/01_event_study.ipynb` (inferred from module dependencies):
- Load events with `load_events()`
- Verify columns present: `assert "event_type" in events.columns`
- Load prices with `load_prices_chunked()`
- Compute abnormal returns with `compute_abnormal_returns()`
- Visualize with `plot_car()`

## Recommended Testing Structure

**For future implementation:**

**Unit Tests Location:** `tests/unit/`
- `test_data_loaders.py` - test `load_events()`, `load_prices_chunked()`, `build_ticker_permno_bridge()`
- `test_feature_engineering.py` - test `build_feature_panel()`, `build_joiner_label()`, `build_leaver_label()`
- `test_models.py` - test `make_rolling_splits()`, `train_and_evaluate()`, `precision_at_k()`
- `test_portfolio.py` - test `build_long_short_portfolio()`, weighting schemes
- `test_metrics.py` - test `compute_performance_metrics()`, `compute_subperiod_metrics()`

**Integration Tests Location:** `tests/integration/`
- `test_pipeline_e2e.py` - run full backtest pipeline with sample data

**Fixture Location:** `tests/fixtures/`
- `conftest.py` - pytest fixtures for sample DataFrames
- Sample data CSVs for testing

**Run Command (when implemented):**
```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest --cov=src tests/
```

---

*Testing analysis: 2026-03-18*

# Coding Conventions

**Analysis Date:** 2026-03-18

## Naming Patterns

**Files:**
- Lowercase with underscores: `load_data.py`, `feature_engineering.py`, `backtester.py`
- Module purpose is clear from filename: `portfolio_construction.py`, `performance_metrics.py`, `transaction_costs.py`
- Private/internal files use leading underscore pattern (e.g., `_project_root()` in `load_data.py`)

**Functions:**
- Snake_case: `load_events()`, `build_feature_panel()`, `compute_performance_metrics()`
- Verb-first for actions: `build_`, `compute_`, `run_`, `load_`, `add_`
- Helper functions use underscore prefix: `_project_root()`, `_mom_skip()` (in `src/features/rolling_features.py`)
- Transformation functions explicitly indicate they modify: `add_momentum_features()`, `add_volatility_features()` (in `src/features/rolling_features.py`)

**Variables:**
- Snake_case throughout: `forward_days`, `permno_col`, `date_col`, `market_cap_rank`
- Abbreviated column names in parameters: `permno_col`, `ticker_col`, `ret_col`, `cap_col` (in `src/features/feature_engineering.py`)
- Intermediate results use descriptive names: `w_long`, `w_short`, `rebalance_dates` (in `src/portfolio/portfolio_construction.py`)

**Types:**
- Union syntax: `str | None` (Python 3.10+ style) in function signatures (seen in `src/data/load_data.py`, `src/features/feature_engineering.py`)
- Type hints use `pd.DataFrame`, `pd.Series`, `pd.Index`, `np.ndarray`
- Dict and List types: `Dict[str, float]`, `List[Tuple[pd.Index, pd.Index]]` from `typing` module

## Code Style

**Formatting:**
- No configuration files detected (no `.pylintrc`, `pyproject.toml`, `.flake8`)
- Consistent 4-space indentation observed
- Line length appears to respect ~100 character limit in most cases
- Docstrings use triple-quoted format (see `src/data/load_data.py`, `src/models/model_utils.py`)

**Linting:**
- No linting configuration detected
- Code follows PEP 8 conventions by observation

## Import Organization

**Order:**
1. Standard library: `sys`, `pathlib.Path`, `typing` modules
2. Third-party scientific: `pandas`, `numpy`, `sklearn`, `statsmodels`, `xgboost`
3. Project relative imports: `from src.utils.config_loader import load_config`

**Examples from codebase:**
- `src/data/load_data.py`: `pathlib.Path`, `typing`, then `pandas`
- `src/backtesting/backtester.py`: `pathlib.Path`, `typing`, then `pandas`, `numpy`, then `src.backtesting`
- `scripts/run_backtest.py`: `sys`, `pathlib.Path`, then project imports after sys.path insertion

**Path Aliases:**
- No path aliases configured (no `jsconfig.json` or TypeScript path mapping)
- Relative imports use explicit `src.` prefix: `from src.utils.config_loader import load_config`
- Scripts insert project root into sys.path: `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`

## Error Handling

**Patterns:**
- Explicit existence checks before operations: `if not path.exists(): raise FileNotFoundError(...)` (in `src/data/load_data.py:50-51`)
- Try-except for import fallback: `try: df.to_parquet(...) except ImportError: df.to_csv(...)` (in `src/features/feature_engineering.py:122-125`)
- Assertion-free; uses explicit conditionals with error messages
- Config fallback pattern: `if config is None: config = load_config()` (seen throughout, e.g., `src/data/load_data.py:18-23`)
- Graceful degradation: Functions return empty DataFrames or NaN when data insufficient (e.g., `src/evaluation/performance_metrics.py:16-21` returns dict of NaN values)

**No global try-catch blocks observed:**
- Functions fail fast with informative errors rather than silent failures
- Missing data handled with pandas `.fillna()`, `.dropna(subset=)`, `.reindex()`

## Logging

**Framework:** Console prints only

**Patterns:**
- Script execution uses `print()` statements: `print("Building daily panel...")` (in `scripts/run_backtest.py:35`)
- No `logging` module imported in any file
- Status messages at key pipeline stages in main scripts
- No debug logging; prints only for user-facing status

## Comments

**When to Comment:**
- Column mapping explanations: e.g., "Status = Joiner | Leaver, Code = ticker" (in `src/data/load_data.py:7-8`)
- Complex logic and domain-specific terms: "Max in next forward_days: shift(-1) then rolling(...).max()" (in `src/features/feature_engineering.py:38-39`)
- Data assumptions: "treat 'initial' as everyone who appears in the first year" (in `src/data/preprocess_data.py:51-52`)
- Algorithm notes: "Rolling standard deviation of returns" (in `src/features/rolling_features.py:57`)

**JSDoc/TSDoc:**
- Use PEP 257 style docstrings: triple-quoted, descriptive
- Parameters documented in docstring: `Args: path: Path to Excel file...` (in `src/data/load_data.py:38-44`)
- Return type and description: `Returns: DataFrame with columns event_type, ticker...` (in `src/data/load_data.py:42-43`)
- No type hints in docstrings (types are in function signature)

**Example docstring pattern from `src/models/model_utils.py`:**
```python
def train_and_evaluate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale: bool = False,
) -> Dict[str, float]:
    """Fit model and return ROC-AUC, precision, recall, F1. Optionally scale X."""
```

## Function Design

**Size:**
- Typically 10-40 lines per function
- Data transformation functions are concise: `build_joiner_label()` is 10 lines (in `src/features/feature_engineering.py:27-41`)
- Complex orchestration functions are longer: `build_daily_panel()` is ~180 lines (in `src/data/preprocess_data.py`)
- Pipeline scripts (main functions) can be 100+ lines

**Parameters:**
- Explicit keyword-only parameters after `*`: `def load_prices_chunked(..., chunksize: int = 500_000, usecols: list[str] | None = None, date_min: str | None = None, date_max: str | None = None)` (in `src/data/load_data.py:102-109`)
- Config as optional first parameter: `config: dict | None = None` with fallback to `load_config()`
- Column name parameters as `*_col` suffix: `date_col: str = "date"`, `permno_col: str = "permno"` (throughout codebase)
- Underscore before keyword args when grouping related options: `build_feature_panel(..., *, min_history_days: int = 252)` (in `src/features/feature_engineering.py:60-65`)

**Return Values:**
- Single DataFrame or Series for data transformations
- Dict for metric results: `Dict[str, float]` (in `src/models/model_utils.py:40`)
- Tuple for paired outputs: `tuple[pd.DataFrame, pd.DataFrame]` (in `src/features/feature_engineering.py:65`)
- List of tuples for multiple splits: `List[Tuple[pd.Index, pd.Index]]` (in `src/models/model_utils.py:16`)

## Module Design

**Exports:**
- No `__all__` definitions observed
- Modules export all public functions and classes
- Main orchestration imported explicitly: `from src.data.load_data import load_events, build_ticker_permno_bridge`

**Barrel Files:**
- `__init__.py` files exist but are empty (checked `src/__init__.py`, `src/data/__init__.py`)
- No re-exports; each module imported directly by full path

**Example import pattern from `scripts/run_backtest.py`:**
```python
from src.data.load_data import load_events, load_config_paths, build_ticker_permno_bridge
from src.features.feature_engineering import build_feature_panel, save_feature_datasets
from src.models.join_prediction import run_join_prediction
from src.portfolio.portfolio_construction import build_long_short_portfolio
```

## Default Parameter Patterns

**Observed defaults:**
- Nested config access with `.get()` and fallback: `cfg.get("paths", {}).get("raw_events", "data/raw/SPX_index...")` (in `src/data/load_data.py:48`)
- Or extracted via `get_section()`: `get_section(cfg, "event_study", "pre_window", default=60)` (in `src/events/event_study.py:73`)
- Numeric defaults for windows/periods: `train_years: int = 5`, `test_years: int = 1` (in `src/models/model_utils.py:12-13`)
- Financial-domain constants: `transaction_cost_bps: float = 10`, `top_decile: float = 0.10` (in `src/backtesting/backtester.py:21`, `src/portfolio/portfolio_construction.py:19`)

---

*Convention analysis: 2026-03-18*

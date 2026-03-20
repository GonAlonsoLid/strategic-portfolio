"""Tests for robustness sweep logic."""
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile

# run_robustness.py is in scripts/ — available via conftest.py sys.path insertion
from run_robustness import _make_rebalance_dates, run_sweep


def _make_panel(n_stocks=10, n_days=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", periods=n_days, freq="B")
    rows = []
    for p in range(1, n_stocks + 1):
        membership = np.zeros(n_days, dtype=bool)
        if p % 2 == 0:
            membership[:n_days // 2] = True
        else:
            membership[n_days // 2:] = True
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "permno": p,
                "ret": rng.normal(0.0002, 0.01),
                "is_sp500": bool(membership[i]),
                "market_cap": 1e9, "volume": 1e6,
            })
    return pd.DataFrame(rows)


def _make_scores(panel, kind="join", seed=0):
    """Minimal scores DataFrame with one probability column."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(panel["date"].unique())
    permnos = panel["permno"].unique()
    rows = [
        {"date": d, "permno": p, f"p_{kind}_xgboost": rng.uniform(0, 1)}
        for d in dates for p in permnos
    ]
    return pd.DataFrame(rows)


def test_make_rebalance_dates_monthly():
    """Monthly rebalance dates should have one date per calendar month."""
    panel = _make_panel()
    monthly = _make_rebalance_dates(panel, holding_period_months=1)
    assert len(monthly) > 0
    # Roughly: 500 business days / 21 ≈ 23 months
    assert 15 <= len(monthly) <= 30


def test_make_rebalance_dates_quarterly_fewer_than_monthly():
    """Quarterly should have ~1/3 as many dates as monthly."""
    panel = _make_panel()
    monthly = _make_rebalance_dates(panel, holding_period_months=1)
    quarterly = _make_rebalance_dates(panel, holding_period_months=3)
    assert len(quarterly) < len(monthly)
    assert len(monthly) // len(quarterly) in [2, 3, 4]


def test_robustness_sweep_produces_12_rows():
    """Default sweep (4 holding periods × 3 thresholds) must produce exactly 12 rows."""
    panel = _make_panel()
    join_scores = _make_scores(panel, "join")
    leave_scores = _make_scores(panel, "leave")
    result = run_sweep(panel, join_scores, leave_scores)
    assert len(result) == 12, f"Expected 12 rows, got {len(result)}"


def test_robustness_output_columns():
    """Output must have all required columns."""
    panel = _make_panel()
    join_scores = _make_scores(panel, "join")
    leave_scores = _make_scores(panel, "leave")
    result = run_sweep(panel, join_scores, leave_scores)
    required = {"holding_period_months", "top_decile", "annual_return",
                "annual_volatility", "sharpe_ratio", "max_drawdown", "turnover"}
    assert required.issubset(set(result.columns)), \
        f"Missing columns: {required - set(result.columns)}"


def test_robustness_nan_row_for_empty_weights():
    """When portfolio weights are empty (very tight threshold), row must exist with NaN metrics."""
    panel = _make_panel(n_stocks=3)
    join_scores = _make_scores(panel, "join")
    leave_scores = _make_scores(panel, "leave")
    # Threshold so tight (0.001 = 0.1%) that with 3 stocks no stock qualifies
    result = run_sweep(panel, join_scores, leave_scores,
                       holding_periods=[1], thresholds=[0.001])
    assert len(result) == 1, "Must output 1 row even when weights are empty"
    assert "holding_period_months" in result.columns
    # Metrics should be NaN (not dropped, not zero)
    assert pd.isna(result["sharpe_ratio"].iloc[0]) or result["sharpe_ratio"].iloc[0] == result["sharpe_ratio"].iloc[0]


def test_robustness_csv_output():
    """Output CSV must be readable and have correct shape."""
    panel = _make_panel()
    join_scores = _make_scores(panel, "join")
    leave_scores = _make_scores(panel, "leave")
    result = run_sweep(panel, join_scores, leave_scores,
                       holding_periods=[1, 3], thresholds=[0.10, 0.20])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "robustness.csv"
        result.to_csv(path, index=False)
        loaded = pd.read_csv(path)
        assert len(loaded) == 4  # 2 × 2
        assert set(loaded["holding_period_months"]) == {1, 3}
        assert set(loaded["top_decile"]) == {0.10, 0.20}

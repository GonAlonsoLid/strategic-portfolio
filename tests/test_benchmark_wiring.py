"""Integration tests for omniscient benchmark wiring."""
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.portfolio.portfolio_construction import build_perfect_foresight_portfolio
from src.backtesting.backtester import Backtester
from src.evaluation.performance_metrics import compute_performance_metrics


def _make_panel(n_stocks=10, n_days=400, seed=42):
    """Minimal daily panel with is_sp500 that has real joiners and leavers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")
    rows = []
    for p in range(1, n_stocks + 1):
        membership = np.zeros(n_days, dtype=bool)
        if p % 2 == 0:
            membership[:n_days // 2] = True   # member then leaves
        else:
            membership[n_days // 2:] = True   # non-member then joins
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "permno": p,
                "ret": rng.normal(0.0002, 0.01),
                "is_sp500": bool(membership[i]),
                "market_cap": 1e9, "volume": 1e6,
            })
    return pd.DataFrame(rows)


def test_build_perfect_foresight_portfolio_nonempty():
    """Panel with joiners and leavers must produce non-empty benchmark weights."""
    panel = _make_panel()
    weights = build_perfect_foresight_portfolio(panel, None, None, forward_days=63, top_decile=0.5)
    assert not weights.empty, "Expected non-empty weights from panel with joiners and leavers"
    assert set(weights.columns) >= {"date", "permno", "weight"}


def test_benchmark_backtest_produces_returns_series():
    """Benchmark backtest must return a non-empty daily returns Series."""
    panel = _make_panel()
    weights = build_perfect_foresight_portfolio(panel, None, None, forward_days=63, top_decile=0.5)
    if weights.empty:
        pytest.skip("No benchmark weights — fixture too small")
    bt = Backtester(panel)
    result = bt.run_backtest(weights)
    ret = result["returns"]
    assert isinstance(ret, pd.Series)
    assert len(ret) > 0


def test_strategy_comparison_table_columns_match_metrics_keys():
    """Comparison table column names must match compute_performance_metrics return keys."""
    panel = _make_panel()
    ret = pd.Series(
        np.random.default_rng(0).normal(0.0005, 0.01, 252),
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )
    metrics = compute_performance_metrics(ret)
    # Build comparison table exactly as run_backtest.py will
    comp = pd.DataFrame([
        {"strategy": "Predictive", **metrics},
        {"strategy": "Omniscient", **metrics},
    ])
    # All metric keys must be present as columns
    for key in metrics.keys():
        assert key in comp.columns, f"Column '{key}' missing from comparison table"
    assert "strategy" in comp.columns
    assert len(comp) == 2


def test_strategy_comparison_table_csv_roundtrip():
    """Comparison table must survive a CSV write/read cycle without data loss."""
    ret = pd.Series(
        np.random.default_rng(0).normal(0.0005, 0.01, 252),
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )
    m1 = compute_performance_metrics(ret)
    m2 = {k: v * 1.5 for k, v in m1.items()}  # simulate omniscient doing better
    comp = pd.DataFrame([
        {"strategy": "Predictive", **m1},
        {"strategy": "Omniscient", **m2},
    ])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "strategy_comparison.csv"
        comp.to_csv(path, index=False)
        loaded = pd.read_csv(path)
        assert len(loaded) == 2
        assert set(loaded["strategy"]) == {"Predictive", "Omniscient"}
        assert abs(loaded.loc[loaded["strategy"] == "Omniscient", "annual_return"].iloc[0]
                   - m2["annual_return"]) < 1e-9

"""Tests for plot_strategy_comparison."""
import matplotlib
matplotlib.use("Agg")  # headless: must be set before any other matplotlib import

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.utils.plotting import plot_strategy_comparison


def _make_returns(n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, n), index=idx)


def test_plot_strategy_comparison_creates_file():
    """Should create a non-empty PNG file with two overlaid series."""
    series = {
        "Predictive": _make_returns(seed=0),
        "Omniscient": _make_returns(seed=1),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "comparison.png"
        plot_strategy_comparison(series, save_path=out)
        assert out.exists(), "Plot file was not created"
        assert out.stat().st_size > 0, "Plot file is empty"


def test_plot_strategy_comparison_single_series():
    """Should work with a single series without raising."""
    series = {"Predictive": _make_returns(seed=0)}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "single.png"
        plot_strategy_comparison(series, save_path=out)
        assert out.exists()


def test_plot_strategy_comparison_no_save():
    """Should not raise when save_path is None."""
    series = {"A": _make_returns(seed=0)}
    plot_strategy_comparison(series, save_path=None)  # just no crash

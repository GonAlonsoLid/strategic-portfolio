"""Temporary: regenerate comparison chart without the duplicate Quantile line."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.config_loader import load_config
from src.data.load_data import load_config_paths
from src.portfolio.portfolio_construction import (
    build_long_short_portfolio,
    build_perfect_foresight_portfolio,
    build_topn_portfolio,
    build_composite_portfolio,
    build_volscaled_portfolio,
)
from src.backtesting.backtester import Backtester
from src.utils.plotting import plot_strategy_comparison

cfg = load_config()
paths = load_config_paths(cfg)
base = Path(__file__).resolve().parent.parent
out_fig = base / paths.get("results_figures", "results/figures")

print("Loading data...")
interim = base / paths.get("interim", "data/interim")
processed = base / paths.get("processed", "data/processed")
panel = pd.read_parquet(interim / "daily_panel.parquet")
join_scores = pd.read_parquet(processed / "join_scores.parquet")
leave_scores = pd.read_parquet(processed / "leave_scores.parquet")
if "market_ret" not in panel.columns:
    panel["market_ret"] = panel.groupby("date")["ret"].transform("mean")

p_cols = [c for c in join_scores.columns if c.startswith("p_join_")]
model_name = p_cols[0].replace("p_join_", "") if p_cols else "xgboost"
signal_start = min(join_scores["date"].min(), leave_scores["date"].min())
bt = Backtester(panel, transaction_cost_bps=10)


def _get_returns(weights):
    result = bt.run_backtest(weights)
    ret = result["returns"]
    return ret[ret.index >= signal_start]


key_series = {}

# Omniscient
print("  Omniscient...")
pf_w = build_perfect_foresight_portfolio(panel, None, None, forward_days=63, top_decile=0.10)
key_series["Omniscient"] = _get_returns(pf_w)

# Best: Predictive (quantile) — only once, no duplicate
print("  Predictive (quantile)...")
q_w = build_long_short_portfolio(join_scores, leave_scores, panel, config=cfg,
                                  top_decile=0.10, weighting="equal", model_name=model_name)
key_series["Best: Predictive (quantile)"] = _get_returns(q_w)

# Best top-N: Top-5 (probability)
print("  Top-5 (probability)...")
t5_w = build_topn_portfolio(join_scores, leave_scores, panel,
                             n_long=5, n_short=5, weighting="probability",
                             gross_exposure=2.0, model_name=model_name)
key_series["Best top-N: Top-5 (probability)"] = _get_returns(t5_w)

# Best composite: Composite-5 (a=0.25)
print("  Composite-5 (a=0.25)...")
c5_w = build_composite_portfolio(join_scores, leave_scores, panel,
                                  n_long=5, n_short=5, alpha=0.25, beta=0.25,
                                  weighting="equal", gross_exposure=2.0, model_name=model_name)
key_series["Best composite: Composite-5 (a=0.25)"] = _get_returns(c5_w)

# Best asymmetric: Asym-5L/20S
print("  Asym-5L/20S...")
a_w = build_topn_portfolio(join_scores, leave_scores, panel,
                            n_long=5, n_short=20, weighting="equal",
                            gross_exposure=2.0, model_name=model_name)
key_series["Best asymmetric: Asym-5L/20S"] = _get_returns(a_w)

# Best vol-scaled: VolScaled-10 (g=0.5)
print("  VolScaled-10 (g=0.5)...")
v_w = build_volscaled_portfolio(join_scores, leave_scores, panel,
                                 n_long=10, n_short=10, gamma=0.5,
                                 gross_exposure=2.0, model_name=model_name)
key_series["Best vol-scaled: VolScaled-10 (g=0.5)"] = _get_returns(v_w)

# Align and plot
common_start = max(s.index.min() for s in key_series.values())
aligned = {k: v[v.index >= common_start] for k, v in key_series.items()}
plot_strategy_comparison(aligned, save_path=out_fig / "cumulative_returns_comparison.png")
print(f"\nSaved to {out_fig / 'cumulative_returns_comparison.png'}")

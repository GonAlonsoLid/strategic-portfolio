"""Robustness sweep: backtest over holding-period x probability-threshold grid.

Uses threshold-based portfolio construction (absolute probability gating).

Requires pre-computed files on disk (run run_backtest.py first):
  data/interim/daily_panel.parquet
  data/processed/join_scores.parquet
  data/processed/leave_scores.parquet

Usage:
  python scripts/run_robustness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.utils.config_loader import load_config
from src.data.load_data import load_config_paths
from src.portfolio.portfolio_construction import build_topn_portfolio
from src.backtesting.backtester import Backtester
from src.evaluation.performance_metrics import compute_performance_metrics
from src.utils.plotting import plot_robustness_heatmap


_DEFAULT_HOLDING_PERIODS = [1, 3, 6, 12]   # months
_DEFAULT_N_POSITIONS = [5, 10, 20, 30, 50]


def _make_rebalance_dates(panel: pd.DataFrame, holding_period_months: int) -> pd.Index:
    """Return first trading day of each Nth calendar month."""
    ud = pd.DatetimeIndex(pd.to_datetime(panel["date"].unique())).sort_values()
    df_u = pd.DataFrame({"date": ud})
    df_u["ym"] = df_u["date"].dt.to_period("M")
    monthly_dates = df_u.groupby("ym")["date"].first().values
    return monthly_dates[::holding_period_months]


def run_sweep(
    panel: pd.DataFrame,
    join_scores: pd.DataFrame,
    leave_scores: pd.DataFrame,
    *,
    holding_periods: list | None = None,
    n_positions_list: list | None = None,
    config: dict | None = None,
    transaction_cost_bps: float = 10,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """Run holding-period x top-N grid with top-N portfolios."""
    holding_periods = holding_periods or _DEFAULT_HOLDING_PERIODS
    n_positions_list = n_positions_list or _DEFAULT_N_POSITIONS
    bt = Backtester(panel, transaction_cost_bps=transaction_cost_bps)
    rows = []
    for hp in holding_periods:
        rebalance_dates = _make_rebalance_dates(panel, hp)
        for n in n_positions_list:
            weights = build_topn_portfolio(
                join_scores, leave_scores, panel,
                rebalance_dates=rebalance_dates,
                n_long=n, n_short=n,
                weighting="equal",
                model_name=model_name,
            )
            if weights.empty:
                rows.append({
                    "holding_period_months": hp, "n_positions": n,
                    "annual_return": np.nan, "annual_volatility": np.nan,
                    "sharpe_ratio": np.nan, "sortino_ratio": np.nan,
                    "max_drawdown": np.nan, "turnover": np.nan,
                    "n_positions_avg": 0,
                })
                continue
            result = bt.run_backtest(weights)
            ret = result["returns"]
            m = compute_performance_metrics(ret)
            to = float(result["turnover"].mean())
            n_pos = weights.groupby("date").size().mean()
            rows.append({
                "holding_period_months": hp,
                "n_positions": n,
                "annual_return": m["annual_return"],
                "annual_volatility": m["annual_volatility"],
                "sharpe_ratio": m["sharpe_ratio"],
                "sortino_ratio": m["sortino_ratio"],
                "max_drawdown": m["max_drawdown"],
                "turnover": to,
                "n_positions_avg": round(n_pos, 1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent
    interim = base / paths.get("interim", "data/interim")
    processed = base / paths.get("processed", "data/processed")
    out_tab = base / paths.get("results_tables", "results/tables")
    out_fig = base / paths.get("results_figures", "results/figures")
    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    # Pre-flight checks
    required = {
        "daily_panel": interim / "daily_panel.parquet",
        "join_scores": processed / "join_scores.parquet",
        "leave_scores": processed / "leave_scores.parquet",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        print("ERROR: Required files not found. Run run_backtest.py first.")
        for m in missing:
            print(f"  Missing: {m}")
        sys.exit(1)

    print("Loading pre-computed data...")
    panel = pd.read_parquet(required["daily_panel"])
    join_scores = pd.read_parquet(required["join_scores"])
    leave_scores = pd.read_parquet(required["leave_scores"])
    if "market_ret" not in panel.columns:
        panel["market_ret"] = panel.groupby("date")["ret"].transform("mean")

    # Auto-detect model name
    p_cols = [c for c in join_scores.columns if c.startswith("p_join_")]
    model_name = p_cols[0].replace("p_join_", "") if p_cols else "xgboost"

    n_combos = len(_DEFAULT_HOLDING_PERIODS) * len(_DEFAULT_N_POSITIONS)
    print(f"Running robustness sweep: {len(_DEFAULT_HOLDING_PERIODS)} holding periods x "
          f"{len(_DEFAULT_N_POSITIONS)} top-N = {n_combos} combinations...")

    result = run_sweep(panel, join_scores, leave_scores, config=cfg, model_name=model_name)

    out_path = out_tab / "robustness_holding_periods.csv"
    result.to_csv(out_path, index=False)
    print(f"\nRobustness sweep complete. Saved to {out_path}")
    print(result.to_string(index=False))

    # Generate heatmaps for key metrics
    for metric in ["sharpe_ratio", "annual_return", "max_drawdown"]:
        plot_robustness_heatmap(
            result, metric=metric,
            row_col="holding_period_months", col_col="n_positions",
            title=f"Robustness: {metric} (holding period x top-N)",
            save_path=out_fig / f"robustness_heatmap_{metric}.png",
        )
    print("Heatmaps saved to results/figures/")


if __name__ == "__main__":
    main()

"""Robustness sweep: backtest over holding-period × threshold combinations.

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
from src.portfolio.portfolio_construction import build_long_short_portfolio
from src.backtesting.backtester import Backtester
from src.evaluation.performance_metrics import compute_performance_metrics


_DEFAULT_HOLDING_PERIODS = [1, 3, 6, 12]   # months
_DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20]


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
    thresholds: list | None = None,
    config: dict | None = None,
    transaction_cost_bps: float = 10,
) -> pd.DataFrame:
    """Run holding-period × threshold grid. Always returns exactly len(hp) × len(thr) rows."""
    holding_periods = holding_periods or _DEFAULT_HOLDING_PERIODS
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    bt = Backtester(panel, transaction_cost_bps=transaction_cost_bps)
    rows = []
    for hp in holding_periods:
        rebalance_dates = _make_rebalance_dates(panel, hp)
        for thr in thresholds:
            weights = build_long_short_portfolio(
                join_scores, leave_scores, panel,
                config=config,
                rebalance_dates=rebalance_dates,
                top_decile=thr,
            )
            if weights.empty:
                rows.append({
                    "holding_period_months": hp, "top_decile": thr,
                    "annual_return": np.nan, "annual_volatility": np.nan,
                    "sharpe_ratio": np.nan, "max_drawdown": np.nan, "turnover": np.nan,
                })
                continue
            result = bt.run_backtest(weights)
            ret = result["returns"]
            m = compute_performance_metrics(ret)
            # turnover is a Series from the backtester; take mean daily turnover
            to = float(result["turnover"].mean())
            rows.append({
                "holding_period_months": hp,
                "top_decile": thr,
                "annual_return": m["annual_return"],
                "annual_volatility": m["annual_volatility"],
                "sharpe_ratio": m["sharpe_ratio"],
                "max_drawdown": m["max_drawdown"],
                "turnover": to,
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent
    interim = base / paths.get("interim", "data/interim")
    processed = base / paths.get("processed", "data/processed")
    out_tab = base / paths.get("results_tables", "results/tables")
    out_tab.mkdir(parents=True, exist_ok=True)

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

    n_combos = len(_DEFAULT_HOLDING_PERIODS) * len(_DEFAULT_THRESHOLDS)
    print(f"Running robustness sweep: {len(_DEFAULT_HOLDING_PERIODS)} holding periods × "
          f"{len(_DEFAULT_THRESHOLDS)} thresholds = {n_combos} combinations...")

    result = run_sweep(panel, join_scores, leave_scores, config=cfg)

    out_path = out_tab / "robustness_holding_periods.csv"
    result.to_csv(out_path, index=False)
    print(f"\nRobustness sweep complete. Saved to {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

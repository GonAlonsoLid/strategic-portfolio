"""End-to-end pipeline: load data, features, models, portfolios, backtest, results.

Generates all deliverables required by Project 1.pdf:
- Omniscient benchmark (perfect foresight)
- Predictive strategies (quantile-based and threshold-based)
- Performance comparison tables
- All figures (cumulative returns, drawdowns, turnover, exposure)
- Factor attribution
- Subperiod analysis
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.utils.config_loader import load_config
from src.data.load_data import load_config_paths
from src.portfolio.portfolio_construction import (
    build_long_short_portfolio,
    build_perfect_foresight_portfolio,
    build_threshold_portfolio,
    build_topn_portfolio,
)
from src.backtesting.backtester import Backtester
from src.evaluation.performance_metrics import (
    compute_performance_metrics,
    compute_drawdowns,
    compute_subperiod_metrics,
)
from src.evaluation.factor_analysis import run_factor_regression, load_factors
from src.utils.plotting import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_factor_loadings,
    plot_strategy_comparison,
    plot_turnover,
    plot_exposure,
)


def _load_data(cfg: dict):
    """Load panel and scores from disk."""
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent
    interim = base / paths.get("interim", "data/interim")
    processed = base / paths.get("processed", "data/processed")

    panel = pd.read_parquet(interim / "daily_panel.parquet")
    if "market_ret" not in panel.columns:
        panel["market_ret"] = panel.groupby("date")["ret"].transform("mean")

    join_scores = pd.read_parquet(processed / "join_scores.parquet")
    leave_scores = pd.read_parquet(processed / "leave_scores.parquet")

    return panel, join_scores, leave_scores


def _detect_model_name(join_scores: pd.DataFrame) -> str:
    """Auto-detect model name from score columns."""
    cols = [c for c in join_scores.columns if c.startswith("p_join_")]
    return cols[0].replace("p_join_", "") if cols else "xgboost"


def _run_strategy(bt: Backtester, weights: pd.DataFrame, name: str) -> dict:
    """Run backtest and compute full metrics for a strategy."""
    if weights.empty:
        print(f"  WARNING: {name} portfolio is empty — skipping.")
        return None
    result = bt.run_backtest(weights)
    metrics = compute_performance_metrics(result["returns"])
    metrics["avg_daily_turnover"] = float(result["turnover"].mean())
    metrics["avg_gross_exposure"] = float(result["gross_exposure"].mean())
    metrics["avg_net_exposure"] = float(result["net_exposure"].mean())
    metrics["avg_transaction_cost"] = float(result["transaction_costs"].mean())
    n_pos = (weights["weight"] > 0).sum()
    n_neg = (weights["weight"] < 0).sum()
    n_rebal = weights["date"].nunique()
    metrics["n_long_positions_total"] = int(n_pos)
    metrics["n_short_positions_total"] = int(n_neg)
    metrics["n_rebalance_dates"] = int(n_rebal)
    metrics["avg_long_per_rebal"] = round(n_pos / max(n_rebal, 1), 1)
    metrics["avg_short_per_rebal"] = round(n_neg / max(n_rebal, 1), 1)
    return {"name": name, "metrics": metrics, "result": result}


def main() -> None:
    cfg = load_config()
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent
    out_fig = base / paths.get("results_figures", "results/figures")
    out_tab = base / paths.get("results_tables", "results/tables")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    print("Loading pre-computed data...")
    panel, join_scores, leave_scores = _load_data(cfg)
    model_name = _detect_model_name(join_scores)
    print(f"  Model: {model_name}")
    print(f"  Panel: {len(panel):,} rows, {panel['permno'].nunique():,} stocks")
    print(f"  Score dates: {join_scores['date'].nunique():,}")

    backtest_cfg = cfg.get("backtest", {})
    bt = Backtester(panel, transaction_cost_bps=backtest_cfg.get("transaction_cost_bps", 10))

    # ── 1. Omniscient benchmark ──────────────────────────────────────────
    print("\n1. Building omniscient (perfect foresight) benchmark...")
    pf_weights = build_perfect_foresight_portfolio(
        panel, None, None, forward_days=63, top_decile=0.10
    )
    omni = _run_strategy(bt, pf_weights, "Omniscient")

    # ── 2. Predictive strategy: quantile-based (original) ────────────────
    print("\n2. Building quantile-based predictive strategy...")
    q_weights = build_long_short_portfolio(
        join_scores, leave_scores, panel, config=cfg,
        top_decile=0.10, weighting="equal", model_name=model_name,
    )
    quantile_strat = _run_strategy(bt, q_weights, "Predictive (quantile)")

    # ── 3. Predictive strategy: top-N (concentrated, calibration-independent) ──
    print("\n3. Building top-N predictive strategies...")
    topn_strategies = []
    for n in [5, 10, 20, 30, 50]:
        for w_scheme in ["equal", "probability"]:
            name = f"Top-{n} ({w_scheme})"
            print(f"  {name}...")
            tw = build_topn_portfolio(
                join_scores, leave_scores, panel,
                n_long=n, n_short=n,
                weighting=w_scheme, gross_exposure=2.0, model_name=model_name,
            )
            s = _run_strategy(bt, tw, name)
            if s:
                topn_strategies.append(s)

    # ── 4. Collect all strategies ────────────────────────────────────────
    all_strategies = []
    if omni:
        all_strategies.append(omni)
    if quantile_strat:
        all_strategies.append(quantile_strat)
    all_strategies.extend(topn_strategies)

    # ── 5. Strategy comparison table ─────────────────────────────────────
    print("\n4. Generating comparison tables...")
    comp_rows = [{"strategy": s["name"], **s["metrics"]} for s in all_strategies]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_tab / "strategy_comparison.csv", index=False)
    print(comp_df[["strategy", "annual_return", "sharpe_ratio", "max_drawdown",
                    "avg_long_per_rebal", "avg_short_per_rebal"]].to_string(index=False))

    # Save individual summaries
    if omni:
        pd.DataFrame([omni["metrics"]]).to_csv(out_tab / "backtest_summary_omniscient.csv", index=False)

    # Find best top-N strategy by Sharpe
    best_thr = max(topn_strategies, key=lambda s: s["metrics"].get("sharpe_ratio", -999)) if topn_strategies else None
    if best_thr:
        pd.DataFrame([best_thr["metrics"]]).to_csv(out_tab / "backtest_summary_predictive.csv", index=False)
        print(f"\n  Best strategy: {best_thr['name']}")
        print(f"    Sharpe: {best_thr['metrics']['sharpe_ratio']:.3f}")
        print(f"    Annual return: {best_thr['metrics']['annual_return']:.4f}")

    # ── 6. Figures ───────────────────────────────────────────────────────
    print("\n5. Generating figures...")

    # Cumulative returns comparison (all key strategies)
    key_series = {}
    if omni:
        key_series["Omniscient"] = omni["result"]["returns"]
    if quantile_strat:
        key_series["Predictive (quantile)"] = quantile_strat["result"]["returns"]
    if best_thr:
        key_series[best_thr["name"]] = best_thr["result"]["returns"]

    if key_series:
        common_start = max(s.index.min() for s in key_series.values())
        aligned = {k: v[v.index >= common_start] for k, v in key_series.items()}
        plot_strategy_comparison(aligned, save_path=out_fig / "cumulative_returns_comparison.png")

    # Individual plots for best threshold strategy
    if best_thr:
        ret_best = best_thr["result"]["returns"]
        plot_cumulative_returns(ret_best, title=f"Cumulative returns: {best_thr['name']}",
                                save_path=out_fig / "cumulative_returns.png")
        plot_drawdowns(ret_best, title=f"Drawdown: {best_thr['name']}",
                       save_path=out_fig / "drawdown.png")
        plot_turnover(best_thr["result"]["turnover"],
                      title=f"Turnover: {best_thr['name']}",
                      save_path=out_fig / "turnover.png")
        plot_exposure(best_thr["result"]["gross_exposure"],
                      best_thr["result"]["net_exposure"],
                      title=f"Exposure: {best_thr['name']}",
                      save_path=out_fig / "exposure.png")

    # Drawdowns for omniscient
    if omni:
        plot_drawdowns(omni["result"]["returns"], title="Drawdown: Omniscient",
                       save_path=out_fig / "drawdown_omniscient.png")

    # ── 7. Subperiod analysis ────────────────────────────────────────────
    print("\n6. Subperiod analysis...")
    if best_thr:
        subperiod = compute_subperiod_metrics(best_thr["result"]["returns"], window_years=3.0)
        if subperiod:
            sub_df = pd.DataFrame(subperiod)
            sub_sampled = sub_df.iloc[::63].tail(20)
            sub_sampled.to_csv(out_tab / "backtest_subperiod_3y.csv", index=False)

    # ── 8. Factor attribution ────────────────────────────────────────────
    print("\n7. Factor attribution...")
    factors = load_factors(config=cfg)
    if factors is not None and best_thr:
        fac_res = run_factor_regression(best_thr["result"]["returns"], factors)
        pd.DataFrame([fac_res]).to_csv(out_tab / "factor_loadings_predictive.csv", index=False)
        if fac_res.get("betas"):
            plot_factor_loadings(pd.Series(fac_res["betas"]),
                                 save_path=out_fig / "factor_loadings.png")
    else:
        print("  No factor data available (raw_factors not configured).")

    print("\nDone. Results saved to results/figures/ and results/tables/")


if __name__ == "__main__":
    main()

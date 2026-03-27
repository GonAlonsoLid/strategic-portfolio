"""End-to-end pipeline: load data, features, models, portfolios, backtest, results.

Generates all deliverables required by Project 1.pdf:
- Omniscient benchmark (perfect foresight)
- Predictive strategies (quantile, top-N, composite, vol-scaled, momentum-filtered, asymmetric)
- Performance comparison tables
- All figures (cumulative returns, drawdowns, turnover, exposure, annual returns, rolling metrics)
- Factor attribution
- Subperiod analysis
- Daily return series (CSV) for report generation
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
    build_composite_portfolio,
    build_volscaled_portfolio,
    build_momentum_filtered_portfolio,
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
    plot_annual_returns,
    plot_rolling_metrics,
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


def _run_strategy(bt: Backtester, weights: pd.DataFrame, name: str,
                   start_date: pd.Timestamp | None = None) -> dict:
    """Run backtest and compute full metrics for a strategy."""
    if weights.empty:
        print(f"  WARNING: {name} portfolio is empty -- skipping.")
        return None
    result = bt.run_backtest(weights)
    if start_date is not None:
        for key in ("returns", "turnover", "gross_exposure", "net_exposure",
                     "transaction_costs"):
            if key in result and hasattr(result[key], "index"):
                result[key] = result[key][result[key].index >= start_date]
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

    # Determine when the predictive model first produces signals.  All
    # strategy returns (including the omniscient benchmark) are trimmed to
    # start from this date so that every strategy is evaluated over an
    # identical period — the one where the predictive model can actually
    # trade.  Without this trim the omniscient and panel-derived strategies
    # show a long flat segment (1995-2004) that inflates the backtest
    # horizon while contributing zero information about strategy quality.
    signal_start = min(join_scores["date"].min(), leave_scores["date"].min())
    print(f"  Signal start date: {signal_start:%Y-%m-%d}")

    backtest_cfg = cfg.get("backtest", {})
    bt = Backtester(panel, transaction_cost_bps=backtest_cfg.get("transaction_cost_bps", 10))

    # ── 1. Omniscient benchmark ──────────────────────────────────────────
    print("\n1. Building omniscient (perfect foresight) benchmark...")
    pf_weights = build_perfect_foresight_portfolio(
        panel, None, None, forward_days=63, top_decile=0.10
    )
    omni = _run_strategy(bt, pf_weights, "Omniscient", start_date=signal_start)

    # ── 2. Predictive strategy: quantile-based (original) ────────────────
    print("\n2. Building quantile-based predictive strategy...")
    q_weights = build_long_short_portfolio(
        join_scores, leave_scores, panel, config=cfg,
        top_decile=0.10, weighting="equal", model_name=model_name,
    )
    quantile_strat = _run_strategy(bt, q_weights, "Predictive (quantile)", start_date=signal_start)

    # ── 3. Top-N strategies (symmetric) ──────────────────────────────────
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
            s = _run_strategy(bt, tw, name, start_date=signal_start)
            if s:
                topn_strategies.append(s)

    # ── 4. NEW: Composite score strategies ───────────────────────────────
    print("\n4. Building composite score strategies...")
    composite_strategies = []
    for n in [5, 10, 20]:
        for alpha in [0.25, 0.5, 0.75]:
            name = f"Composite-{n} (a={alpha})"
            print(f"  {name}...")
            cw = build_composite_portfolio(
                join_scores, leave_scores, panel,
                n_long=n, n_short=n, alpha=alpha, beta=alpha,
                weighting="equal", gross_exposure=2.0, model_name=model_name,
            )
            s = _run_strategy(bt, cw, name, start_date=signal_start)
            if s:
                composite_strategies.append(s)

    # ── 5. NEW: Asymmetric leg strategies ────────────────────────────────
    print("\n5. Building asymmetric leg strategies...")
    asym_strategies = []
    for n_l, n_s in [(5, 20), (5, 30), (10, 30), (10, 50)]:
        name = f"Asym-{n_l}L/{n_s}S"
        print(f"  {name}...")
        aw = build_topn_portfolio(
            join_scores, leave_scores, panel,
            n_long=n_l, n_short=n_s,
            weighting="equal", gross_exposure=2.0, model_name=model_name,
        )
        s = _run_strategy(bt, aw, name, start_date=signal_start)
        if s:
            asym_strategies.append(s)

    # ── 6. NEW: Vol-scaled strategies ────────────────────────────────────
    print("\n6. Building vol-scaled strategies...")
    vol_strategies = []
    for n in [10, 20]:
        for gamma in [0.0, 0.3, 0.5]:
            name = f"VolScaled-{n} (g={gamma})"
            print(f"  {name}...")
            vw = build_volscaled_portfolio(
                join_scores, leave_scores, panel,
                n_long=n, n_short=n, gamma=gamma,
                gross_exposure=2.0, model_name=model_name,
            )
            s = _run_strategy(bt, vw, name, start_date=signal_start)
            if s:
                vol_strategies.append(s)

    # ── 7. NEW: Momentum-filtered strategies ─────────────────────────────
    print("\n7. Building momentum-filtered strategies...")
    mom_strategies = []
    for n in [10, 20]:
        name = f"MomFilter-{n}"
        print(f"  {name}...")
        mw = build_momentum_filtered_portfolio(
            join_scores, leave_scores, panel,
            n_long=n, n_short=n, mom_window=21,
            weighting="equal", gross_exposure=2.0, model_name=model_name,
        )
        s = _run_strategy(bt, mw, name, start_date=signal_start)
        if s:
            mom_strategies.append(s)

    # ── 8. Collect all strategies ────────────────────────────────────────
    all_strategies = []
    if omni:
        all_strategies.append(omni)
    if quantile_strat:
        all_strategies.append(quantile_strat)
    all_strategies.extend(topn_strategies)
    all_strategies.extend(composite_strategies)
    all_strategies.extend(asym_strategies)
    all_strategies.extend(vol_strategies)
    all_strategies.extend(mom_strategies)

    # ── 9. Strategy comparison table ─────────────────────────────────────
    print("\n8. Generating comparison tables...")
    comp_rows = [{"strategy": s["name"], **s["metrics"]} for s in all_strategies]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_tab / "strategy_comparison.csv", index=False)
    print(comp_df[["strategy", "annual_return", "sharpe_ratio", "max_drawdown",
                    "avg_long_per_rebal", "avg_short_per_rebal"]].to_string(index=False))

    # Save individual summaries
    if omni:
        pd.DataFrame([omni["metrics"]]).to_csv(out_tab / "backtest_summary_omniscient.csv", index=False)

    # Find best strategy by Sharpe (excluding omniscient)
    predictive = [s for s in all_strategies if s["name"] != "Omniscient"]
    best = max(predictive, key=lambda s: s["metrics"].get("sharpe_ratio", -999)) if predictive else None
    if best:
        pd.DataFrame([best["metrics"]]).to_csv(out_tab / "backtest_summary_predictive.csv", index=False)
        print(f"\n  Best strategy: {best['name']}")
        print(f"    Sharpe: {best['metrics']['sharpe_ratio']:.3f}")
        print(f"    Annual return: {best['metrics']['annual_return']:.4f}")

    # ── 10. Figures ──────────────────────────────────────────────────────
    print("\n10. Generating figures...")

    # Identify key strategies for comparison plots
    key_series = {}
    if omni:
        key_series["Omniscient"] = omni["result"]["returns"]
    if best:
        key_series[f"Best: {best['name']}"] = best["result"]["returns"]
    # Also add best Sharpe from each new category
    for cat_name, cat_list in [("top-N", topn_strategies),
                                ("composite", composite_strategies),
                                ("asymmetric", asym_strategies),
                                ("vol-scaled", vol_strategies)]:
        if cat_list:
            cat_best = max(cat_list, key=lambda s: s["metrics"].get("sharpe_ratio", -999))
            label = f"Best {cat_name}: {cat_best['name']}"
            key_series[label] = cat_best["result"]["returns"]

    if key_series:
        common_start = max(s.index.min() for s in key_series.values())
        aligned = {k: v[v.index >= common_start] for k, v in key_series.items()}
        plot_strategy_comparison(aligned, save_path=out_fig / "cumulative_returns_comparison.png")

    # Annual returns bar chart (key strategies)
    if key_series:
        plot_annual_returns(
            {k: v for k, v in aligned.items()},
            title="Annual returns by year",
            save_path=out_fig / "annual_returns.png",
        )

    # Rolling metrics (Sharpe + return)
    if key_series:
        plot_rolling_metrics(
            aligned,
            title="Rolling 1-year Sharpe ratio and annual return",
            save_path=out_fig / "rolling_metrics.png",
        )

    # Individual plots for best strategy
    if best:
        ret_best = best["result"]["returns"]
        plot_cumulative_returns(ret_best, title=f"Cumulative returns: {best['name']}",
                                save_path=out_fig / "cumulative_returns.png")
        plot_drawdowns(ret_best, title=f"Drawdown: {best['name']}",
                       save_path=out_fig / "drawdown.png")
        plot_turnover(best["result"]["turnover"],
                      title=f"Turnover: {best['name']}",
                      save_path=out_fig / "turnover.png")
        plot_exposure(best["result"]["gross_exposure"],
                      best["result"]["net_exposure"],
                      title=f"Exposure: {best['name']}",
                      save_path=out_fig / "exposure.png")

    # Drawdowns for omniscient
    if omni:
        plot_drawdowns(omni["result"]["returns"], title="Drawdown: Omniscient",
                       save_path=out_fig / "drawdown_omniscient.png")

    # ── 12. Subperiod analysis ───────────────────────────────────────────
    print("\n11. Subperiod analysis...")
    if best:
        subperiod = compute_subperiod_metrics(best["result"]["returns"], window_years=3.0)
        if subperiod:
            sub_df = pd.DataFrame(subperiod)
            sub_sampled = sub_df.iloc[::63].tail(20)
            sub_sampled.to_csv(out_tab / "backtest_subperiod_3y.csv", index=False)

    # ── 13. Factor attribution ───────────────────────────────────────────
    print("\n12. Factor attribution...")
    factors = load_factors(config=cfg)
    if factors is not None and best:
        fac_res = run_factor_regression(best["result"]["returns"], factors)
        pd.DataFrame([fac_res]).to_csv(out_tab / "factor_loadings_predictive.csv", index=False)
        if fac_res.get("betas"):
            plot_factor_loadings(pd.Series(fac_res["betas"]),
                                 save_path=out_fig / "factor_loadings.png")
    else:
        print("  No factor data available (raw_factors not configured).")

    print("\nDone. Results saved to results/figures/ and results/tables/")


if __name__ == "__main__":
    main()

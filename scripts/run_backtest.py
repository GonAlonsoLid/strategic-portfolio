"""End-to-end pipeline: load data, features, models, portfolios, backtest, results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.config_loader import load_config
from src.data.load_data import load_events, load_config_paths, build_ticker_permno_bridge
from src.data.preprocess_data import build_daily_panel
from src.features.feature_engineering import build_feature_panel, save_feature_datasets
from src.models.join_prediction import run_join_prediction
from src.models.leave_prediction import run_leave_prediction
from src.portfolio.portfolio_construction import build_long_short_portfolio
from src.backtesting.backtester import Backtester
from src.evaluation.performance_metrics import compute_performance_metrics, compute_drawdowns, compute_subperiod_metrics
from src.evaluation.factor_analysis import run_factor_regression, load_factors
from src.utils.plotting import plot_cumulative_returns, plot_drawdowns, plot_factor_loadings


def main() -> None:
    cfg = load_config()
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent
    interim = base / paths.get("interim", "data/interim")
    processed = base / paths.get("processed", "data/processed")
    out_fig = base / paths.get("results_figures", "results/figures")
    out_tab = base / paths.get("results_tables", "results/tables")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    panel_path = interim / "daily_panel.parquet"
    panel_csv = interim / "daily_panel.csv"
    if not panel_path.exists() and not panel_csv.exists():
        print("Building daily panel (max_chunks=20 for testing; set max_chunks=None for full run)...")
        build_daily_panel(config=cfg, max_chunks=20)
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        panel = pd.read_csv(panel_csv, parse_dates=["date"])
    if "market_ret" not in panel.columns:
        panel["market_ret"] = panel.groupby("date")["ret"].transform("mean")

    # Features
    features_join_path = processed / "features_join.parquet"
    features_leave_path = processed / "features_leave.parquet"
    features_join_csv = processed / "features_join.csv"
    features_leave_csv = processed / "features_leave.csv"
    if not (features_join_path.exists() or features_join_csv.exists()):
        print("Building features and labels...")
        features_join, features_leave = build_feature_panel(panel, config=cfg)
        save_feature_datasets(features_join, features_leave, config=cfg)
    if features_join_path.exists():
        features_join = pd.read_parquet(features_join_path)
        features_leave = pd.read_parquet(features_leave_path)
    elif features_join_csv.exists():
        features_join = pd.read_csv(features_join_csv, parse_dates=["date"])
        features_leave = pd.read_csv(features_leave_csv, parse_dates=["date"])
    else:
        features_join, features_leave = build_feature_panel(panel, config=cfg)

    # Models and scores
    join_scores_path = processed / "join_scores.parquet"
    leave_scores_path = processed / "leave_scores.parquet"
    join_scores_csv = processed / "join_scores.csv"
    leave_scores_csv = processed / "leave_scores.csv"
    if not (join_scores_path.exists() or join_scores_csv.exists()):
        print("Training models and generating scores...")
        run_join_prediction(features_join, config=cfg)
        run_leave_prediction(features_leave, config=cfg)
    if join_scores_path.exists():
        join_scores = pd.read_parquet(join_scores_path)
        leave_scores = pd.read_parquet(leave_scores_path)
    elif join_scores_csv.exists():
        join_scores = pd.read_csv(join_scores_csv, parse_dates=["date"])
        leave_scores = pd.read_csv(leave_scores_csv, parse_dates=["date"])
    else:
        run_join_prediction(features_join, config=cfg)
        run_leave_prediction(features_leave, config=cfg)
        join_scores = pd.read_parquet(join_scores_path) if join_scores_path.exists() else pd.read_csv(join_scores_csv, parse_dates=["date"])
        leave_scores = pd.read_parquet(leave_scores_path) if leave_scores_path.exists() else pd.read_csv(leave_scores_csv, parse_dates=["date"])

    # Portfolio: predictive strategy
    backtest_cfg = cfg.get("backtest", {})
    target_weights = build_long_short_portfolio(
        join_scores,
        leave_scores,
        panel,
        config=cfg,
        top_decile=backtest_cfg.get("top_decile", 0.10),
        weighting=backtest_cfg.get("weighting", "equal"),
        model_name="random_forest",
    )
    if target_weights.empty:
        p_join_col = [c for c in join_scores.columns if c.startswith("p_join_")][0]
        model_name = p_join_col.replace("p_join_", "")
        target_weights = build_long_short_portfolio(
            join_scores, leave_scores, panel, config=cfg, model_name=model_name
        )

    # Backtest
    bt = Backtester(panel, transaction_cost_bps=backtest_cfg.get("transaction_cost_bps", 10))
    result = bt.run_backtest(target_weights)
    ret = result["returns"]

    # Metrics (incl. VaR, skewness; syllabus: allocators look at these)
    metrics = compute_performance_metrics(ret)
    pd.DataFrame([metrics]).to_csv(out_tab / "backtest_summary_predictive.csv", index=False)
    subperiod = compute_subperiod_metrics(ret, window_years=3.0)
    if subperiod:
        pd.DataFrame(subperiod[-10:]).to_csv(out_tab / "backtest_subperiod_3y.csv", index=False)
    dd_df = compute_drawdowns(ret)
    plot_cumulative_returns(ret, title="Strategy cumulative returns", save_path=out_fig / "cumulative_returns.png")
    plot_drawdowns(ret, title="Drawdown", save_path=out_fig / "drawdown.png")

    # Factor attribution (optional)
    factors = load_factors(config=cfg)
    if factors is not None:
        fac_res = run_factor_regression(ret, factors)
        pd.DataFrame([fac_res]).to_csv(out_tab / "factor_loadings_predictive.csv", index=False)
        if fac_res.get("betas"):
            plot_factor_loadings(pd.Series(fac_res["betas"]), save_path=out_fig / "factor_loadings.png")

    print("Done. Results in results/figures and results/tables.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()

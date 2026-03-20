"""Load features, run join and leave prediction pipelines, save scores and metrics.

Pass --rebuild-features to force feature regeneration even if cached files exist.
Pass --skip-quality to skip model quality analysis (IC, ICIR, SHAP) after training.
Pass --use-gpu to enable GPU acceleration for XGBoost and LightGBM (auto-detects CUDA/MPS).
With --use-gpu and 2+ GPUs available, join and leave predictions run in parallel
on separate GPUs automatically.
"""
import sys
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.utils.config_loader import load_config
from src.data.load_data import load_config_paths
from src.features.feature_engineering import build_feature_panel, save_feature_datasets
from src.models.join_prediction import run_join_prediction
from src.models.leave_prediction import run_leave_prediction


def _count_gpus() -> int:
    """Return number of available CUDA GPUs via nvidia-smi."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return len([l for l in r.stdout.strip().splitlines() if l.strip()])
    except Exception:
        pass
    return 0


def _prediction_worker(task: str, features_path: str, cfg: dict, gpu_id: int) -> None:
    """Subprocess worker: load features from parquet, run prediction on a specific GPU.

    CUDA_VISIBLE_DEVICES restricts the process to one GPU; 'cuda' inside XGBoost/LightGBM
    resolves to that GPU only.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    df = pd.read_parquet(features_path)
    if task == "join":
        print(f"[GPU {gpu_id}] Starting JOIN prediction ({len(df):,} rows)...")
        run_join_prediction(df, config=cfg, use_gpu=True)
    else:
        print(f"[GPU {gpu_id}] Starting LEAVE prediction ({len(df):,} rows)...")
        run_leave_prediction(df, config=cfg, use_gpu=True)


def run_model_quality_analysis(
    scores: pd.DataFrame,
    features_join: pd.DataFrame,
    metrics_df: pd.DataFrame,
    processed: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Run Phase 1 model quality analyses: IC, ICIR, SHAP, comparison table.

    Produces:
        results/tables/model_comparison.csv  (MODEL-05)
        results/figures/ic_decay.png         (MODEL-02)
        results/tables/ic_decay.csv
        results/figures/shap_importance.png  (MODEL-04)
        results/tables/shap_importance.csv
    """
    import joblib

    from src.evaluation.model_quality import (
        compute_ic_series,
        compute_ic_decay,
        compute_icir,
        compute_shap_importance,
        build_model_comparison_table,
        plot_ic_decay,
        plot_shap_importance,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Forward returns for IC computation
    fwd_ret_cols = [c for c in features_join.columns if c.startswith("fwd_ret_")]
    fwd_returns = features_join[["date", "permno"] + fwd_ret_cols].drop_duplicates()

    prob_cols = [c for c in scores.columns if c.startswith("p_join_")]
    model_names = [c.replace("p_join_", "") for c in prob_cols]
    print(f"Models found: {model_names}")

    # MODEL-01 / MODEL-03: IC and ICIR per model (21-day forward return)
    ic_per_model: dict[str, float] = {}
    icir_per_model: dict[str, float] = {}
    for model_name, prob_col in zip(model_names, prob_cols):
        ic_series = compute_ic_series(scores, fwd_returns, prob_col, fwd_ret_col="fwd_ret_21d")
        mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
        icir = compute_icir(ic_series)
        ic_per_model[model_name] = mean_ic
        icir_per_model[model_name] = icir
        print(f"  {model_name}: IC={mean_ic:.4f}, ICIR={icir:.4f}, N_dates={len(ic_series)}")

    # Best model by ICIR; fall back to AUC if ICIR undefined
    valid_icir = {k: v for k, v in icir_per_model.items() if not np.isnan(v)}
    if valid_icir:
        best_model_name = max(valid_icir, key=valid_icir.get)
    else:
        best_model_name = metrics_df.groupby("model")["roc_auc"].mean().idxmax()
    print(f"Best model (by ICIR): {best_model_name}")

    # MODEL-02: IC decay for best model
    best_prob_col = f"p_join_{best_model_name}"
    ic_decay = compute_ic_decay(scores, fwd_returns, best_prob_col, horizons=[1, 5, 21, 63])
    print(f"IC decay for {best_model_name}: {ic_decay}")
    plot_ic_decay(ic_decay, best_model_name, save_path=str(figures_dir / "ic_decay.png"))
    ic_decay_df = pd.DataFrame([{"horizon": h, "ic": v} for h, v in ic_decay.items()])
    ic_decay_df.to_csv(tables_dir / "ic_decay.csv", index=False)
    print("Saved results/figures/ic_decay.png and results/tables/ic_decay.csv")

    # MODEL-04: SHAP importance for best model
    best_model_path = processed / "best_model.joblib"
    best_features_path = processed / "best_model_features.joblib"
    if best_model_path.exists() and best_features_path.exists():
        model = joblib.load(best_model_path)
        feat_cols = joblib.load(best_features_path)
        oos_keys = scores[["date", "permno"]].drop_duplicates()
        oos_features = features_join.merge(oos_keys, on=["date", "permno"], how="inner")
        X_shap = oos_features[feat_cols].fillna(0)
        if len(X_shap) > 5000:
            X_shap = X_shap.sample(5000, random_state=42)
        shap_importance = compute_shap_importance(model, X_shap, max_samples=2000)
        print(f"Top 10 SHAP features:\n{shap_importance.head(10)}")
        plot_shap_importance(shap_importance, best_model_name, top_n=15,
                             save_path=str(figures_dir / "shap_importance.png"))
        shap_importance.to_csv(tables_dir / "shap_importance.csv")
        print("Saved results/figures/shap_importance.png and results/tables/shap_importance.csv")
    else:
        print(f"WARNING: best_model.joblib not found at {best_model_path}; skipping SHAP.")

    # MODEL-05: Model comparison table
    comparison = build_model_comparison_table(metrics_df, ic_per_model, icir_per_model)
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    print(f"Model comparison table:\n{comparison.to_string()}")
    print("Saved results/tables/model_comparison.csv")

    print("\n=== Phase 1 Model Quality Analysis Complete ===")


def main() -> None:
    import argparse
    from src.models.model_utils import detect_gpu

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true", help="Force feature regeneration")
    parser.add_argument("--skip-quality", action="store_true", help="Skip model quality analysis after training")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU for XGBoost/LightGBM (auto-detects CUDA/MPS)")
    args = parser.parse_args()

    if args.use_gpu:
        gpu_device = detect_gpu()
        print(f"GPU requested — detected device: {gpu_device}")
    use_gpu = args.use_gpu

    cfg = load_config()
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent
    interim = base / paths.get("interim", "data/interim")
    processed = base / paths.get("processed", "data/processed")

    features_join_path = processed / "features_join.parquet"
    features_leave_path = processed / "features_leave.parquet"

    if not args.rebuild_features and features_join_path.exists() and features_leave_path.exists():
        print("Loading cached features...")
        features_join = pd.read_parquet(features_join_path)
        features_leave = pd.read_parquet(features_leave_path)
    else:
        panel_path = interim / "daily_panel.parquet"
        panel_csv = interim / "daily_panel.csv"
        if not panel_path.exists() and not panel_csv.exists():
            print("daily_panel not found. Building panel (max_chunks=20 for speed)...")
            from src.data.preprocess_data import build_daily_panel
            build_daily_panel(config=cfg, max_chunks=20)
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
        else:
            panel = pd.read_csv(panel_csv, parse_dates=["date"])
        if "market_ret" not in panel.columns:
            panel["market_ret"] = panel.groupby("date")["ret"].transform("mean")
        print("Building feature panel and labels...")
        features_join, features_leave = build_feature_panel(panel, config=cfg)
        save_feature_datasets(features_join, features_leave, config=cfg)

    n_gpus = _count_gpus() if use_gpu else 0
    tables = base / paths.get("results_tables", "results/tables")

    if use_gpu and n_gpus >= 2:
        # Parallel execution: join on GPU 0, leave on GPU 1.
        # Each worker reads its own parquet (already on disk) and saves results independently.
        # spawn avoids CUDA context inheritance issues from the parent process.
        print(f"Multi-GPU mode ({n_gpus} GPUs): join→GPU 0 | leave→GPU 1 (parallel)")
        ctx = mp.get_context("spawn")
        p_join = ctx.Process(
            target=_prediction_worker,
            args=("join", str(features_join_path), cfg, 0),
        )
        p_leave = ctx.Process(
            target=_prediction_worker,
            args=("leave", str(features_leave_path), cfg, 1),
        )
        p_join.start()
        p_leave.start()
        p_join.join()
        p_leave.join()
        if p_join.exitcode != 0 or p_leave.exitcode != 0:
            raise RuntimeError(
                f"Prediction worker failed — join exit={p_join.exitcode}, leave exit={p_leave.exitcode}"
            )
        # Reload results saved by worker processes
        scores_df = pd.read_parquet(processed / "join_scores.parquet")
        metrics_csv = tables / "model_performance_join.csv"
        metrics_df = pd.read_csv(metrics_csv) if metrics_csv.exists() else pd.DataFrame()
        print("Done. Both GPUs finished.")
    else:
        if use_gpu and n_gpus < 2:
            print(f"Single GPU detected ({n_gpus}) — running sequentially.")
        print("Running join prediction...")
        scores_df, metrics_df = run_join_prediction(features_join, config=cfg, use_gpu=use_gpu)
        print("Running leave prediction...")
        run_leave_prediction(features_leave, config=cfg, use_gpu=use_gpu)
        print("Done. Scores in data/processed, metrics in results/tables.")

    # Save best model (by avg ROC-AUC) for SHAP analysis in Plan 03
    if not metrics_df.empty and len(scores_df) > 0:
        import joblib
        from src.models.join_prediction import _get_model
        from src.models.model_utils import get_feature_columns, make_rolling_splits

        best_model_name = metrics_df.groupby("model")["roc_auc"].mean().idxmax()
        print(f"Best model by AUC: {best_model_name}")

        feat_cols = get_feature_columns(features_join, exclude=["date", "permno", "ticker", "label_join"])
        X = features_join[feat_cols].fillna(0)
        y = features_join["label_join"]

        splits = make_rolling_splits(features_join, train_years=5, test_years=1)
        if splits:
            last_train_idx, _ = splits[-1]
            model = _get_model(best_model_name, cfg, 42)
            if model is not None:
                if best_model_name == "gradient_boosting":
                    n_pos_train = y.loc[last_train_idx].sum()
                    n_neg_train = len(last_train_idx) - n_pos_train
                    imbalance_ratio = n_neg_train / max(n_pos_train, 1)
                    sw = np.where(y.loc[last_train_idx] == 1, imbalance_ratio, 1.0)
                    model.fit(X.loc[last_train_idx], y.loc[last_train_idx], sample_weight=sw)
                elif best_model_name == "xgboost" and hasattr(model, "set_params"):
                    n_pos_train = y.loc[last_train_idx].sum()
                    n_neg_train = len(last_train_idx) - n_pos_train
                    model.set_params(scale_pos_weight=n_neg_train / max(n_pos_train, 1))
                    model.fit(X.loc[last_train_idx], y.loc[last_train_idx])
                else:
                    model.fit(X.loc[last_train_idx], y.loc[last_train_idx])
                joblib.dump(model, processed / "best_model.joblib")
                joblib.dump(feat_cols, processed / "best_model_features.joblib")
                print(f"Saved best model ({best_model_name}) to data/processed/best_model.joblib")
    else:
        print("WARNING: No metrics or scores produced; skipping best model save.")

    # Run Phase 1 model quality analysis (IC, ICIR, SHAP, comparison table)
    if not args.skip_quality and not metrics_df.empty and len(scores_df) > 0:
        figures_dir = base / "results" / "figures"
        tables_dir = base / "results" / "tables"
        print("\nRunning Phase 1 model quality analysis...")
        run_model_quality_analysis(
            scores=scores_df,
            features_join=features_join,
            metrics_df=metrics_df,
            processed=processed,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
        )
    elif args.skip_quality:
        print("Skipping model quality analysis (--skip-quality flag set).")
    else:
        print("WARNING: No metrics or scores produced; skipping model quality analysis.")


if __name__ == "__main__":
    main()

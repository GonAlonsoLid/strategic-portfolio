"""Load features, run join and leave prediction pipelines, save scores and metrics.

Pass --rebuild-features to force feature regeneration even if cached files exist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.config_loader import load_config
from src.data.load_data import load_config_paths
from src.features.feature_engineering import build_feature_panel, save_feature_datasets
from src.models.join_prediction import run_join_prediction
from src.models.leave_prediction import run_leave_prediction


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true", help="Force feature regeneration")
    args = parser.parse_args()

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

    print("Running join prediction...")
    run_join_prediction(features_join, config=cfg)
    print("Running leave prediction...")
    run_leave_prediction(features_leave, config=cfg)
    print("Done. Scores in data/processed, metrics in results/tables.")


if __name__ == "__main__":
    main()

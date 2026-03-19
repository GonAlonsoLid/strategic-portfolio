"""Train models predicting probability of leaving S&P 500; save scores and metrics."""
from pathlib import Path
from typing import List, Optional
import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.utils.config_loader import load_config, get_section
from src.models.model_utils import make_rolling_splits, train_and_evaluate, get_feature_columns, detect_gpu, _predict_proba

try:
    import xgboost as xgb  # noqa: F401
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb  # noqa: F401
    HAS_LGB = True
except Exception:
    HAS_LGB = False


def _get_model(name: str, config: dict, random_state: int, use_gpu: bool = False):
    cfg = config.get("models", {})
    if name == "logistic":
        p = cfg.get("logistic", {})
        return LogisticRegression(max_iter=p.get("max_iter", 1000), C=p.get("C", 1.0),
                                  class_weight="balanced", random_state=random_state)
    if name == "random_forest":
        p = cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 10),
            min_samples_leaf=p.get("min_samples_leaf", 50),
            class_weight="balanced",
            random_state=random_state,
        )
    if name == "gradient_boosting":
        p = cfg.get("gradient_boosting", {})
        return GradientBoostingClassifier(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.1),
            random_state=random_state,
        )
    if name == "xgboost" and HAS_XGB:
        p = cfg.get("xgboost", {})
        xgb_params: dict = dict(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.1),
            scale_pos_weight=1,
            random_state=random_state,
            eval_metric="logloss",
        )
        if use_gpu:
            xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
            if xgb_version >= (2, 0):
                xgb_params["device"] = "cuda"
            else:
                xgb_params["tree_method"] = "gpu_hist"
        return xgb.XGBClassifier(**xgb_params)
    if name == "lightgbm" and HAS_LGB:
        p = cfg.get("lightgbm", {})
        lgb_params: dict = dict(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.1),
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
        if use_gpu:
            lgb_params["device"] = "cuda"
        return lgb.LGBMClassifier(**lgb_params)
    return None


def run_leave_prediction(
    features_leave: pd.DataFrame,
    config: dict | None = None,
    *,
    model_types: List[str] | None = None,
    output_scores_path: str | Path | None = None,
    output_metrics_path: str | Path | None = None,
    use_gpu: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train leave models over rolling windows; return (scores panel, metrics table)."""
    cfg = config or load_config()
    base = Path(__file__).resolve().parent.parent.parent
    paths = cfg.get("paths", {})
    processed = base / paths.get("processed", "data/processed")
    tables = base / paths.get("results_tables", "results/tables")
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    model_types = model_types or get_section(cfg, "models", "types") or ["logistic", "random_forest", "gradient_boosting"]
    if "xgboost" in model_types and not HAS_XGB:
        model_types = [m for m in model_types if m != "xgboost"]
    if "lightgbm" in model_types and not HAS_LGB:
        model_types = [m for m in model_types if m != "lightgbm"]
    if use_gpu:
        gpu_device = detect_gpu()
        print(f"GPU mode enabled — device: {gpu_device}")
        if gpu_device == "cpu":
            print("WARNING: No GPU detected; falling back to CPU.")
            use_gpu = False
    train_years = get_section(cfg, "models", "train_years") or 5
    test_years = get_section(cfg, "models", "test_years") or 1
    random_state = get_section(cfg, "models", "random_state") or 42

    feat_cols = get_feature_columns(features_leave, exclude=["date", "permno", "ticker", "label_leave"])
    if not feat_cols:
        feat_cols = [c for c in features_leave.select_dtypes(include=[np.number]).columns if c != "label_leave"]
    X = features_leave[feat_cols].fillna(0)
    y = features_leave["label_leave"]
    date_idx = features_leave["date"]
    permno = features_leave["permno"]

    splits = make_rolling_splits(features_leave, train_years=train_years, test_years=test_years, date_col="date")
    all_scores = []
    metrics_rows = []
    n_folds = len(splits)
    n_models = len(model_types)

    print(f"\n{'='*60}")
    print(f"LEAVE PREDICTION — {n_models} model(s), {n_folds} folds each")
    print(f"{'='*60}")

    for m_idx, model_name in enumerate(model_types):
        model = _get_model(model_name, cfg, random_state, use_gpu=use_gpu)
        if model is None:
            continue
        scale = model_name == "logistic"
        print(f"\n[{time.strftime('%H:%M:%S')}] Model {m_idx+1}/{n_models}: {model_name.upper()}")
        fold_times = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                print(f"  [{time.strftime('%H:%M:%S')}] Fold {fold+1:02d}/{n_folds} — skipped (single class)")
                continue

            # Compute class imbalance ratio for this fold
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            imbalance_ratio = n_neg / n_pos if n_pos > 0 else 1.0

            # Set scale_pos_weight for XGBoost
            if model_name == "xgboost" and hasattr(model, "set_params"):
                model.set_params(scale_pos_weight=imbalance_ratio)

            # Compute sample_weight for GradientBoosting
            if model_name == "gradient_boosting":
                sw = np.where(y_train == 1, imbalance_ratio, 1.0)
            else:
                sw = None

            t0 = time.time()
            met = train_and_evaluate(model, X_train, y_train, X_test, y_test, scale=scale, sample_weight=sw)
            elapsed = time.time() - t0
            fold_times.append(elapsed)

            avg_time = sum(fold_times) / len(fold_times)
            remaining = (n_folds - fold - 1) * avg_time
            eta = time.strftime('%H:%M:%S', time.localtime(time.time() + remaining))

            auc = met.get("roc_auc", float("nan"))
            print(
                f"  [{time.strftime('%H:%M:%S')}] Fold {fold+1:02d}/{n_folds} | "
                f"train={len(X_train):>7,} test={len(X_test):>6,} | "
                f"AUC={auc:.4f} | {elapsed:.1f}s | ETA {eta}"
            )

            met["model"] = model_name
            met["fold"] = fold
            metrics_rows.append(met)
            proba = _predict_proba(model, X_test)[:, 1]
            for i, idx in enumerate(test_idx):
                all_scores.append({
                    "date": date_idx.loc[idx],
                    "permno": permno.loc[idx],
                    f"p_leave_{model_name}": proba[i],
                })

        total_model_time = sum(fold_times)
        print(f"  [{time.strftime('%H:%M:%S')}] {model_name.upper()} done — total {total_model_time/60:.1f} min")

    scores_df = pd.DataFrame(all_scores)
    if not scores_df.empty:
        scores_df = scores_df.groupby(["date", "permno"]).mean().reset_index()
    metrics_df = pd.DataFrame(metrics_rows)

    out_scores = output_scores_path or processed / "leave_scores.parquet"
    out_metrics = output_metrics_path or tables / "model_performance_leave.csv"
    try:
        scores_df.to_parquet(out_scores, index=False)
    except ImportError:
        scores_df.to_csv(Path(out_scores).with_suffix(".csv"), index=False)
    if not metrics_df.empty:
        metrics_df.to_csv(out_metrics, index=False)
    return scores_df, metrics_df

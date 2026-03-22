"""Model quality analysis: IC, ICIR, IC decay, SHAP importance, model comparison table."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic_series(
    scores_df: pd.DataFrame,
    fwd_returns_df: pd.DataFrame,
    prob_col: str,
    fwd_ret_col: str = "fwd_ret_21d",
    date_col: str = "date",
    permno_col: str = "permno",
    min_obs: int = 5,
) -> pd.Series:
    """Per-date IC: Spearman rank correlation between predicted probability and forward return.

    Args:
        scores_df: OOS predictions with columns [date, permno, prob_col].
        fwd_returns_df: Forward returns with columns [date, permno, fwd_ret_col].
        prob_col: Column name for predicted probability (e.g., 'p_join_logistic').
        fwd_ret_col: Column name for forward return (e.g., 'fwd_ret_21d').
        min_obs: Minimum observations per date to compute IC (avoid undefined correlation).

    Returns:
        pd.Series indexed by date with IC values.
    """
    merged = scores_df[[date_col, permno_col, prob_col]].merge(
        fwd_returns_df[[date_col, permno_col, fwd_ret_col]],
        on=[date_col, permno_col],
        how="inner",
    )
    merged = merged.dropna(subset=[prob_col, fwd_ret_col])

    def _ic(g):
        if len(g) < min_obs:
            return np.nan
        corr, _ = spearmanr(g[prob_col], g[fwd_ret_col])
        return corr

    ic = merged.groupby(date_col).apply(_ic, include_groups=False)
    return ic.dropna().rename("ic")


def compute_icir(ic_series: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC). Higher = more consistent signal quality."""
    if len(ic_series) == 0 or ic_series.std() == 0:
        return np.nan
    return float(ic_series.mean() / ic_series.std())


def compute_ic_decay(
    scores_df: pd.DataFrame,
    fwd_returns_df: pd.DataFrame,
    prob_col: str,
    horizons: list[int] | None = None,
    date_col: str = "date",
    permno_col: str = "permno",
) -> dict[int, float]:
    """IC at multiple forward-return horizons for a single model.

    Returns dict mapping horizon (days) -> mean IC.
    """
    if horizons is None:
        horizons = [1, 5, 21, 63]
    decay = {}
    for h in horizons:
        fwd_col = f"fwd_ret_{h}d"
        if fwd_col not in fwd_returns_df.columns:
            decay[h] = np.nan
            continue
        ic = compute_ic_series(scores_df, fwd_returns_df, prob_col, fwd_ret_col=fwd_col,
                               date_col=date_col, permno_col=permno_col)
        decay[h] = float(ic.mean()) if len(ic) > 0 else np.nan
    return decay


def compute_shap_importance(
    model,
    X_sample: pd.DataFrame,
    max_samples: int = 2000,
) -> pd.Series:
    """Mean |SHAP| per feature, sorted descending.

    Uses TreeExplainer for tree-based models, LinearExplainer for linear models.
    Subsamples to max_samples to avoid slow computation.
    """
    import shap

    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=42)

    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        # LinearExplainer for LogisticRegression and similar
        explainer = shap.LinearExplainer(model, X_sample)

    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers, TreeExplainer returns list of 2 arrays; take class-1 (positive)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns,
    ).sort_values(ascending=False)

    return importance.rename("mean_abs_shap")


def build_model_comparison_table(
    metrics_df: pd.DataFrame,
    ic_per_model: dict[str, float],
    icir_per_model: dict[str, float],
) -> pd.DataFrame:
    """Build model comparison table: AUC, Brier, OOS accuracy, IC, ICIR per model.

    Args:
        metrics_df: Per-fold metrics with columns [model, fold, roc_auc, brier_score, oos_accuracy, ...].
        ic_per_model: {model_name: mean_IC}.
        icir_per_model: {model_name: ICIR}.

    Returns:
        DataFrame with one row per model, columns: model, roc_auc, brier_score, oos_accuracy, ic, icir.
    """
    agg = metrics_df.groupby("model")[["roc_auc", "brier_score", "oos_accuracy"]].mean()
    agg["ic"] = agg.index.map(lambda m: ic_per_model.get(m, np.nan))
    agg["icir"] = agg.index.map(lambda m: icir_per_model.get(m, np.nan))
    return agg.reset_index()


def plot_ic_decay(
    ic_decay: dict[int, float],
    model_name: str,
    save_path: str | None = None,
):
    """Line chart: IC vs holding horizon (1d, 5d, 21d, 63d)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted(ic_decay)
    values = [ic_decay[h] for h in horizons]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(horizons, values, marker="o", linewidth=2, color="#2563eb")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Holding Horizon (trading days)")
    ax.set_ylabel("IC (Spearman rank correlation)")
    ax.set_title(f"IC Decay - {model_name}")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_shap_importance(
    importance: pd.Series,
    model_name: str,
    top_n: int = 15,
    save_path: str | None = None,
):
    """Horizontal bar chart of top N features by mean |SHAP|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance.head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(top.index, top.values, color="#2563eb")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance (SHAP) - {model_name}")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig

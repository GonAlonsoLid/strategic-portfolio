"""Build feature matrix and joiner/leaver labels (next 3 months)."""
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.utils.config_loader import load_config, get_section
from src.features.rolling_features import (
    add_momentum_features,
    add_momentum_skip_month,
    add_volatility_features,
    add_liquidity_features,
    add_abnormal_performance,
    add_quality_proxy,
)


def build_market_cap_rank(df: pd.DataFrame, *, date_col: str = "date", cap_col: str = "market_cap", permno_col: str = "permno") -> pd.DataFrame:
    """Cross-sectional rank and percentile of market cap per date."""
    df = df.copy()
    grouped = df.groupby(date_col, sort=False)[cap_col]
    df["market_cap_rank"] = grouped.rank(ascending=False, method="first")
    df["size_percentile"] = grouped.rank(pct=True, method="first")
    return df


def build_joiner_label(
    panel: pd.DataFrame,
    *,
    forward_days: int = 63,
    date_col: str = "date",
    permno_col: str = "permno",
    is_sp500_col: str = "is_sp500",
) -> pd.Series:
    """Label = 1 if firm enters S&P 500 in next forward_days trading days, else 0."""
    # Expects input sorted by [permno, date] — build_feature_panel pre-sorts once.
    # Reverse-roll-reverse within each permno to get a true forward-looking max.
    # At day t: max of is_sp500 over [t+1, t+forward_days].
    future_max = (
        panel.groupby(permno_col)[is_sp500_col]
        .transform(lambda x: x.iloc[::-1].rolling(forward_days, min_periods=1).max().iloc[::-1].shift(-1))
    )
    label = ((panel[is_sp500_col] == False) & (future_max == True)).astype(int)
    return label


def build_leaver_label(
    panel: pd.DataFrame,
    *,
    forward_days: int = 63,
    date_col: str = "date",
    permno_col: str = "permno",
    is_sp500_col: str = "is_sp500",
) -> pd.Series:
    """Label = 1 if firm exits S&P 500 in next forward_days trading days, else 0."""
    # Expects input sorted by [permno, date] — build_feature_panel pre-sorts once.
    # Reverse-roll-reverse within each permno to get a true forward-looking min.
    # At day t: min of is_sp500 over [t+1, t+forward_days].
    future_min = (
        panel.groupby(permno_col)[is_sp500_col]
        .transform(lambda x: x.iloc[::-1].rolling(forward_days, min_periods=1).min().iloc[::-1].shift(-1))
    )
    label = ((panel[is_sp500_col] == True) & (future_min == False)).astype(int)
    return label


def add_forward_returns(
    panel: pd.DataFrame,
    horizons: list[int] | None = None,
    *,
    date_col: str = "date",
    permno_col: str = "permno",
    ret_col: str = "ret",
) -> pd.DataFrame:
    """Add forward cumulative return columns: fwd_ret_{h}d for each horizon h.

    Forward return at horizon h = cumulative return from day t+1 to t+h.
    Uses groupby(permno) to avoid cross-firm contamination.
    """
    if horizons is None:
        horizons = [1, 5, 21, 63]
    panel = panel.copy()
    # Cache groupby once; log-sum trick: expm1(sum(log1p(r))) == prod(1+r) - 1, fully vectorized.
    grouped = panel.groupby(permno_col, sort=False)[ret_col]
    for h in horizons:
        col_name = f"fwd_ret_{h}d"
        if h == 1:
            # Forward 1-day return = next day's return
            panel[col_name] = grouped.shift(-1)
        else:
            # Forward h-day cumulative return = product of (1+ret) over next h days, minus 1
            def _fwd_cumret(x, h=h):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.expm1(np.log1p(x).rolling(h).sum()).shift(-h)
            panel[col_name] = grouped.transform(_fwd_cumret)
    return panel


def build_feature_panel(
    panel: pd.DataFrame,
    config: dict | None = None,
    *,
    min_history_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature matrix and joiner/leaver labels. Drops rows with insufficient history."""
    cfg = config or load_config()
    paths_cfg = cfg.get("paths", {})
    feat_cfg = cfg.get("features", {})
    momentum_w = feat_cfg.get("momentum_windows", [21, 63, 126, 252])
    vol_w = feat_cfg.get("volatility_windows", [21, 63, 126])
    label_days = feat_cfg.get("label_forward_trading_days", 63)
    min_hist = feat_cfg.get("min_history_days", min_history_days)

    # Sort once here — all sub-functions skip internal sorts and trust this order.
    panel = panel.sort_values(["permno", "date"]).copy()

    # Downcast float64 → float32: halves peak memory (~26 GB → ~13 GB) with no meaningful
    # loss of precision for financial returns or tree-based model accuracy.
    float_cols = panel.select_dtypes("float64").columns.tolist()
    panel[float_cols] = panel[float_cols].astype("float32")

    # Ensure market return for abnormal performance
    if "market_ret" not in panel.columns:
        panel["market_ret"] = panel.groupby("date", sort=False)["ret"].transform("mean")

    panel = add_momentum_features(panel, momentum_w)
    panel = add_momentum_skip_month(panel, long_window=252, skip_days=21)
    panel = add_volatility_features(panel, vol_w)
    panel = add_liquidity_features(panel, windows=[21])
    panel = add_abnormal_performance(panel, momentum_w, market_ret_col="market_ret")
    panel = add_quality_proxy(panel, vol_window=63)
    panel = build_market_cap_rank(panel)

    # Forward returns for IC computation (MODEL-02 IC decay)
    panel = add_forward_returns(panel, horizons=[1, 5, 21, 63])

    panel["label_join"] = build_joiner_label(panel, forward_days=label_days)
    panel["label_leave"] = build_leaver_label(panel, forward_days=label_days)

    # Feature columns (all rolling and cross-sectional)
    feat_cols = (
        [c for c in panel.columns if c.startswith("ret_") and "d" in c]
        + [c for c in panel.columns if c == "mom_12m_skip1m"]
        + [c for c in panel.columns if c.startswith("vol_")]
        + [c for c in panel.columns if c.startswith("turnover_") or c.startswith("volume_avg_")]
        + [c for c in panel.columns if c.startswith("excess_ret_")]
        + ["market_cap", "market_cap_rank", "size_percentile", "quality_proxy"]
    )
    feat_cols = [c for c in feat_cols if c in panel.columns]
    fwd_ret_cols = [c for c in panel.columns if c.startswith("fwd_ret_")]
    key_feats = [c for c in ["ret_21d", "market_cap_rank", "size_percentile"] if c in panel.columns]
    if key_feats:
        panel = panel.dropna(subset=key_feats)

    base_cols = ["date", "permno", "ticker"]
    features_join = panel[base_cols + feat_cols + fwd_ret_cols + ["label_join"]].copy()
    features_leave = panel[base_cols + feat_cols + fwd_ret_cols + ["label_leave"]].copy()
    return features_join, features_leave


def save_feature_datasets(
    features_join: pd.DataFrame,
    features_leave: pd.DataFrame,
    config: dict | None = None,
) -> None:
    """Save to data/processed/features_join and features_leave (parquet or csv)."""
    cfg = config or load_config()
    base = Path(__file__).resolve().parent.parent.parent
    processed = base / cfg.get("paths", {}).get("processed", "data/processed")
    processed.mkdir(parents=True, exist_ok=True)
    for name, df in [("features_join", features_join), ("features_leave", features_leave)]:
        try:
            df.to_parquet(processed / f"{name}.parquet", index=False)
        except ImportError:
            df.to_csv(processed / f"{name}.csv", index=False)

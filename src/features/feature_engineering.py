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
    df["market_cap_rank"] = df.groupby(date_col)[cap_col].rank(ascending=False, method="first")
    df["size_percentile"] = df.groupby(date_col)[cap_col].rank(pct=True, method="first")
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
    panel = panel.sort_values([permno_col, date_col]).copy()
    s = panel.groupby(permno_col)[is_sp500_col]
    # Max in next forward_days: shift(-1) then rolling(forward_days).max()
    next_max = s.shift(-1).rolling(forward_days, min_periods=1).max()
    label = ((panel[is_sp500_col] == False) & (next_max == True)).astype(int)
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
    panel = panel.sort_values([permno_col, date_col]).copy()
    s = panel.groupby(permno_col)[is_sp500_col]
    next_min = s.shift(-1).rolling(forward_days, min_periods=1).min()
    label = ((panel[is_sp500_col] == True) & (next_min == False)).astype(int)
    return label


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

    # Ensure market return for abnormal performance
    if "market_ret" not in panel.columns:
        panel = panel.copy()
        panel["market_ret"] = panel.groupby("date")["ret"].transform("mean")

    panel = add_momentum_features(panel, momentum_w)
    panel = add_momentum_skip_month(panel, long_window=252, skip_days=21)
    panel = add_volatility_features(panel, vol_w)
    panel = add_liquidity_features(panel, windows=[21])
    panel = add_abnormal_performance(panel, momentum_w, market_ret_col="market_ret")
    panel = add_quality_proxy(panel, vol_window=63)
    panel = build_market_cap_rank(panel)

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
    key_feats = [c for c in ["ret_21d", "market_cap_rank", "size_percentile"] if c in panel.columns]
    if key_feats:
        panel = panel.dropna(subset=key_feats)

    base_cols = ["date", "permno", "ticker"]
    features_join = panel[base_cols + feat_cols + ["label_join"]].copy()
    features_leave = panel[base_cols + feat_cols + ["label_leave"]].copy()
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

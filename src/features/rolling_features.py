"""Rolling-window features: momentum, volatility, liquidity (no lookahead).

Syllabus/underreaction: momentum (skip last month to limit reversal), profitability proxy,
excess returns vs market. See docs/RESEARCH_NOTES.md.

Expects input sorted by [permno, date] — build_feature_panel pre-sorts once.
"""
from typing import List

import pandas as pd
import numpy as np


def add_momentum_skip_month(
    df: pd.DataFrame,
    long_window: int = 252,
    skip_days: int = 21,
    *,
    ret_col: str = "ret",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Momentum from t-12m to t-1m (skip last month). Reduces short-term reversal (Jegadeesh).
    Standard in literature and underreaction lecture: Sit = r(t-12,t-2)."""
    df = df.copy()
    # At date t we want cumulative return from t-252 to t-21 (skip last ~1 month)
    def _mom_skip(x: pd.Series) -> pd.Series:
        r = (1 + x).cumprod()
        # r.shift(21) / r.shift(21+252) - 1 = cumret ending 21 days ago over 252 days
        out = r.shift(skip_days) / r.shift(skip_days + long_window) - 1
        out = out.shift(1)  # no lookahead
        return out
    df["mom_12m_skip1m"] = df.groupby(permno_col, sort=False)[ret_col].transform(_mom_skip)
    return df


def add_momentum_features(
    df: pd.DataFrame,
    windows: List[int],
    *,
    ret_col: str = "ret",
    date_col: str = "date",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Add cumulative return over rolling windows (1m, 3m, 6m, 12m). Past data only."""
    df = df.copy()
    # Cache groupby once; reuse across all windows.
    # log-sum trick: expm1(sum(log1p(r))) == prod(1+r) - 1, fully vectorized — no Python callbacks.
    # shift(1) inside transform is group-aware: first row of each permno → NaN (no lookahead).
    grouped = df.groupby(permno_col, sort=False)[ret_col]
    for w in windows:
        def _cumret(x, w=w):
            # ret=-1 (delisted) gives log1p(-1)=-inf → expm1(-inf)=-1 (correct); suppress warning.
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.expm1(np.log1p(x).rolling(w, min_periods=1).sum()).shift(1)
        df[f"ret_{w}d"] = grouped.transform(_cumret)
    return df


def add_volatility_features(
    df: pd.DataFrame,
    windows: List[int],
    *,
    ret_col: str = "ret",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Rolling standard deviation of returns."""
    df = df.copy()
    grouped = df.groupby(permno_col, sort=False)[ret_col]
    for w in windows:
        df[f"vol_{w}d"] = grouped.transform(
            lambda x, w=w: x.rolling(w, min_periods=min(5, w)).std()
        ).shift(1)
    return df


def add_liquidity_features(
    df: pd.DataFrame,
    *,
    volume_col: str = "volume",
    cap_col: str = "market_cap",
    permno_col: str = "permno",
    windows: List[int] = [21],
) -> pd.DataFrame:
    """Rolling average volume; turnover = volume / (market_cap/1e6) as liquidity proxy if cap available."""
    df = df.copy()
    if cap_col in df.columns and volume_col in df.columns:
        df["turnover"] = (df[volume_col] / (df[cap_col].replace(0, np.nan) / 1e6)).replace(np.inf, np.nan)
    else:
        df["turnover"] = np.nan
    # Cache groupby once; reuse for both turnover and volume columns.
    grouped = df.groupby(permno_col, sort=False)
    for w in windows:
        df[f"turnover_avg_{w}d"] = grouped["turnover"].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        ).shift(1)
        df[f"volume_avg_{w}d"] = grouped[volume_col].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        ).shift(1)
    return df


def add_abnormal_performance(
    df: pd.DataFrame,
    windows: List[int],
    *,
    ret_col: str = "ret",
    market_ret_col: str = "market_ret",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Excess return vs market over rolling windows (sector/market relative)."""
    if market_ret_col not in df.columns:
        return df
    df = df.copy()
    df["excess_ret"] = df[ret_col] - df[market_ret_col]
    grouped = df.groupby(permno_col, sort=False)["excess_ret"]
    for w in windows:
        df[f"excess_ret_{w}d"] = grouped.transform(
            lambda x, w=w: x.rolling(w, min_periods=1).sum()
        ).shift(1)
    return df


def add_quality_proxy(
    df: pd.DataFrame,
    *,
    ret_col: str = "ret",
    vol_window: int = 63,
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Profitability/quality proxy: return / volatility (Sharpe-like). No accounting data.
    Underreaction: high profitability (ROA/ROE) predicts returns; we proxy with past return/vol."""
    df = df.copy()
    # Cache groupby once for both transforms on the same column.
    grouped = df.groupby(permno_col, sort=False)[ret_col]
    vol = grouped.transform(lambda x: x.rolling(vol_window, min_periods=21).std()).shift(1)
    ret_ann = grouped.transform(lambda x: x.rolling(252, min_periods=63).sum()).shift(1)
    df["quality_proxy"] = (ret_ann / vol.replace(0, np.nan)).replace(np.inf, np.nan)
    return df

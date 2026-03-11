"""Rolling-window features: momentum, volatility, liquidity (no lookahead).

Syllabus/underreaction: momentum (skip last month to limit reversal), profitability proxy,
excess returns vs market. See docs/RESEARCH_NOTES.md.
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
    df = df.sort_values([permno_col, "date"]).copy()
    # At date t we want cumulative return from t-252 to t-21 (skip last ~1 month)
    def _mom_skip(x: pd.Series) -> pd.Series:
        r = (1 + x).cumprod()
        # r.shift(21) / r.shift(21+252) - 1 = cumret ending 21 days ago over 252 days
        out = r.shift(skip_days) / r.shift(skip_days + long_window) - 1
        out = out.shift(1)  # no lookahead
        return out
    df["mom_12m_skip1m"] = df.groupby(permno_col)[ret_col].transform(_mom_skip)
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
    df = df.sort_values([permno_col, date_col]).copy()
    for w in windows:
        df[f"ret_{w}d"] = df.groupby(permno_col)[ret_col].transform(
            lambda x: x.rolling(w, min_periods=1).apply(lambda y: (1 + y).prod() - 1 if len(y) >= 1 else np.nan)
        )
        # Shift so we use only past returns (no same-day lookahead)
        df[f"ret_{w}d"] = df.groupby(permno_col)[f"ret_{w}d"].shift(1)
    return df


def add_volatility_features(
    df: pd.DataFrame,
    windows: List[int],
    *,
    ret_col: str = "ret",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Rolling standard deviation of returns."""
    df = df.sort_values([permno_col, "date"]).copy()
    for w in windows:
        df[f"vol_{w}d"] = df.groupby(permno_col)[ret_col].transform(
            lambda x: x.rolling(w, min_periods=min(5, w)).std()
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
    df = df.sort_values([permno_col, "date"]).copy()
    if cap_col in df.columns and volume_col in df.columns:
        df["turnover"] = (df[volume_col] / (df[cap_col].replace(0, np.nan) / 1e6)).replace(np.inf, np.nan)
    else:
        df["turnover"] = np.nan
    for w in windows:
        df[f"turnover_avg_{w}d"] = df.groupby(permno_col)["turnover"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        ).shift(1)
        df[f"volume_avg_{w}d"] = df.groupby(permno_col)[volume_col].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
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
    df = df.sort_values([permno_col, "date"]).copy()
    df["excess_ret"] = df[ret_col] - df[market_ret_col]
    for w in windows:
        df[f"excess_ret_{w}d"] = df.groupby(permno_col)["excess_ret"].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
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
    df = df.sort_values([permno_col, "date"]).copy()
    vol = df.groupby(permno_col)[ret_col].transform(
        lambda x: x.rolling(vol_window, min_periods=21).std()
    ).shift(1)
    ret_ann = df.groupby(permno_col)[ret_col].transform(
        lambda x: x.rolling(252, min_periods=63).sum()
    ).shift(1)
    df["quality_proxy"] = (ret_ann / vol.replace(0, np.nan)).replace(np.inf, np.nan)
    return df

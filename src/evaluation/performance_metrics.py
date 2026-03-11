"""Performance and risk metrics: Sharpe, Sortino, drawdown, Calmar, VaR, skewness, subperiod."""
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


def compute_performance_metrics(
    returns: pd.Series,
    rf: Optional[pd.Series] = None,
    *,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Annualized return, vol, Sharpe, Sortino, max drawdown, Calmar, VaR(5%), skewness."""
    ret = returns.dropna()
    if ret.empty:
        return {
            "annual_return": np.nan, "annual_volatility": np.nan, "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan, "max_drawdown": np.nan, "calmar_ratio": np.nan,
            "var_05": np.nan, "skewness": np.nan,
        }
    n = len(ret)
    if rf is not None:
        rf = rf.reindex(ret.index).fillna(0)
        excess = ret - rf
    else:
        excess = ret
    ann_ret = (1 + ret).prod() ** (periods_per_year / n) - 1 if n else np.nan
    ann_vol = ret.std() * np.sqrt(periods_per_year) if n > 1 else np.nan
    sharpe = (excess.mean() / ret.std() * np.sqrt(periods_per_year)) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std()
    sortino = (excess.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan
    cum = (1 + ret).cumprod()
    dd = (cum.cummax() - cum) / cum.cummax()
    max_dd = dd.max()
    calmar = ann_ret / (-max_dd) if max_dd != 0 else np.nan
    var_05 = float(ret.quantile(0.05)) if n >= 20 else np.nan
    skewness = float(ret.skew()) if n > 2 else np.nan
    return {
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "var_05": var_05,
        "skewness": skewness,
    }


def compute_subperiod_metrics(
    returns: pd.Series,
    window_years: float = 3.0,
    *,
    periods_per_year: int = 252,
) -> List[Dict[str, float]]:
    """Rolling subperiod stats (e.g. 3-year Sharpe) for regime/stability (syllabus)."""
    ret = returns.dropna()
    if ret.empty or len(ret) < max(60, int(periods_per_year * window_years * 0.5)):
        return []
    window = int(periods_per_year * window_years)
    out = []
    for i in range(window, len(ret) + 1):
        sub = ret.iloc[i - window:i]
        if len(sub) < window // 2:
            continue
        m = compute_performance_metrics(sub, periods_per_year=periods_per_year)
        m["start"] = sub.index[0]
        m["end"] = sub.index[-1]
        out.append(m)
    return out


def compute_drawdowns(returns: pd.Series) -> pd.DataFrame:
    """DataFrame with date, cumulative, running_max, drawdown."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (running_max - cum) / running_max
    return pd.DataFrame({"cumulative": cum, "running_max": running_max, "drawdown": dd})

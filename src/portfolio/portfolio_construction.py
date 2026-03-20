"""Build long (top join prob) / short (top leave prob) portfolios; benchmark = perfect foresight."""
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.utils.config_loader import load_config, get_section
from src.portfolio.weighting_schemes import equal_weight, probability_weight, risk_parity_weight, rank_weight


def build_long_short_portfolio(
    join_scores: pd.DataFrame,
    leave_scores: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict | None = None,
    *,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
    top_decile: float = 0.10,
    weighting: str = "equal",
    gross_exposure: float = 2.0,
    net_exposure: float = 0.0,
    model_name: str = "random_forest",
) -> pd.DataFrame:
    """Target weights per (date, permno): long top join prob, short top leave prob.

    join_scores: columns date, permno, p_join_<model>
    leave_scores: columns date, permno, p_leave_<model>
    panel: daily panel with date, permno, ret (for risk_parity vol if needed).
    Returns: DataFrame with date, permno, weight.
    """
    cfg = config or load_config()
    top_decile = top_decile or get_section(cfg, "backtest", "top_decile") or 0.10
    weighting = weighting or get_section(cfg, "backtest", "weighting") or "equal"
    gross_exposure = gross_exposure or get_section(cfg, "backtest", "gross_exposure") or 2.0
    net_exposure = net_exposure or get_section(cfg, "backtest", "net_exposure") or 0.0

    p_join_col = f"p_join_{model_name}"
    p_leave_col = f"p_leave_{model_name}"
    if p_join_col not in join_scores.columns:
        p_join_col = [c for c in join_scores.columns if c.startswith("p_join_")][0] if any(c.startswith("p_join_") for c in join_scores.columns) else None
    if p_leave_col not in leave_scores.columns:
        p_leave_col = [c for c in leave_scores.columns if c.startswith("p_leave_")][0] if any(c.startswith("p_leave_") for c in leave_scores.columns) else None
    if not p_join_col or not p_leave_col:
        return pd.DataFrame(columns=["date", "permno", "weight"])

    if rebalance_dates is None:
        ud = pd.to_datetime(join_scores["date"].unique()).sort_values()
        df_u = pd.DataFrame({"date": ud})
        df_u["ym"] = df_u["date"].dt.to_period("M")
        rebalance_dates = df_u.groupby("ym")["date"].first().values

    rows = []
    for rb_date in rebalance_dates:
        j = join_scores[join_scores["date"] == rb_date].set_index("permno")[p_join_col]
        lv = leave_scores[leave_scores["date"] == rb_date].set_index("permno")[p_leave_col]
        if weighting == "probability":
            w_long = probability_weight(j, top_decile, gross_target=gross_exposure / 2)
            w_short = -probability_weight(lv, top_decile, gross_target=gross_exposure / 2)
        elif weighting == "risk_parity":
            vol = panel[panel["date"] <= rb_date].groupby("permno")["ret"].std().reindex(j.index).fillna(0.01)
            w_long = risk_parity_weight(j, vol, top_decile, gross_target=gross_exposure / 2)
            vol_l = panel[panel["date"] <= rb_date].groupby("permno")["ret"].std().reindex(lv.index).fillna(0.01)
            w_short = -risk_parity_weight(lv, vol_l, top_decile, gross_target=gross_exposure / 2)
        elif weighting == "rank":
            w_long = rank_weight(j, top_decile, gross_target=gross_exposure / 2)
            w_short = -rank_weight(lv, top_decile, gross_target=gross_exposure / 2)
        else:
            w_long = equal_weight(j, top_decile, gross_target=gross_exposure / 2)
            w_short = -equal_weight(lv, top_decile, gross_target=gross_exposure / 2)
        for permno, w in w_long.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})
        for permno, w in w_short.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})

    return pd.DataFrame(rows)


def build_perfect_foresight_portfolio(
    panel: pd.DataFrame,
    events_join: pd.DataFrame,
    events_leave: pd.DataFrame,
    config: dict | None = None,
    *,
    forward_days: int = 63,
    rebalance_freq: str = "monthly",
    top_decile: float = 0.10,
) -> pd.DataFrame:
    """Benchmark: long stocks that will join in next forward_days, short that will leave. Uses realized labels."""
    # For each rebalance date, label = 1 if join in next forward_days
    panel = panel.sort_values(["permno", "date"]).copy()
    panel["_future_join"] = panel.groupby("permno")["is_sp500"].shift(-forward_days)
    panel["_future_leave"] = panel.groupby("permno")["is_sp500"].shift(-forward_days)
    panel["label_join_ff"] = ((panel["is_sp500"] == False) & (panel["_future_join"] == True)).astype(int)
    panel["label_leave_ff"] = ((panel["is_sp500"] == True) & (panel["_future_leave"] == False)).astype(int)

    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"].unique())).sort_values()
    if rebalance_freq == "monthly":
        df_d = pd.DataFrame({"date": dates})
        df_d["ym"] = df_d["date"].dt.to_period("M")
        rebalance_dates = df_d.groupby("ym")["date"].first().values
    else:
        rebalance_dates = dates

    rows = []
    for rb in rebalance_dates:
        sub = panel[panel["date"] == rb]
        j = sub.set_index("permno")["label_join_ff"]
        lv = sub.set_index("permno")["label_leave_ff"]
        w_long = equal_weight(j, top_decile, gross_target=1.0)
        w_short = -equal_weight(lv, top_decile, gross_target=1.0)
        for permno, w in w_long.items():
            if w != 0:
                rows.append({"date": rb, "permno": permno, "weight": w})
        for permno, w in w_short.items():
            if w != 0:
                rows.append({"date": rb, "permno": permno, "weight": w})
    return pd.DataFrame(rows)

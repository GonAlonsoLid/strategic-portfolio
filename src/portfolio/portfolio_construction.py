"""Build long (top join prob) / short (top leave prob) portfolios; benchmark = perfect foresight."""
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.utils.config_loader import load_config, get_section
from src.portfolio.weighting_schemes import equal_weight, probability_weight, risk_parity_weight, rank_weight, threshold_weight, topn_weight, signal_risk_weight


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
        ud = pd.DatetimeIndex(pd.to_datetime(join_scores["date"].unique())).sort_values()
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


def build_threshold_portfolio(
    join_scores: pd.DataFrame,
    leave_scores: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
    prob_threshold_join: float = 0.15,
    prob_threshold_leave: float = 0.15,
    weighting: str = "equal",
    gross_exposure: float = 2.0,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """Build long-short portfolio using absolute probability thresholds.

    Instead of selecting the top decile by rank (which includes hundreds
    of near-zero-probability stocks), only trade stocks where the model
    assigns probability above a meaningful threshold.

    Long: stocks with p_join > prob_threshold_join
    Short: stocks with p_leave > prob_threshold_leave
    """
    p_join_col = f"p_join_{model_name}"
    p_leave_col = f"p_leave_{model_name}"
    if p_join_col not in join_scores.columns:
        p_join_col = [c for c in join_scores.columns if c.startswith("p_join_")][0] if any(c.startswith("p_join_") for c in join_scores.columns) else None
    if p_leave_col not in leave_scores.columns:
        p_leave_col = [c for c in leave_scores.columns if c.startswith("p_leave_")][0] if any(c.startswith("p_leave_") for c in leave_scores.columns) else None
    if not p_join_col or not p_leave_col:
        return pd.DataFrame(columns=["date", "permno", "weight"])

    if rebalance_dates is None:
        ud = pd.DatetimeIndex(pd.to_datetime(join_scores["date"].unique())).sort_values()
        df_u = pd.DataFrame({"date": ud})
        df_u["ym"] = df_u["date"].dt.to_period("M")
        rebalance_dates = df_u.groupby("ym")["date"].first().values

    rows = []
    for rb_date in rebalance_dates:
        j = join_scores[join_scores["date"] == rb_date].set_index("permno")[p_join_col]
        lv = leave_scores[leave_scores["date"] == rb_date].set_index("permno")[p_leave_col]

        w_long = threshold_weight(j, prob_threshold_join,
                                  gross_target=gross_exposure / 2, weighting=weighting)
        w_short = -threshold_weight(lv, prob_threshold_leave,
                                    gross_target=gross_exposure / 2, weighting=weighting)

        for permno, w in w_long.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})
        for permno, w in w_short.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})

    return pd.DataFrame(rows)


def build_topn_portfolio(
    join_scores: pd.DataFrame,
    leave_scores: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
    n_long: int = 20,
    n_short: int = 20,
    weighting: str = "equal",
    gross_exposure: float = 2.0,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """Build long-short portfolio selecting top N stocks by probability.

    Calibration-independent: uses rank ordering, not absolute probability.
    With ~20 actual joins/year, n_long=20 concentrates on highest-conviction names.
    """
    p_join_col = f"p_join_{model_name}"
    p_leave_col = f"p_leave_{model_name}"
    if p_join_col not in join_scores.columns:
        p_join_col = [c for c in join_scores.columns if c.startswith("p_join_")][0] if any(c.startswith("p_join_") for c in join_scores.columns) else None
    if p_leave_col not in leave_scores.columns:
        p_leave_col = [c for c in leave_scores.columns if c.startswith("p_leave_")][0] if any(c.startswith("p_leave_") for c in leave_scores.columns) else None
    if not p_join_col or not p_leave_col:
        return pd.DataFrame(columns=["date", "permno", "weight"])

    if rebalance_dates is None:
        ud = pd.DatetimeIndex(pd.to_datetime(join_scores["date"].unique())).sort_values()
        df_u = pd.DataFrame({"date": ud})
        df_u["ym"] = df_u["date"].dt.to_period("M")
        rebalance_dates = df_u.groupby("ym")["date"].first().values

    rows = []
    for rb_date in rebalance_dates:
        j = join_scores[join_scores["date"] == rb_date].set_index("permno")[p_join_col]
        lv = leave_scores[leave_scores["date"] == rb_date].set_index("permno")[p_leave_col]

        w_long = topn_weight(j, n_long, gross_target=gross_exposure / 2, weighting=weighting)
        w_short = -topn_weight(lv, n_short, gross_target=gross_exposure / 2, weighting=weighting)

        for permno, w in w_long.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})
        for permno, w in w_short.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})

    return pd.DataFrame(rows)


def _detect_score_cols(join_scores, leave_scores, model_name):
    """Resolve p_join / p_leave column names."""
    p_join_col = f"p_join_{model_name}"
    p_leave_col = f"p_leave_{model_name}"
    if p_join_col not in join_scores.columns:
        p_join_col = next((c for c in join_scores.columns if c.startswith("p_join_")), None)
    if p_leave_col not in leave_scores.columns:
        p_leave_col = next((c for c in leave_scores.columns if c.startswith("p_leave_")), None)
    return p_join_col, p_leave_col


def _make_rebalance_dates(join_scores, rebalance_dates=None):
    """Monthly first-trading-day rebalance schedule."""
    if rebalance_dates is not None:
        return rebalance_dates
    ud = pd.DatetimeIndex(pd.to_datetime(join_scores["date"].unique())).sort_values()
    df_u = pd.DataFrame({"date": ud})
    df_u["ym"] = df_u["date"].dt.to_period("M")
    return df_u.groupby("ym")["date"].first().values


def build_composite_portfolio(
    join_scores: pd.DataFrame,
    leave_scores: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
    n_long: int = 10,
    n_short: int = 10,
    alpha: float = 0.5,
    beta: float = 0.5,
    weighting: str = "equal",
    gross_exposure: float = 2.0,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """Composite score strategy: combine join and leave signals.

    Long score  = p_join - alpha * p_leave  (penalise stocks with high leave risk)
    Short score = p_leave - beta * p_join   (penalise stocks with high join potential)
    Then select top-N from each composite score.
    """
    p_join_col, p_leave_col = _detect_score_cols(join_scores, leave_scores, model_name)
    if not p_join_col or not p_leave_col:
        return pd.DataFrame(columns=["date", "permno", "weight"])

    rebalance_dates = _make_rebalance_dates(join_scores, rebalance_dates)

    rows = []
    for rb_date in rebalance_dates:
        j = join_scores[join_scores["date"] == rb_date].set_index("permno")[p_join_col]
        lv = leave_scores[leave_scores["date"] == rb_date].set_index("permno")[p_leave_col]

        # Align on common universe
        common = j.index.intersection(lv.index)
        j_c, lv_c = j.reindex(common).fillna(0), lv.reindex(common).fillna(0)

        composite_long = j_c - alpha * lv_c
        composite_short = lv_c - beta * j_c

        # Also include stocks only in join universe (no leave penalty)
        only_join = j.index.difference(common)
        if len(only_join) > 0:
            composite_long = pd.concat([composite_long, j[only_join]])

        only_leave = lv.index.difference(common)
        if len(only_leave) > 0:
            composite_short = pd.concat([composite_short, lv[only_leave]])

        w_long = topn_weight(composite_long, n_long, gross_target=gross_exposure / 2, weighting=weighting)
        w_short = -topn_weight(composite_short, n_short, gross_target=gross_exposure / 2, weighting=weighting)

        for permno, w in w_long.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})
        for permno, w in w_short.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})

    return pd.DataFrame(rows)


def build_volscaled_portfolio(
    join_scores: pd.DataFrame,
    leave_scores: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
    n_long: int = 10,
    n_short: int = 10,
    gamma: float = 0.3,
    vol_window: int = 63,
    gross_exposure: float = 2.0,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """Vol-scaled portfolio: position size inversely proportional to volatility.

    w_i = score^gamma / vol_i. Targets equal risk contribution per position,
    reducing dependence on a few high-volatility names.
    """
    p_join_col, p_leave_col = _detect_score_cols(join_scores, leave_scores, model_name)
    if not p_join_col or not p_leave_col:
        return pd.DataFrame(columns=["date", "permno", "weight"])

    rebalance_dates = _make_rebalance_dates(join_scores, rebalance_dates)

    # Pre-compute rolling volatility from panel
    panel_sorted = panel.sort_values(["permno", "date"])

    rows = []
    for rb_date in rebalance_dates:
        j = join_scores[join_scores["date"] == rb_date].set_index("permno")[p_join_col]
        lv = leave_scores[leave_scores["date"] == rb_date].set_index("permno")[p_leave_col]

        # Get trailing volatility for all stocks as of rebalance date
        mask = (panel_sorted["date"] <= rb_date) & (panel_sorted["date"] > rb_date - pd.Timedelta(days=vol_window * 2))
        vol = panel_sorted.loc[mask].groupby("permno")["ret"].std()

        w_long = signal_risk_weight(j, vol.reindex(j.index).fillna(0.01), n_long,
                                     gamma=gamma, gross_target=gross_exposure / 2)
        w_short = -signal_risk_weight(lv, vol.reindex(lv.index).fillna(0.01), n_short,
                                       gamma=gamma, gross_target=gross_exposure / 2)

        for permno, w in w_long.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})
        for permno, w in w_short.items():
            if w != 0:
                rows.append({"date": rb_date, "permno": permno, "weight": w})

    return pd.DataFrame(rows)


def build_momentum_filtered_portfolio(
    join_scores: pd.DataFrame,
    leave_scores: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
    n_long: int = 10,
    n_short: int = 10,
    mom_window: int = 21,
    weighting: str = "equal",
    gross_exposure: float = 2.0,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """Momentum-filtered portfolio: only enter when price trend confirms signal.

    Long: high p_join AND positive recent momentum (stock trending up).
    Short: high p_leave AND negative recent momentum (stock trending down).
    Filters out stocks where model and price action disagree.
    """
    p_join_col, p_leave_col = _detect_score_cols(join_scores, leave_scores, model_name)
    if not p_join_col or not p_leave_col:
        return pd.DataFrame(columns=["date", "permno", "weight"])

    rebalance_dates = _make_rebalance_dates(join_scores, rebalance_dates)

    # Pre-compute rolling returns for momentum
    panel_sorted = panel.sort_values(["permno", "date"])

    rows = []
    for rb_date in rebalance_dates:
        j = join_scores[join_scores["date"] == rb_date].set_index("permno")[p_join_col]
        lv = leave_scores[leave_scores["date"] == rb_date].set_index("permno")[p_leave_col]

        # Compute trailing momentum
        mask = (panel_sorted["date"] <= rb_date) & (panel_sorted["date"] > rb_date - pd.Timedelta(days=mom_window * 2))
        sub = panel_sorted.loc[mask]
        mom = sub.groupby("permno")["ret"].apply(lambda x: (1 + x).prod() - 1)

        # Filter: long only if momentum > 0
        j_mom = mom.reindex(j.index).fillna(0)
        j_filtered = j[j_mom > 0]

        # Filter: short only if momentum < 0
        lv_mom = mom.reindex(lv.index).fillna(0)
        lv_filtered = lv[lv_mom < 0]

        # Fallback: if filter removes all candidates, use unfiltered top-N
        if len(j_filtered) < 3:
            j_filtered = j
        if len(lv_filtered) < 3:
            lv_filtered = lv

        w_long = topn_weight(j_filtered, n_long, gross_target=gross_exposure / 2, weighting=weighting)
        w_short = -topn_weight(lv_filtered, n_short, gross_target=gross_exposure / 2, weighting=weighting)

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

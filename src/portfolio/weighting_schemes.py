"""Portfolio weighting: equal, probability-weighted, risk-parity, rank (academic)."""
from typing import Optional

import pandas as pd
import numpy as np


def rank_weight(
    scores: pd.Series,
    top_decile: float = 0.10,
    *,
    gross_target: float = 1.0,
) -> pd.Series:
    """Academic rank weighting: w_i = (rank(S_i) - (N+1)/2) / (N-1), dollar-neutral.
    Long top decile, short bottom decile; then scale to gross_target (half each side)."""
    r = scores.rank(method="first")
    n = r.notna().sum()
    if n < 2:
        return scores * 0
    # Standardized rank in [-0.5, 0.5]; we want long top decile, short bottom
    q_hi = scores.quantile(1 - top_decile)
    q_lo = scores.quantile(top_decile)
    w = pd.Series(0.0, index=scores.index)
    w[scores >= q_hi] = 1.0
    w[scores <= q_lo] = -1.0
    if w.abs().sum() > 0:
        w = w / w.abs().sum() * gross_target
    return w


def equal_weight(
    scores: pd.Series,
    top_decile: float = 0.10,
    *,
    gross_target: float = 1.0,
) -> pd.Series:
    """Equal weight within top decile; normalize so sum(abs(w)) = gross_target."""
    q = scores.quantile(1 - top_decile)
    mask = scores >= q
    w = mask.astype(float)
    w = w / w.sum() * gross_target if w.sum() > 0 else w
    return w


def probability_weight(
    scores: pd.Series,
    top_decile: float = 0.10,
    *,
    gross_target: float = 1.0,
) -> pd.Series:
    """Weights proportional to scores within top decile; normalize to gross_target."""
    q = scores.quantile(1 - top_decile)
    mask = scores >= q
    w = (scores * mask).fillna(0)
    if w.sum() > 0:
        w = w / w.sum() * gross_target
    return w


def risk_parity_weight(
    scores: pd.Series,
    volatility: pd.Series,
    top_decile: float = 0.10,
    *,
    gross_target: float = 1.0,
) -> pd.Series:
    """Inverse-volatility weights within top decile. volatility is per-name (e.g. rolling std)."""
    q = scores.quantile(1 - top_decile)
    mask = scores >= q
    inv_vol = (1.0 / volatility.replace(0, np.nan)).fillna(0)
    w = (mask * inv_vol).fillna(0)
    if w.sum() > 0:
        w = w / w.sum() * gross_target
    return w


def threshold_weight(
    scores: pd.Series,
    prob_threshold: float = 0.15,
    *,
    gross_target: float = 1.0,
    weighting: str = "equal",
) -> pd.Series:
    """Select stocks exceeding absolute probability threshold.

    Unlike quantile-based selection, this only picks stocks the model
    is genuinely confident about (p > threshold), avoiding the problem
    of selecting hundreds of near-zero-probability stocks.

    Args:
        scores: Predicted probabilities per stock.
        prob_threshold: Minimum probability to include (absolute, not quantile).
        gross_target: Target gross exposure for this leg.
        weighting: 'equal' or 'probability' (prop to score).
    """
    mask = scores >= prob_threshold
    if mask.sum() == 0:
        return pd.Series(0.0, index=scores.index)
    if weighting == "probability":
        w = (scores * mask).fillna(0)
    else:
        w = mask.astype(float)
    if w.sum() > 0:
        w = w / w.sum() * gross_target
    return w


def topn_weight(
    scores: pd.Series,
    n: int = 20,
    *,
    gross_target: float = 1.0,
    weighting: str = "equal",
) -> pd.Series:
    """Select top N stocks by score, independent of probability calibration.

    This is the standard quant approach: rank by signal, take the top N.
    Calibration-independent — works even when model probabilities are
    distorted by scale_pos_weight or class imbalance corrections.

    Args:
        scores: Predicted probabilities (or any signal) per stock.
        n: Number of top stocks to select.
        gross_target: Target gross exposure for this leg.
        weighting: 'equal' or 'probability' (prop to score).
    """
    if len(scores) == 0:
        return pd.Series(dtype=float)
    n = min(n, len(scores))
    top_idx = scores.nlargest(n).index
    w = pd.Series(0.0, index=scores.index)
    if weighting == "probability":
        w[top_idx] = scores[top_idx]
    else:
        w[top_idx] = 1.0
    if w.sum() > 0:
        w = w / w.sum() * gross_target
    return w

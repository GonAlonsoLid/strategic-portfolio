"""Backtest engine: daily portfolio returns from target weights, with turnover and costs."""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from src.backtesting.transaction_costs import estimate_costs


class Backtester:
    """Compute portfolio returns from target weights and daily panel."""

    def __init__(
        self,
        panel: pd.DataFrame,
        *,
        date_col: str = "date",
        permno_col: str = "permno",
        ret_col: str = "ret",
        transaction_cost_bps: float = 10,
    ):
        self.panel = panel.sort_values([date_col, permno_col])
        self.date_col = date_col
        self.permno_col = permno_col
        self.ret_col = ret_col
        self.cost_bps = transaction_cost_bps

    def run_backtest(
        self,
        target_weights: pd.DataFrame,
        *,
        weight_date_col: str = "date",
        weight_permno_col: str = "permno",
        weight_val_col: str = "weight",
    ) -> Dict[str, pd.Series]:
        """Run backtest. target_weights: rebalance dates and weights; forward-fill to daily.

        Returns:
            returns: daily portfolio return
            turnover: daily turnover (sum of abs weight change)
            gross_exposure, net_exposure: time series
            transaction_costs: cost in return space per day
        """
        target_weights = target_weights.rename(columns={
            weight_date_col: "date",
            weight_permno_col: "permno",
            weight_val_col: "weight",
        })
        target_weights["date"] = pd.to_datetime(target_weights["date"]).dt.normalize()
        rebal_dates = sorted(target_weights["date"].unique())
        all_dates = pd.DatetimeIndex(pd.to_datetime(self.panel[self.date_col].unique())).sort_values()
        # Pivot to (rebal_date x permno), then reindex to all_dates and ffill.
        # At each rebalance date, unmentioned stocks must be 0 (closed),
        # not NaN (which ffill would propagate from previous rebalance).
        w_pivot = target_weights.pivot_table(index="date", columns="permno", values="weight", aggfunc="first")
        # Fill NaN with 0 at rebalance dates so positions are explicitly closed
        rebal_set = set(rebal_dates)
        for d in w_pivot.index:
            if d in rebal_set:
                w_pivot.loc[d] = w_pivot.loc[d].fillna(0)
        w_daily = w_pivot.reindex(all_dates).ffill().fillna(0)

        # Portfolio return and turnover per day
        returns = []
        turnover = []
        gross_exp = []
        net_exp = []
        prev_w = pd.Series(dtype=float)
        for i, d in enumerate(all_dates):
            w = w_daily.loc[d] if d in w_daily.index else prev_w
            if prev_w.empty:
                prev_w = w.fillna(0)
            day_ret = self.panel[self.panel[self.date_col] == d].set_index(self.permno_col)[self.ret_col]
            common = w.index.union(day_ret.index).drop_duplicates()
            port_ret = (w.reindex(common).fillna(0) * day_ret.reindex(common).fillna(0)).sum()
            returns.append(port_ret)
            gross_exp.append(w.abs().sum())
            net_exp.append(w.sum())
            turn = (w.reindex(common).fillna(0) - prev_w.reindex(common).fillna(0)).abs().sum()
            turnover.append(turn)
            prev_w = w.fillna(0)

        ret_series = pd.Series(returns, index=all_dates)
        turn_series = pd.Series(turnover, index=all_dates)
        cost_series = estimate_costs(turn_series, cost_bps=self.cost_bps)
        net_ret = ret_series - cost_series
        return {
            "returns": net_ret,
            "gross_returns": ret_series,
            "turnover": turn_series,
            "transaction_costs": cost_series,
            "gross_exposure": pd.Series(gross_exp, index=all_dates),
            "net_exposure": pd.Series(net_exp, index=all_dates),
        }

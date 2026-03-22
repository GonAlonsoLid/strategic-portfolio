"""Shared plotting style and helpers for CAR, returns, drawdowns, factor loadings."""
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """Set consistent matplotlib/seaborn style for the project."""
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    sns.set_style("whitegrid")


def plot_car(
    car_by_rel_day: pd.DataFrame,
    *,
    event_types: list[str] | None = None,
    title: str = "Cumulative Abnormal Returns",
    save_path: str | Path | None = None,
) -> None:
    """Plot average CAR vs relative event day for joiners/leavers.

    Args:
        car_by_rel_day: DataFrame with index = rel_day and columns = event_type (e.g. ADD, DEL).
        event_types: Which columns to plot; default all.
        title: Plot title.
        save_path: If set, save figure here.
    """
    set_plot_style()
    if event_types is None:
        event_types = [c for c in car_by_rel_day.columns if car_by_rel_day[c].dtype in ("float64", "float32")]
    fig, ax = plt.subplots()
    for et in event_types:
        if et in car_by_rel_day.columns:
            ax.plot(car_by_rel_day.index, car_by_rel_day[et], label=et, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Relative event day")
    ax.set_ylabel("CAR")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_returns(
    returns: pd.Series,
    *,
    title: str = "Cumulative returns",
    save_path: str | Path | None = None,
) -> None:
    """Plot cumulative gross return (1 + r).cumprod()."""
    set_plot_style()
    cum = (1 + returns).cumprod()
    fig, ax = plt.subplots()
    ax.plot(cum.index, cum.values, linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_comparison(
    series: dict,
    *,
    title: str = "Strategy comparison: cumulative returns",
    save_path=None,
) -> None:
    """Overlay cumulative returns for multiple strategies.

    Args:
        series: dict mapping strategy label (str) to daily return pd.Series.
        title: Plot title.
        save_path: If set, save figure here.
    """
    set_plot_style()
    fig, ax = plt.subplots()
    for label, ret in series.items():
        cum = (1 + ret).cumprod()
        ax.plot(cum.index, cum.values, linewidth=2, label=label)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(
    returns: pd.Series,
    *,
    title: str = "Drawdown",
    save_path: str | Path | None = None,
) -> None:
    """Plot drawdown series (cummax - cum) / cummax."""
    set_plot_style()
    cum = (1 + returns).cumprod()
    dd = (cum.cummax() - cum) / cum.cummax()
    fig, ax = plt.subplots()
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_factor_loadings(
    loadings: pd.Series,
    *,
    title: str = "Factor loadings",
    save_path: str | Path | None = None,
) -> None:
    """Bar plot of factor loadings (e.g. from Fama-French regression)."""
    set_plot_style()
    fig, ax = plt.subplots()
    loadings.plot(kind="bar", ax=ax)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Loading")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_heatmap(
    robustness_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    *,
    row_col: str = "holding_period_months",
    col_col: str = "prob_threshold",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Heatmap of a metric over the robustness grid (holding period x threshold)."""
    set_plot_style()
    pivot = robustness_df.pivot_table(index=row_col, columns=col_col, values=metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"label": metric})
    ax.set_title(title or f"Robustness: {metric}")
    ax.set_ylabel("Holding period (months)")
    ax.set_xlabel("Probability threshold")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_turnover(
    turnover: pd.Series,
    *,
    title: str = "Portfolio turnover",
    save_path: str | Path | None = None,
) -> None:
    """Plot daily turnover over time."""
    set_plot_style()
    fig, ax = plt.subplots()
    rolling = turnover.rolling(21).mean()
    ax.plot(rolling.index, rolling.values, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily turnover (21d avg)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_exposure(
    gross_exposure: pd.Series,
    net_exposure: pd.Series,
    *,
    title: str = "Portfolio exposure",
    save_path: str | Path | None = None,
) -> None:
    """Plot gross and net exposure over time."""
    set_plot_style()
    fig, ax = plt.subplots()
    ax.plot(gross_exposure.index, gross_exposure.values, label="Gross", linewidth=1.5)
    ax.plot(net_exposure.index, net_exposure.values, label="Net", linewidth=1.5, alpha=0.7)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importance: pd.Series,
    *,
    title: str = "Feature importance",
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> None:
    """Horizontal bar plot of feature importance (e.g. from tree model)."""
    set_plot_style()
    top = importance.nlargest(top_n)
    fig, ax = plt.subplots()
    top.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

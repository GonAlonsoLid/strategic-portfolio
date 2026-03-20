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

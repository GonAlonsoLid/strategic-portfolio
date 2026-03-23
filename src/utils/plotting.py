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


def plot_annual_returns(
    returns_dict: dict,
    *,
    title: str = "Annual returns by year",
    save_path: str | Path | None = None,
) -> None:
    """Grouped bar chart of annual returns for multiple strategies."""
    set_plot_style()
    records = []
    for label, ret in returns_dict.items():
        annual = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
        for yr, val in annual.items():
            records.append({"Year": yr, "Strategy": label, "Return": val})
    df = pd.DataFrame(records)
    if df.empty:
        return

    years = sorted(df["Year"].unique())
    strategies = list(returns_dict.keys())
    n_strats = len(strategies)
    bar_width = 0.8 / n_strats
    fig, ax = plt.subplots(figsize=(max(12, len(years) * 0.8), 6))

    for i, strat in enumerate(strategies):
        sub = df[df["Strategy"] == strat].set_index("Year")["Return"]
        positions = [years.index(y) + i * bar_width for y in sub.index]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub.values]
        ax.bar(positions, sub.values, bar_width * 0.9, label=strat, color=colors, alpha=0.8)

    ax.set_xticks([y + bar_width * (n_strats - 1) / 2 for y in range(len(years))])
    ax.set_xticklabels(years, rotation=45)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Annual return")
    ax.set_title(title)
    if n_strats > 1:
        ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_metrics(
    returns_dict: dict,
    *,
    window: int = 252,
    title: str = "Rolling 1-year metrics",
    save_path: str | Path | None = None,
) -> None:
    """2-panel plot: rolling Sharpe and rolling annual return for multiple strategies."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label, ret in returns_dict.items():
        rolling_mean = ret.rolling(window).mean() * 252
        rolling_vol = ret.rolling(window).std() * (252 ** 0.5)
        rolling_sharpe = rolling_mean / rolling_vol

        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5, label=label, alpha=0.8)
        ax2.plot(rolling_mean.index, rolling_mean.values, linewidth=1.5, label=label, alpha=0.8)

    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Rolling Sharpe (1y)")
    ax1.set_title(title)
    ax1.legend(fontsize=9)

    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Rolling annual return (1y)")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

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

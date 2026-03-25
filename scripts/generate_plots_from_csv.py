"""Generate missing plots from existing CSV results (no backtest re-run needed).

Produces:
  - results/figures/annual_returns.png   (rolling 3-year return over time)
  - results/figures/rolling_metrics.png  (rolling Sharpe, drawdown, volatility)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.plotting import set_plot_style

BASE = Path(__file__).resolve().parent.parent
TABLES = BASE / "results" / "tables"
FIGURES = BASE / "results" / "figures"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(TABLES / "backtest_subperiod_3y.csv")
    sub["midpoint"] = pd.to_datetime(sub["start"]) + (
        pd.to_datetime(sub["end"]) - pd.to_datetime(sub["start"])
    ) / 2

    set_plot_style()

    # ── 1. Rolling 3-year annualised return over time ──────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        sub["midpoint"],
        sub["annual_return"],
        width=60,
        color=["#2ecc71" if v >= 0 else "#e74c3c" for v in sub["annual_return"]],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Annualised return")
    ax.set_title("Best predictive strategy: rolling 3-year annualised return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    fig.savefig(FIGURES / "annual_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES / 'annual_returns.png'}")

    # ── 2. Rolling metrics panel (Sharpe, max drawdown, volatility) ────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Sharpe ratio
    axes[0].plot(sub["midpoint"], sub["sharpe_ratio"], "o-", color="#3498db", linewidth=2, markersize=4)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Sharpe ratio")
    axes[0].set_title("Best predictive strategy: rolling 3-year risk metrics")

    # Max drawdown (negative convention)
    axes[1].fill_between(sub["midpoint"], sub["max_drawdown"], 0, alpha=0.4, color="#e74c3c")
    axes[1].plot(sub["midpoint"], sub["max_drawdown"], "o-", color="#e74c3c", linewidth=2, markersize=4)
    axes[1].set_ylabel("Max drawdown")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Volatility
    axes[2].plot(sub["midpoint"], sub["annual_volatility"], "o-", color="#9b59b6", linewidth=2, markersize=4)
    axes[2].set_ylabel("Annualised volatility")
    axes[2].set_xlabel("Window midpoint")
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].xaxis.set_major_locator(mdates.YearLocator())

    fig.tight_layout()
    fig.savefig(FIGURES / "rolling_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES / 'rolling_metrics.png'}")

    print("\nDone. Re-run generate_report.py to embed in HTML.")


if __name__ == "__main__":
    main()

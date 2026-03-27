"""Download Fama-French factors and run 4-factor regression on best strategy returns.

Downloads daily FF3 + Momentum from Kenneth French's website,
runs OLS regression, and saves results to results/tables/factor_regression.csv.

Usage:
  python scripts/run_factor_regression.py
"""
import sys
from pathlib import Path
import io
import zipfile
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import statsmodels.api as sm

BASE = Path(__file__).resolve().parent.parent
TABLES = BASE / "results" / "tables"


def download_ff_daily(url):
    """Download a daily factor CSV from Kenneth French's website."""
    try:
        resp = urllib.request.urlopen(url, timeout=30)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download factor data from {url}: {exc}") from exc
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    fname = z.namelist()[0]
    raw = z.read(fname).decode("utf-8")
    lines = raw.strip().split("\n")
    # Find data start (first line starting with 8-digit date)
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and len(stripped.split(",")[0].strip()) == 8:
            start = i
            break
    # Find data end
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or not stripped[0].isdigit():
            end = i
            break
    return "\n".join(lines[start:end])


def load_factors():
    """Download and merge FF3 + Momentum daily factors."""
    print("Downloading Fama-French 3-factor daily data...")
    ff3_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    ff3_str = download_ff_daily(ff3_url)
    ff3 = pd.read_csv(io.StringIO(ff3_str), header=None, names=["date", "MKT_RF", "SMB", "HML", "RF"])

    print("Downloading Momentum factor daily data...")
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    mom_str = download_ff_daily(mom_url)
    mom = pd.read_csv(io.StringIO(mom_str), header=None, names=["date", "MOM"])

    # Parse dates
    ff3["date"] = pd.to_datetime(ff3["date"].astype(str).str.strip(), format="%Y%m%d")
    mom["date"] = pd.to_datetime(mom["date"].astype(str).str.strip(), format="%Y%m%d")

    # Merge
    factors = ff3.merge(mom, on="date", how="inner").set_index("date")

    # Convert from percent to decimal
    for col in ["MKT_RF", "SMB", "HML", "RF", "MOM"]:
        factors[col] = factors[col] / 100.0

    print(f"  Factor data: {factors.index.min().date()} to {factors.index.max().date()} ({len(factors)} days)")
    return factors


def _load_common():
    """Load panel, scores, and backtester shared across strategies."""
    from src.utils.config_loader import load_config
    from src.data.load_data import load_config_paths
    from src.backtesting.backtester import Backtester

    cfg = load_config()
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent
    interim = base / paths.get("interim", "data/interim")
    processed = base / paths.get("processed", "data/processed")

    panel = pd.read_parquet(interim / "daily_panel.parquet")
    join_scores = pd.read_parquet(processed / "join_scores.parquet")
    leave_scores = pd.read_parquet(processed / "leave_scores.parquet")

    p_cols = [c for c in join_scores.columns if c.startswith("p_join_")]
    model_name = p_cols[0].replace("p_join_", "") if p_cols else "xgboost"

    signal_start = min(join_scores["date"].min(), leave_scores["date"].min())
    bt = Backtester(panel, transaction_cost_bps=10)

    return cfg, panel, join_scores, leave_scores, model_name, signal_start, bt


def _trim_returns(result, signal_start):
    """Trim backtest returns to signal start date."""
    ret = result["returns"]
    return ret[ret.index >= signal_start]


def load_strategy_returns():
    """Reconstruct best strategy returns from backtest."""
    from src.portfolio.portfolio_construction import build_long_short_portfolio

    cfg, panel, join_scores, leave_scores, model_name, signal_start, bt = _load_common()

    print(f"Building quantile strategy returns (model={model_name})...")
    weights = build_long_short_portfolio(
        join_scores, leave_scores, panel, config=cfg,
        top_decile=0.10, weighting="equal", model_name=model_name,
    )

    result = bt.run_backtest(weights)
    return _trim_returns(result, signal_start)


def load_composite_returns():
    """Reconstruct best composite strategy (Composite-5, alpha=0.25) returns."""
    from src.portfolio.portfolio_construction import build_composite_portfolio

    cfg, panel, join_scores, leave_scores, model_name, signal_start, bt = _load_common()

    print(f"Building Composite-5 (a=0.25) returns (model={model_name})...")
    weights = build_composite_portfolio(
        join_scores, leave_scores, panel,
        n_long=5, n_short=5, alpha=0.25, beta=0.25,
        weighting="equal", gross_exposure=2.0, model_name=model_name,
    )

    result = bt.run_backtest(weights)
    return _trim_returns(result, signal_start)


def run_regression(returns, factors):
    """Run Fama-French 4-factor regression."""
    factor_cols = ["MKT_RF", "SMB", "HML", "MOM"]
    common = returns.index.intersection(factors.index)
    print(f"  Overlapping dates: {len(common)}")

    y = returns.reindex(common).dropna()
    rf = factors.reindex(common)["RF"]
    y_excess = y - rf  # excess returns

    X = factors.reindex(common)[factor_cols].dropna(how="any")
    common2 = y_excess.index.intersection(X.index)
    y_excess = y_excess.loc[common2]
    X = sm.add_constant(X.loc[common2])

    model = sm.OLS(y_excess, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})

    print(f"\n{model.summary()}\n")

    # Save results
    rows = []
    rows.append({
        "factor": "Alpha (annualised)",
        "coefficient": model.params.get("const", np.nan) * 252,
        "t_stat": model.tvalues.get("const", np.nan),
        "p_value": model.pvalues.get("const", np.nan),
    })
    for col in factor_cols:
        rows.append({
            "factor": col,
            "coefficient": model.params.get(col, np.nan),
            "t_stat": model.tvalues.get(col, np.nan),
            "p_value": model.pvalues.get(col, np.nan),
        })
    rows.append({
        "factor": "R-squared",
        "coefficient": model.rsquared,
        "t_stat": np.nan,
        "p_value": np.nan,
    })

    df = pd.DataFrame(rows)
    return df


def main():
    TABLES.mkdir(parents=True, exist_ok=True)

    factors = load_factors()

    # 1. Quantile strategy (best risk-adjusted)
    print("\n── Quantile strategy ──")
    returns = load_strategy_returns()
    results = run_regression(returns, factors)
    out_path = TABLES / "factor_regression.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(results.to_string(index=False))

    # 2. Composite-5 strategy (best composite, beats omniscient in return)
    print("\n── Composite-5 (a=0.25) strategy ──")
    comp_returns = load_composite_returns()
    comp_results = run_regression(comp_returns, factors)
    comp_path = TABLES / "factor_regression_composite.csv"
    comp_results.to_csv(comp_path, index=False)
    print(f"\nSaved to {comp_path}")
    print(comp_results.to_string(index=False))


if __name__ == "__main__":
    main()

# S&P 500 Index Arbitrage Research

Research-grade quantitative project to study whether investors can profit from predicting S&P 500 index inclusions and exclusions.

## Research goal

- **Perfect foresight benchmark**: theoretical upper bound on strategy performance.
- **Predictive models**: join/leave probability models under uncertainty; backtest long top-join / short top-leave.

## Data (in `data/raw`)

- **SPX_index leavers & joiners_17-Feb-2026.xlsx**: S&P 500 additions and deletions (1994–Feb 2026).
- **daily.csv**: CRSP-style daily stock data (prices, returns, volume, market cap). Large file; read in chunks.

## Setup

```bash
pip install -r requirements.txt
```

## Project structure

- `config/config.yaml`: paths and parameters (event study, features, models, backtest).
- `src/data`: load events (Excel) and daily prices (CSV); build daily panel with index membership.
- `src/events`: event study (CAR, volume, volatility).
- `src/features`: rolling features and joiner/leaver labels.
- `src/models`: join/leave prediction (logistic, RF, GB, XGBoost).
- `src/portfolio`: portfolio construction (equal, probability, risk-parity weights).
- `src/backtesting`: backtester with transaction costs.
- `src/evaluation`: performance metrics and Fama–French factor attribution.
- `scripts/`: `run_event_study.py`, `train_models.py`, `run_backtest.py`.
- `notebooks/`: event study and feature exploration.
- `results/figures`, `results/tables`: outputs.

## Main commands

```bash
# Event study only
python scripts/run_event_study.py

# Train join/leave models and save scores
python scripts/train_models.py

# Full pipeline: load data, features, models, portfolios, backtest, results
python scripts/run_backtest.py
```

## Reproducibility

Set `random_state` in `config/config.yaml` (under `models`). Paths and parameters are config-driven.

## Research design

See [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) for mapping to the Hedge Funds syllabus and underreaction/inattention lecture: features (momentum skip-month, quality proxy), rank weighting, VaR/skewness/subperiod evaluation, and suggested extensions.

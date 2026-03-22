# S&P 500 Joiners and Leavers Portfolio Construction

Quantitative research project evaluating investment strategies that anticipate additions to and deletions from the S&P 500 index.

## Objective

Evaluate whether predictive models can capture the well-documented "index effect" — the abnormal returns around S&P 500 constituent changes — and compare their performance against a perfect foresight benchmark.

## Data

- **S&P 500 constituent changes** (1995 – Feb 2026): additions and deletions with dates
- **CRSP daily stock data**: prices, returns, volume, market capitalization

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
├── data/           Load events and daily prices; build daily panel with membership
├── events/         Event study: cumulative abnormal returns (CAR), volume, volatility
├── features/       Rolling features (momentum, volatility, liquidity) and labels
├── models/         Join/leave prediction (XGBoost with rolling cross-validation)
├── portfolio/      Portfolio construction: equal, probability, risk-parity, top-N
├── backtesting/    Backtest engine with transaction costs
└── evaluation/     Performance metrics, factor attribution, model quality (IC, SHAP)

scripts/
├── run_event_study.py   Event study around index changes
├── train_models.py      Train join/leave prediction models
├── run_backtest.py      Full pipeline: data → models → portfolios → backtest → results
└── run_robustness.py    Robustness sweep: holding period × number of positions

docs/
├── SP500_INSTITUTIONAL_RULES.md   S&P 500 inclusion/exclusion rules and procedures
├── METHODOLOGY.md                 Prediction methodology and portfolio construction
└── RESEARCH_NOTES.md              Feature design rationale and academic references

config/
└── config.yaml   All paths and parameters (features, models, backtest, robustness)
```

## Main Commands

```bash
# Event study (CAR around index additions/deletions)
python scripts/run_event_study.py

# Train join/leave prediction models and save scores
python scripts/train_models.py

# Full backtest pipeline with strategy comparison
python scripts/run_backtest.py

# Robustness sweep (holding period × number of positions grid)
python scripts/run_robustness.py
```

## Output

All results are saved to `results/`:

**Tables** (`results/tables/`):
- `strategy_comparison.csv` — Predictive vs omniscient benchmark metrics
- `robustness_holding_periods.csv` — Robustness grid results
- `model_performance_join.csv`, `model_performance_leave.csv` — Per-fold model metrics
- `backtest_subperiod_3y.csv` — Rolling 3-year subperiod analysis

**Figures** (`results/figures/`):
- `cumulative_returns_comparison.png` — Overlaid strategy cumulative returns
- `drawdown.png` — Drawdown time series
- `turnover.png`, `exposure.png` — Portfolio characteristics
- `robustness_heatmap_sharpe_ratio.png` — Robustness heatmap
- `ic_decay.png`, `shap_importance.png` — Model quality diagnostics

## Documentation

- **[S&P 500 Institutional Rules](docs/SP500_INSTITUTIONAL_RULES.md)**: Eligibility criteria, reconstitution process, and the index effect literature
- **[Methodology](docs/METHODOLOGY.md)**: Prediction approach, portfolio construction rules, position sizing, and robustness framework
- **[Research Notes](docs/RESEARCH_NOTES.md)**: Feature design rationale and academic references

## Reproducibility

All parameters (feature windows, model hyperparameters, backtest settings) are controlled via `config/config.yaml`. Random state is fixed (`random_state: 42`).

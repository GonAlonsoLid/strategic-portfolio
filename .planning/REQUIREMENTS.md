# Requirements: S&P 500 Joiners & Leavers (HEC Paris)

**Defined:** 2026-03-18
**Core Value:** The strategy must demonstrably predict S&P 500 join/leave events and convert that signal into a profitable long-short portfolio with rigorous out-of-sample evidence.

## v1 Requirements

### Analysis — Model Quality

- [ ] **MODEL-01**: IC (Information Coefficient) computed for each model: rank correlation between predicted join/leave probability and realized 21-day forward return
- [ ] **MODEL-02**: IC decay curve by holding horizon (1d, 5d, 21d, 63d) for best model — shows how predictive power fades
- [ ] **MODEL-03**: ICIR (IC / std(IC)) computed per model — industry-standard signal quality metric
- [ ] **MODEL-04**: SHAP feature importance for best model — interpretable attribution of which signals drive predictions
- [ ] **MODEL-05**: Model comparison table: AUC, Brier score, OOS accuracy, portfolio Sharpe per model type

### Analysis — Portfolio & Performance

- [ ] **PERF-01**: Perfect foresight benchmark returns computed and plotted (cumulative)
- [ ] **PERF-02**: Predictive strategy returns (best model) computed and plotted vs. benchmark
- [ ] **PERF-03**: Consolidated performance table: annual return, volatility, Sharpe, Sortino, max drawdown, Calmar, VaR(5%), turnover — for all strategies
- [ ] **PERF-04**: 4-factor attribution: alpha, t(alpha), beta exposures for each strategy
- [ ] **PERF-05**: Bootstrap confidence intervals (1000 iterations) on Sharpe ratio — statistical significance of alpha

### Analysis — Robustness

- [ ] **ROBUST-01**: Holding horizon robustness: repeat backtest for 1d, 5d, 21d, 63d holding periods — Sharpe table
- [ ] **ROBUST-02**: Decile sensitivity: test top 5%, 10%, 20% decile thresholds — Sharpe table
- [ ] **ROBUST-03**: Subperiod stability: 3-year rolling Sharpe plot — shows regime dependency
- [ ] **ROBUST-04**: Transaction cost sensitivity: Sharpe at 0, 10, 20, 30 bps

### Deliverable — Master Notebook

- [ ] **NB-01**: `notebooks/03_master_analysis.ipynb` — complete research narrative end-to-end
- [ ] **NB-02**: Section 1: Executive Summary + S&P 500 institutional rules documentation
- [ ] **NB-03**: Section 2: Event study — CAR plots, announcement effect, anticipation window
- [ ] **NB-04**: Section 3: Feature analysis — IC per feature, SHAP, predictive power
- [ ] **NB-05**: Section 4: Model methodology + OOS evaluation + model comparison table
- [ ] **NB-06**: Section 5: Portfolio construction rules + perfect foresight benchmark
- [ ] **NB-07**: Section 6: Performance evaluation — full metrics table + factor attribution
- [ ] **NB-08**: Section 7: Robustness checks (horizon, decile, costs, subperiods)
- [ ] **NB-09**: Section 8: Economic interpretation — why does this work? limits? capacity?

## v2 Requirements (deferred)

### Extensions (post-submission)

- PEAD features (requires IBES earnings data)
- Size-attention interaction analysis (small cap neglect hypothesis)
- Market beta hedging (rolling beta-neutral portfolio)
- Crowding/capacity analysis with AUM data

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time trading system | Academic project only |
| Accounting features (ROE, ROA) | Compustat not in provided dataset |
| HTML/PDF report export | Jupyter notebook is the deliverable |
| OAuth / deployment | Not applicable |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MODEL-01 to MODEL-05 | Phase 1 | Pending |
| PERF-01 to PERF-05 | Phase 2 | Pending |
| ROBUST-01 to ROBUST-04 | Phase 2 | Pending |
| NB-01 to NB-09 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-18 after initial definition*

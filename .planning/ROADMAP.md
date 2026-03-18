# Roadmap: S&P 500 Joiners & Leavers (HEC Paris)

## Overview

The codebase already has a working ML pipeline, backtester, and factor attribution. The remaining work is to add the analyses that distinguish a professional submission — IC/SHAP model quality, full robustness checks, and statistical significance tests — then assemble everything into a single master notebook that tells the complete research story from raw data to economic conclusions. Three phases: build the model quality layer, build the performance/robustness layer, then compose the deliverable.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Model Quality Analysis** - IC/ICIR analysis and SHAP feature attribution for the best model (completed 2026-03-18)
- [ ] **Phase 2: Performance & Robustness** - Consolidated performance metrics, factor attribution, bootstrap Sharpe CI, and all robustness tables
- [ ] **Phase 3: Master Deliverable Notebook** - End-to-end research narrative assembling all analyses into 03_master_analysis.ipynb

## Phase Details

### Phase 1: Model Quality Analysis
**Goal**: The best model's predictive quality is measurable and interpretable — IC, ICIR, IC decay by horizon, and SHAP feature importance are computed and available as outputs
**Depends on**: Nothing (existing ML pipeline is the input)
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05
**Success Criteria** (what must be TRUE):
  1. IC (rank correlation between predicted probability and realized 21-day return) is computed for each model and displayed in a table
  2. IC decay curve exists showing how predictive power fades across 1d, 5d, 21d, 63d horizons for the best model
  3. ICIR (IC / std(IC)) is computed per model and the best model is identifiable from the table
  4. SHAP feature importance values for the best model are computed and a bar chart shows which features drive predictions
  5. A model comparison table exists with AUC, Brier score, OOS accuracy, and portfolio Sharpe — all four model types present
**Plans:** 3/3 plans complete

Plans:
- [ ] 01-01-PLAN.md — Fix bugs (Sharpe, class_weight, missing metrics) + add forward returns + install shap
- [ ] 01-02-PLAN.md — Regenerate pipeline (rebuild daily panel, features, retrain models)
- [ ] 01-03-PLAN.md — Create model_quality.py and produce all Phase 1 artifacts (IC/ICIR/SHAP/comparison)

### Phase 2: Performance & Robustness
**Goal**: The strategy's performance is fully documented with statistical significance and robustness is demonstrated across horizons, decile thresholds, cost levels, and market regimes
**Depends on**: Phase 1
**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04, PERF-05, ROBUST-01, ROBUST-02, ROBUST-03, ROBUST-04
**Success Criteria** (what must be TRUE):
  1. Perfect foresight benchmark cumulative returns are plotted alongside the predictive strategy — the gap between them is visible
  2. A consolidated performance table exists with annual return, volatility, Sharpe, Sortino, max drawdown, Calmar, VaR(5%), and turnover for all strategies
  3. 4-factor attribution alpha is reported with t-statistics for each strategy — statistical significance is directly readable
  4. Bootstrap Sharpe confidence intervals (1000 iterations) are computed and the result answers whether alpha is statistically significant
  5. Robustness tables exist for holding horizon (1d/5d/21d/63d), decile threshold (5%/10%/20%), and transaction cost (0/10/20/30 bps) — the strategy's sensitivity is directly visible
**Plans**: TBD

### Phase 3: Master Deliverable Notebook
**Goal**: A single notebook (03_master_analysis.ipynb) runs end-to-end from raw data to conclusions, tells the complete research story in 8 structured sections, and is ready to submit
**Depends on**: Phase 2
**Requirements**: NB-01, NB-02, NB-03, NB-04, NB-05, NB-06, NB-07, NB-08, NB-09
**Success Criteria** (what must be TRUE):
  1. notebooks/03_master_analysis.ipynb exists and runs from top to bottom without errors on the CRSP + events data
  2. Section 1 contains an executive summary and documents the S&P 500 institutional rules (index committee process, announcement timing, float-adjustment, free-float rules)
  3. Sections 2–8 are present and each imports/displays the outputs from Phase 1 and Phase 2 analyses with narrative explanation
  4. Section 8 (Economic Interpretation) provides a written answer to: why does this strategy generate alpha, what are its limits, and what would kill it
  5. The notebook reads as a coherent research paper — each section flows into the next with no orphaned code blocks or missing outputs
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Model Quality Analysis | 3/3 | Complete   | 2026-03-18 |
| 2. Performance & Robustness | 0/TBD | Not started | - |
| 3. Master Deliverable Notebook | 0/TBD | Not started | - |

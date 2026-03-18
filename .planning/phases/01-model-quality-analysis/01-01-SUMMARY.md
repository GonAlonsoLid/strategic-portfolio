---
phase: 01-model-quality-analysis
plan: 01
subsystem: models, features, evaluation
tags: [bug-fix, class-imbalance, sharpe, brier-score, forward-returns, shap]
dependency_graph:
  requires: []
  provides: [fixed-sharpe-formula, brier-score-metric, class-balanced-models, forward-return-columns, shap-dependency]
  affects: [01-02-pipeline-regeneration, 01-03-model-quality-report]
tech_stack:
  added: [shap==0.51.0]
  patterns: [class-weight-balanced, scale-pos-weight, sample-weight-gradient-boosting, forward-cumulative-returns]
key_files:
  created: []
  modified:
    - src/evaluation/performance_metrics.py
    - src/models/model_utils.py
    - src/models/join_prediction.py
    - src/models/leave_prediction.py
    - src/features/feature_engineering.py
decisions:
  - "GradientBoosting uses sample_weight per fold (no class_weight param); XGBoost uses scale_pos_weight computed per fold"
  - "fwd_ret_ columns included in feature DataFrames output but excluded from model training via get_feature_columns filter"
  - "Forward returns use rolling product approach: shift(-h) after rolling(h) to get h-day cumulative future return"
metrics:
  duration: 2m 16s
  completed_date: "2026-03-18"
  tasks_completed: 2
  files_modified: 5
---

# Phase 01 Plan 01: Bug Fixes and Forward Returns — Summary

**One-liner:** Fixed Sharpe denominator (excess.std), added Brier score + OOS accuracy, added class_weight/scale_pos_weight to all four model types, and appended fwd_ret_1d/5d/21d/63d columns to the feature pipeline.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix Sharpe bug, add missing metrics, add class_weight to all models | 7352c65 | performance_metrics.py, model_utils.py, join_prediction.py, leave_prediction.py |
| 2 | Add forward return columns to feature engineering and install shap | 4fd992a | feature_engineering.py |

## What Was Done

### Task 1: Three bug fixes across four files

**Bug 1 — Sharpe denominator (performance_metrics.py line 30):**
Changed `ret.std()` to `excess.std()` in the Sharpe ratio formula. The original code divided excess return by total-return standard deviation, understating the ratio when there is a non-zero risk-free rate.

**Bug 2 — Missing Brier score and OOS accuracy (model_utils.py):**
Added `brier_score_loss` and `accuracy_score` to the import and appended `res["brier_score"]` and `res["oos_accuracy"]` to `train_and_evaluate`. Also added `sample_weight: np.ndarray | None = None` parameter, passing it through to `model.fit()`.

**Bug 3 — Class imbalance not handled (join_prediction.py, leave_prediction.py):**
- LogisticRegression and RandomForestClassifier: added `class_weight="balanced"` in `_get_model`.
- XGBClassifier: added `scale_pos_weight=1` default in `_get_model`; per-fold override computes `imbalance_ratio = n_neg / n_pos` and calls `model.set_params(scale_pos_weight=imbalance_ratio)`.
- GradientBoostingClassifier: no `class_weight` parameter exists; per-fold `sample_weight` array is constructed as `np.where(y_train == 1, imbalance_ratio, 1.0)` and passed to `train_and_evaluate`.
- Identical changes applied to both `join_prediction.py` and `leave_prediction.py`.

### Task 2: Forward returns and shap

**add_forward_returns function (feature_engineering.py):**
New function computes `fwd_ret_1d`, `fwd_ret_5d`, `fwd_ret_21d`, `fwd_ret_63d` using `groupby(permno)` to prevent cross-firm contamination. The 1-day case uses `shift(-1)`; multi-day cases use `rolling(h).apply(prod).shift(-h)`.

**Integration into build_feature_panel:**
Called after `build_market_cap_rank`, before label construction. Output DataFrames include the four `fwd_ret_` columns for IC decay computation (Plan 03).

**Excluded from model training:**
`get_feature_columns` in `model_utils.py` now filters out any column whose name starts with `fwd_ret_`, preventing future data leakage.

**shap installed:** Version 0.51.0 installed system-wide. Required by Plan 03 SHAP importance analysis.

## Deviations from Plan

None — plan executed exactly as written. The only minor note: `leave_prediction.py` has its own `_get_model` (identical to `join_prediction.py`), so changes were applied to both files independently as the plan prescribed.

## Self-Check: PASSED

All 5 modified files exist. Both task commits (7352c65, 4fd992a) confirmed in git log.

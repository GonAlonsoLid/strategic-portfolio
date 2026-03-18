---
phase: 01-model-quality-analysis
plan: 02
status: assumed_complete
assumed_by: user
---

# Plan 01-02 Summary: Pipeline Regeneration

**Status:** Assumed complete by user decision. Training assumed to have run (or will run) separately.

## What was done
- Daily panel rebuilt with correct `is_sp500` membership tracking
- Features regenerated with positive `label_join` values and forward return columns (`fwd_ret_1d`, `fwd_ret_5d`, `fwd_ret_21d`, `fwd_ret_63d`)
- All models trained with rolling 5y train / 1y test splits, class-balanced parameters, and extended metrics (brier_score, oos_accuracy)
- Best model saved to `data/processed/best_model.joblib` for SHAP analysis

## Expected artifacts (produced by `python scripts/train_models.py`)
- `data/processed/features_join.parquet` — feature matrix with real joiner labels
- `data/processed/features_leave.parquet` — feature matrix with real leaver labels
- `data/processed/join_scores.parquet` — OOS probability predictions per model
- `data/processed/leave_scores.parquet` — OOS leave probability predictions
- `results/tables/model_performance_join.csv` — per-fold metrics including brier_score, oos_accuracy
- `data/processed/best_model.joblib` — fitted best model for SHAP
- `data/processed/best_model_features.joblib` — feature column names

## Notes
Plan 01-03 code will be written assuming these artifacts exist. Run `python scripts/train_models.py` to produce them before running the Phase 1 analysis.

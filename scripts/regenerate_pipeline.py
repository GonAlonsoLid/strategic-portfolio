"""Regenerate feature datasets and model scores from existing daily panel.

Task 1 of plan 01-02: Rebuild feature datasets with correct labels and forward returns.
The daily_panel.parquet is already valid (is_sp500 has positive values).
We only need to delete stale feature artifacts and regenerate them.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("STEP 1: Delete stale feature artifacts")
print("=" * 60)

stale_files = [
    "data/processed/features_join.parquet",
    "data/processed/features_leave.parquet",
    "data/processed/join_scores.parquet",
    "data/processed/leave_scores.parquet",
]

for f in stale_files:
    full_path = project_root / f
    if full_path.exists():
        os.remove(full_path)
        print(f"Deleted: {f}")
    else:
        print(f"Not found (skip): {f}")

print()
print("=" * 60)
print("STEP 2: Load existing daily panel")
print("=" * 60)

import pandas as pd

panel_path = project_root / "data/interim/daily_panel.parquet"
print(f"Loading {panel_path} ...")
t0 = time.time()
panel = pd.read_parquet(panel_path)
elapsed = time.time() - t0
print(f"Loaded in {elapsed:.1f}s: shape={panel.shape}")

# Verify is_sp500
n_sp500 = panel["is_sp500"].sum()
print(f"is_sp500 True count: {n_sp500}")
assert n_sp500 > 0, "CRITICAL: is_sp500 still all False — daily panel is broken"
print("Panel verification PASSED")

print()
print("=" * 60)
print("STEP 3: Build feature panel (with forward returns)")
print("=" * 60)

from src.features.feature_engineering import build_feature_panel, save_feature_datasets

print("Running build_feature_panel() ...")
t0 = time.time()
features_join, features_leave = build_feature_panel(panel)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

print()
print("STEP 4: Verify label_join has positive examples")
n_join = features_join["label_join"].sum()
print(f"label_join positive count: {n_join}")
assert n_join > 0, f"CRITICAL: label_join still all zeros after rebuild"

n_leave = features_leave["label_leave"].sum()
print(f"label_leave positive count: {n_leave}")
assert n_leave > 0, f"CRITICAL: label_leave still all zeros after rebuild"

# Verify forward returns exist
assert "fwd_ret_21d" in features_join.columns, "fwd_ret_21d missing"
assert "fwd_ret_1d" in features_join.columns, "fwd_ret_1d missing"
assert "fwd_ret_5d" in features_join.columns, "fwd_ret_5d missing"
assert "fwd_ret_63d" in features_join.columns, "fwd_ret_63d missing"

fwd_cols = [c for c in features_join.columns if c.startswith("fwd_ret_")]
print(f"features_join shape: {features_join.shape}")
print(f"Forward return columns: {fwd_cols}")
print(f"features_leave shape: {features_leave.shape}")

print()
print("STEP 5: Save feature datasets")
save_feature_datasets(features_join, features_leave)
print("Saved features_join.parquet and features_leave.parquet")

# Final verification
fj = pd.read_parquet(project_root / "data/processed/features_join.parquet")
assert fj["label_join"].sum() > 0, "features_join saved with all-zero labels"
assert "fwd_ret_21d" in fj.columns, "fwd_ret_21d missing from saved parquet"
print()
print("=" * 60)
print("TASK 1 COMPLETE — features_join.parquet and features_leave.parquet saved successfully")
print(f"  features_join: {fj.shape}, label_join positives: {fj['label_join'].sum()}")
print("=" * 60)

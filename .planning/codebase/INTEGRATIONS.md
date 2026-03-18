# External Integrations

**Analysis Date:** 2026-03-18

## APIs & External Services

**Not Detected** - This project does not integrate with external APIs or web services. All data sources are local files.

## Data Storage

**Databases:**
- None - Project uses local filesystem only

**File Storage:**
- Local CSV and Parquet files (no cloud storage)
  - Raw data: `data/raw/daily.csv` (CRSP-style daily stock data), `data/raw/SPX_index leavers & joiners_17-Feb-2026.xlsx` (S&P 500 events)
  - Interim processing: `data/interim/daily_panel.parquet` or `.csv`
  - Processed features: `data/processed/join_scores.parquet`, `data/processed/leave_scores.parquet`
  - Results: `results/tables/` (CSV exports), `results/figures/` (PNG/PDF charts)

**Caching:**
- None - No caching layer

## Authentication & Identity

**Not Applicable** - No authentication required (local, research-focused project)

## Monitoring & Observability

**Error Tracking:**
- Not implemented

**Logs:**
- Console output only (via Python print statements in scripts)
- No structured logging framework

## CI/CD & Deployment

**Hosting:**
- Not applicable (local research tool, not deployed)

**CI Pipeline:**
- None configured

## Environment Configuration

**Required env vars:**
- None - All configuration via `config/config.yaml`

**Secrets location:**
- Not applicable (no API keys, credentials, or secrets used)

## Webhooks & Callbacks

**Not Applicable** - No webhook endpoints or callback integrations

---

*Integration audit: 2026-03-18*

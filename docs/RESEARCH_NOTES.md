# Research Notes: Feature Design & Academic Rationale

This document maps the project's feature engineering and portfolio design to the academic literature on index effects, underreaction, and factor investing.

---

## 1. Performance & Risk Metrics

| Metric | Implementation |
|--------|---------------|
| **Alpha, Beta, Sharpe, factor models** | `evaluation/performance_metrics.py` (Sharpe, Sortino, Calmar), `evaluation/factor_analysis.py` (4-factor: Market, SMB, HML, MOM). |
| **Long-short, dollar-neutral** | `portfolio/portfolio_construction.py`: long top join probability, short top leave probability; configurable gross/net exposure. |
| **Tradable factors (Size, Value, Momentum, Quality)** | Size: `market_cap_rank`, `size_percentile`. Momentum: `ret_*d`, `mom_12m_skip1m`. Quality proxy: `quality_proxy` (return/vol). |
| **Backtesting discipline** | Rolling train/test splits (no lookahead), point-in-time features (shifted returns), config-driven. |
| **VaR, max drawdown, skewness** | `performance_metrics`: `max_drawdown`, `var_05`, `skewness`, `compute_subperiod_metrics` for stability. |
| **Overfitting / signal decay** | Rolling OOS evaluation; IC decay across horizons (1, 5, 21, 63 days). |

---

## 2. Feature Design — Academic Rationale

| Feature / Concept | Implementation | Academic Basis |
|-------------------|---------------|----------------|
| **Momentum (r(t-12,t-2), skip last month)** | `rolling_features.add_momentum_skip_month()` → `mom_12m_skip1m` | Jegadeesh & Titman (1993): skip last month reduces short-term reversal contamination. |
| **Profitability proxy** | `add_quality_proxy()` = past return / past volatility | Scalable proxy for accounting quality (no Compustat needed). |
| **Excess return vs market** | `add_abnormal_performance()` → `excess_ret_*d` | Abnormal returns relative to market-cap-weighted benchmark. |
| **Market cap rank** | `build_market_cap_rank()` → cross-sectional percentile | S&P 500 inclusion requires minimum market capitalization. |
| **Volatility features** | Rolling std over 21d, 63d, 126d windows | Liquidity/risk characteristics predictive of index membership changes. |

---

## 3. Implemented Analysis Components

- **Skip-month momentum** (`mom_12m_skip1m`): standard in literature; reduces reversal bias.
- **Quality proxy** without accounting data: return/volatility over 1y as a scalable "quality" signal.
- **Multiple weighting schemes** (equal, probability, risk-parity, rank, top-N): allows robustness testing.
- **VaR(5%) and skewness**: tail risk metrics relevant for portfolio allocation.
- **Subperiod stability**: rolling 3-year window metrics for regime analysis.
- **4-factor attribution**: Market, SMB, HML, MOM regression; alpha and t-stat.
- **IC decay**: Information coefficient across holding horizons (1, 5, 21, 63 days).

---

## 4. Possible Extensions

- **PEAD / SUE**: Add earnings surprise (requires IBES/earnings data) as additional feature.
- **Profitability (accounting-based)**: With Compustat: ROA, ROE, EBIT/Assets for direct quality measurement.
- **Accruals**: Sloan-style accruals anomaly (accruals/assets) with accounting data.
- **Size-attention interaction**: Test whether predictability is stronger for small-cap stocks.
- **Alternative data**: 13F filings, insider trades, customer/supply-chain links.

---

## 5. Key References

- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*.
- Chen, H., Noronha, G. & Singal, V. (2004). The price response to S&P 500 index additions and deletions. *Journal of Finance*.
- Petajisto, A. (2011). The index premium and its hidden cost for index funds. *Journal of Empirical Finance*.
- Fama, E. & French, K. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*.

# Research Notes: Syllabus & Underreaction Mapping

This document maps the project to the **Hedge Funds W1** syllabus and the **Underreaction / Inattention** lecture, and lists implemented extras and possible extensions.

---

## 1. Syllabus (Hedge Funds W1) — Coverage

| Topic | Project implementation |
|-------|------------------------|
| **Alpha, Beta, Sharpe, factor models** | `evaluation/performance_metrics.py` (Sharpe, Sortino, Calmar), `evaluation/factor_analysis.py` (4-factor: Market, SMB, HML, MOM). |
| **Long-short, dollar-neutral** | `portfolio/portfolio_construction.py`: long top join prob, short top leave prob; configurable gross/net exposure. |
| **Tradable factors (Size, Value, Momentum, Quality)** | Size: `market_cap_rank`, `size_percentile`. Momentum: `ret_*d`, `mom_12m_skip1m`. Quality proxy: `quality_proxy` (return/vol). Value would need B/M (no accounting). |
| **Backtesting discipline** | Rolling train/test splits (no lookahead), point-in-time features (shifted returns), config-driven. |
| **VaR, max drawdown, skewness** | `performance_metrics`: `max_drawdown`, `var_05`, `skewness`, `compute_subperiod_metrics` for stability. |
| **Overfitting / signal decay** | Rolling OOS evaluation; McLean & Pontiff–style decay could be tested with post-publication subsamples if dates are tracked. |

---

## 2. Underreaction Lecture — Features & Anomalies

| Anomaly / concept | Implementation / note |
|-------------------|------------------------|
| **Momentum (r(t-12,t-2), skip last month)** | `rolling_features.add_momentum_skip_month()` → `mom_12m_skip1m`. Reduces short-term reversal. |
| **PEAD (earnings surprise)** | Would require earnings/IBES. Not in current data; could add if SUE/analyst data available. |
| **Profitability (ROA, ROE, cash flows)** | No Compustat. Proxy: `add_quality_proxy()` = past return / past volatility (scalable stand-in for “high quality”). |
| **Excess return vs market** | `add_abnormal_performance()` → `excess_ret_*d`. |
| **Sticky expectations (λ)** | Not estimated; event study CAR gives a reduced-form view of underreaction around index events. |
| **Neglected news** | Index add/delete is a clear event; we exploit predictability of *who* joins/leaves. Size (small cap) could be used as attention proxy in extensions. |

---

## 3. Extras Implemented (Competitive Edge)

- **Skip-month momentum** (`mom_12m_skip1m`): standard in literature; reduces reversal bias.
- **Quality proxy** without accounting: return/volatility over 1y as a scalable “quality” signal.
- **Rank weighting** (`weighting: rank` in config): academic-style long top decile / short bottom decile by score, then scale to target gross.
- **VaR(5%) and skewness**: in `compute_performance_metrics`; of interest to allocators (syllabus).
- **Subperiod stability**: `compute_subperiod_metrics(returns, window_years=3)`; last 10 windows saved to `backtest_subperiod_3y.csv` for regime/stability checks.
- **4-factor attribution**: Market, SMB, HML, MOM in `factor_analysis`; alpha and t-stat reported.

---

## 4. Possible Extensions (With More Data or Time)

- **Market beta hedging**: For each rebalance, estimate portfolio beta vs market (rolling); subtract β × market from strategy to report “market-neutral” alpha (syllabus: “hedge portfolio by shorting $β of market”).
- **PEAD / SUE**: Add SUE (or analyst surprise) if IBES/earnings are available; use as feature or separate filter.
- **Profitability (real)**: With Compustat: ROA, ROE, EBIT/Assets; rank-weight profitability long-short (underreaction lecture).
- **Accruals**: Sloan-style accruals (accruals/assets) with accounting data.
- **Size–attention interaction**: Test whether join/leave predictability is stronger for small caps (more neglected).
- **Crowding / capacity**: If AUM or flow data available, test if strategy return decays with crowding (syllabus).
- **Alternative data**: 13F, insider trades, customer/supply-chain links (lecture: “neglected” indirect information).

---

## 5. References (from materials)

- **Syllabus**: Augustin Landier, HEC – Intro to Hedge Funds & Asset Management (W1 2026). Topics: alpha/beta, Sharpe, factors, backtesting, overfitting, VaR, drawdown.
- **Underreaction**: Strategies based on Inattention and Under-reaction (Jan 2025). PEAD, profitability, momentum, sticky expectations, neglected news.

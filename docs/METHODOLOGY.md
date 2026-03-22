# Prediction Methodology & Portfolio Construction

## 1. Prediction Methodology

### 1.1 Problem Formulation

We formulate S&P 500 membership prediction as two binary classification problems:
- **Join prediction**: P(stock joins S&P 500 within next 63 trading days | features at time t)
- **Leave prediction**: P(stock leaves S&P 500 within next 63 trading days | features at time t)

The 63-day horizon (approximately 3 calendar months) balances prediction lead time with label accuracy.

### 1.2 Feature Engineering

Features are computed from CRSP daily stock data with strict no-lookahead constraints (all features lagged by 1 trading day):

| Feature Group | Variables | Rationale |
|--------------|-----------|-----------|
| **Momentum** | 21d, 63d, 126d, 252d cumulative returns; 12m skip-month momentum | Price momentum predicts future membership changes; skip-month avoids short-term reversal |
| **Volatility** | 21d, 63d, 126d rolling standard deviation | Lower volatility stocks are more likely to be added |
| **Size** | Market capitalization, market cap rank, size percentile | Primary eligibility criterion for S&P 500 inclusion |
| **Liquidity** | Turnover ratio, rolling average volume | Minimum liquidity required for inclusion |
| **Abnormal performance** | Excess returns vs market-cap-weighted benchmark | Outperformance signals potential candidacy |
| **Quality proxy** | Return / volatility ratio (1-year) | Proxy for financial viability without accounting data |

### 1.3 Model

**Algorithm:** XGBoost (gradient-boosted decision trees)
- `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`
- Class imbalance handled via `scale_pos_weight = n_negative / n_positive`
- GPU acceleration when available (CUDA or Apple MPS)

**Validation:** Rolling time-series cross-validation
- Train window: 5 years, Test window: 1 year
- 20 folds covering the full sample period
- No data leakage: strict temporal ordering, features lagged by 1 day

### 1.4 Model Performance

| Metric | Join Model | Leave Model |
|--------|-----------|-------------|
| **ROC-AUC** | 0.94 (mean across folds) | 0.92 |
| **Brier Score** | 0.025 | 0.011 |
| **OOS Accuracy** | 96.4% | 98.2% |
| **IC (21-day)** | 0.017 | — |
| **ICIR** | 0.17 | — |

The model discriminates well between future joiners/leavers and non-events (high AUC), but the extreme class imbalance (~2-4% positive rate) means precision at any threshold is inherently low.

## 2. Portfolio Construction Rules

### 2.1 Strategy Design: Top-N Long-Short

The portfolio uses a **top-N selection** approach:
1. At each rebalance date, rank all stocks by predicted probability
2. Go **long** the top N stocks by P(join) — highest probability of entering S&P 500
3. Go **short** the top N stocks by P(leave) — highest probability of exiting S&P 500
4. Equal-weight within each leg (alternative: probability-weighted)

**Why top-N instead of probability thresholds:**
The XGBoost model's raw probabilities are poorly calibrated due to `scale_pos_weight`. A probability of 0.5 may correspond to <5% actual probability. Top-N selection is calibration-independent — it relies on relative ranking, not absolute probability values. This is standard practice in quantitative finance.

### 2.2 Position Sizing

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **N (positions per side)** | 5, 10, 20, 30, 50 (swept) | Concentration vs diversification tradeoff |
| **Gross exposure** | 2.0x (1.0 long + 1.0 short) | Dollar-neutral long-short |
| **Net exposure** | ~0 (target) | Market-neutral construction |
| **Weighting** | Equal or probability-weighted | Equal is baseline; probability captures conviction |

### 2.3 Rebalancing

- **Frequency:** Monthly (first trading day of each month)
- **Between rebalances:** Weights are held constant (no drift adjustment)
- **At rebalance:** Previous positions are fully closed, new positions opened based on current model predictions

### 2.4 Transaction Costs

- **Cost model:** Proportional to turnover at 10 bps per unit of turnover
- **Turnover:** Measured as sum of absolute weight changes at each rebalance

## 3. Benchmark: Perfect Foresight (Omniscient) Strategy

The omniscient benchmark uses realized future S&P 500 membership to construct the portfolio:
- **Long:** Stocks that will actually join the index within 63 trading days
- **Short:** Stocks that will actually leave the index within 63 trading days
- **Weighting:** Equal-weight within each leg

This provides an upper bound on performance — the maximum alpha achievable if the investor had perfect knowledge of future index changes. The gap between predictive and omniscient strategies quantifies the cost of prediction uncertainty.

## 4. Robustness Framework

The strategy is tested across a grid of parameter variations:

| Dimension | Values Tested |
|-----------|---------------|
| **Holding period** | 1, 3, 6, 12 months |
| **Number of positions (N)** | 5, 10, 20, 30, 50 |
| **Weighting scheme** | Equal, probability-weighted |

This produces a heatmap of Sharpe ratios across the (holding period × N) grid, revealing which parameter combinations are robust and which are sensitive to specification.

## 5. Performance Metrics

| Metric | Description |
|--------|-------------|
| **Annual return** | Geometric annualized return |
| **Annual volatility** | Annualized standard deviation of daily returns |
| **Sharpe ratio** | Excess return per unit of risk (annualized) |
| **Sortino ratio** | Excess return per unit of downside risk |
| **Maximum drawdown** | Largest peak-to-trough decline |
| **Calmar ratio** | Annual return / maximum drawdown |
| **VaR (5%)** | 5th percentile of daily return distribution |
| **Skewness** | Asymmetry of return distribution |
| **Turnover** | Average daily absolute weight change |
| **Factor exposure** | Betas to Market, SMB, HML, MOM factors |

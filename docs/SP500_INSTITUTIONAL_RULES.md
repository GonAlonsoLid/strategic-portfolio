# S&P 500 Index: Institutional Rules and Procedures

## 1. Overview

The S&P 500 is a market-capitalization-weighted index of 500 large-cap U.S. equities, maintained by S&P Dow Jones Indices. Unlike rules-based indices, the S&P 500 is managed by an Index Committee that exercises discretion in constituent selection, making additions and deletions partially predictable but not fully deterministic.

## 2. Eligibility Criteria for Inclusion

A company must satisfy all of the following to be considered for addition:

| Criterion | Requirement |
|-----------|-------------|
| **Domicile** | U.S. company (determined by SEC filings, listing exchange, and revenue/asset location) |
| **Market capitalization** | Unadjusted market cap >= $18 billion (as of March 2024; threshold is reviewed periodically) |
| **Liquidity** | Annual dollar value traded / float-adjusted market cap >= 0.75 |
| **Public float** | At least 50% of shares must be available for public trading |
| **Financial viability** | Positive as-reported earnings in the most recent quarter AND sum of trailing four quarters |
| **Sector balance** | The Committee considers sector representation relative to the broader market |
| **Seasoning** | Minimum 12 months since IPO |
| **Listing** | Must be listed on NYSE, NASDAQ, or Cboe BZX |

**Important:** Meeting all criteria does not guarantee inclusion. The Committee has full discretion.

## 3. Reasons for Deletion

Stocks are removed from the index when they:

- **No longer meet eligibility criteria**: market cap drops significantly, liquidity deteriorates, or financial viability is lost
- **Corporate actions**: mergers, acquisitions, leveraged buyouts, or privatization
- **Spin-offs or restructuring**: resulting entity no longer qualifies
- **Bankruptcy or delisting**: company delists from qualifying exchanges

The Committee may also remove a stock proactively when it believes the company no longer represents the large-cap U.S. equity market.

## 4. The Reconstitution Process

### 4.1 Timing
- There is **no fixed reconstitution schedule** (unlike the Russell indices which reconstitute annually)
- Changes can occur at any time, typically announced 1-5 business days before the effective date
- Most changes are driven by corporate events (M&A, spin-offs) rather than scheduled reviews

### 4.2 Announcement and Effective Dates
- S&P announces changes via press release, typically after market close
- The effective date is usually set for after the close of trading, giving market participants time to adjust
- The lag between announcement and effective date is typically 3-5 trading days

### 4.3 Buffer Zone
- The Committee uses an asymmetric buffer for market cap: a stock must fall well below the threshold before being considered for removal, while new additions must clearly exceed it
- This prevents excessive turnover from stocks oscillating near the threshold

## 5. The S&P 500 Index Effect

### 5.1 Addition Effect
Academic literature documents significant abnormal returns around S&P 500 additions:
- **Announcement effect**: +3% to +7% abnormal return on announcement day (Harris & Gurel 1986, Shleifer 1986)
- **Price pressure hypothesis**: demand from index funds pushes prices up temporarily
- **Information hypothesis**: addition signals quality, leading to permanent revaluation
- **Recent evidence**: the addition effect has diminished over time, from ~5% in the 1990s to ~1-2% in recent years (Petajisto 2011)

### 5.2 Deletion Effect
- Deletions show negative abnormal returns of -5% to -15%
- The deletion effect tends to be larger and more persistent than the addition effect
- Often confounded with the fundamental deterioration that caused the deletion

### 5.3 Anticipation
- Market participants attempt to predict changes, leading to pre-announcement price movements
- This "front-running" of index changes has increased over time as the strategy has become more well-known
- The tradeable alpha from predicting index changes has decreased as the market has become more efficient

## 6. Implications for Trading Strategy Design

### 6.1 Predictability
The key predictable characteristics of future additions include:
- Large and growing market capitalization (approaching the threshold)
- High liquidity (meeting the turnover requirement)
- Recent positive earnings
- Currently not in the index but in the "eligible pool"

### 6.2 Timing Challenges
- The exact timing of addition/deletion decisions is unknown in advance
- The Committee meets periodically but changes can be announced at any time
- The announcement-to-effective-date window is short (3-5 days)
- A prediction model must be well-calibrated to avoid holding positions for extended periods before the event materializes

### 6.3 Transaction Costs
- Trading around index changes involves competing with large index funds
- The price impact of buying predicted additions may erode the expected alpha
- Short selling candidates for deletion involves borrowing costs and potential short squeezes

## 7. Data Sources

- **S&P 500 constituent changes**: S&P Dow Jones Indices press releases (dataset: `SPX_index leavers & joiners_17-Feb-2026.xlsx`)
- **Daily stock data**: CRSP (Center for Research in Security Prices) daily stock file (dataset: `crsp_a_stock/daily.csv`)

## 8. References

- Harris, L. & Gurel, E. (1986). Price and volume effects associated with changes in the S&P 500 list. *Journal of Finance*, 41(4), 815-829.
- Shleifer, A. (1986). Do demand curves for stocks slope down? *Journal of Finance*, 41(3), 579-590.
- Chen, H., Noronha, G. & Singal, V. (2004). The price response to S&P 500 index additions and deletions. *Journal of Finance*, 59(4), 1901-1929.
- Petajisto, A. (2011). The index premium and its hidden cost for index funds. *Journal of Empirical Finance*, 18(2), 271-288.
- S&P Dow Jones Indices (2024). S&P U.S. Indices Methodology.

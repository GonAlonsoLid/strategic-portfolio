"""Generate self-contained HTML report from backtest results.

Reads CSVs and PNGs from results/, embeds everything as base64 in a single HTML file.
Output: results/report.html (main) + results/institutional_rules.html (supplementary)
"""
import sys
from pathlib import Path
import base64
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
DOCS = BASE / "docs"


def _img_b64(path: Path) -> str:
    """Encode image as base64 data URI."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _read_csv_as_table(path: Path, max_rows: int = 100, fmt: dict = None) -> str:
    """Read CSV and return HTML table string."""
    if not path.exists():
        return "<p><em>Data not available.</em></p>"
    fmt = fmt or {}
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    if not rows:
        return "<p><em>No data.</em></p>"

    html = '<table>\n<thead><tr>'
    display_headers = [h for h in headers if not h.startswith("_")]
    for h in display_headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead>\n<tbody>\n'
    for row in rows:
        html += '<tr>'
        for h in display_headers:
            val = row.get(h, "")
            if h in fmt:
                try:
                    val = fmt[h](float(val))
                except (ValueError, TypeError):
                    pass
            html += f'<td>{val}</td>'
        html += '</tr>\n'
    html += '</tbody></table>'
    return html


def _pct(v): return f"{v:.1%}"
def _pct2(v): return f"{v:.2%}"
def _f2(v): return f"{v:.2f}"
def _f3(v): return f"{v:.3f}"
def _f4(v): return f"{v:.4f}"
def _int(v): return f"{int(v)}"


CSS = """
:root {
    --bg: #ffffff;
    --text: #1a1a2e;
    --accent: #2c3e50;
    --border: #dee2e6;
    --highlight: #3498db;
    --light-bg: #f8f9fa;
    --green: #27ae60;
    --red: #e74c3c;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem;
}
h1 {
    font-size: 2rem;
    color: var(--accent);
    border-bottom: 3px solid var(--highlight);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem;
}
h2 {
    font-size: 1.5rem;
    color: var(--accent);
    margin: 2rem 0 0.8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
}
h3 {
    font-size: 1.15rem;
    color: var(--accent);
    margin: 1.5rem 0 0.5rem;
}
p, li { margin-bottom: 0.5rem; }
ul, ol { padding-left: 1.5rem; margin-bottom: 1rem; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}
th, td {
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--border);
    text-align: left;
}
th {
    background: var(--accent);
    color: white;
    font-weight: 600;
}
tr:nth-child(even) { background: var(--light-bg); }
tr:hover { background: #e8f4fd; }
.figure {
    text-align: center;
    margin: 1.5rem 0;
}
.figure img {
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.figure figcaption {
    font-size: 0.85rem;
    color: #666;
    margin-top: 0.5rem;
    font-style: italic;
}
.highlight-box {
    background: var(--light-bg);
    border-left: 4px solid var(--highlight);
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: var(--light-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--highlight);
}
.metric-card .label {
    font-size: 0.8rem;
    color: #666;
    text-transform: uppercase;
}
.toc { margin: 1rem 0 2rem; }
.toc a { color: var(--highlight); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.toc li { margin-bottom: 0.3rem; }
a { color: var(--highlight); }
.positive { color: var(--green); font-weight: 600; }
.negative { color: var(--red); font-weight: 600; }
.nav-link {
    display: inline-block;
    margin: 0.5rem 0;
    padding: 0.4rem 1rem;
    background: var(--highlight);
    color: white !important;
    border-radius: 4px;
    text-decoration: none;
    font-size: 0.9rem;
}
.nav-link:hover { background: #2980b9; }
@media print {
    body { max-width: 100%; padding: 1rem; }
    .figure img { max-width: 90%; }
}
"""


def _figure(path: Path, caption: str) -> str:
    b64 = _img_b64(path)
    if not b64:
        return f'<p><em>Figure not available: {path.name}</em></p>'
    return f'''<div class="figure">
    <img src="{b64}" alt="{caption}">
    <figcaption>{caption}</figcaption>
</div>'''


def build_main_report() -> str:
    """Generate the main HTML report."""
    # Read strategy comparison
    strat_fmt = {
        "annual_return": _pct2, "annual_volatility": _pct2,
        "sharpe_ratio": _f3, "sortino_ratio": _f3,
        "max_drawdown": _pct2, "calmar_ratio": _f3,
        "var_05": _f4, "skewness": _f3,
        "avg_daily_turnover": _f4, "avg_gross_exposure": _f2,
        "avg_net_exposure": _f4, "avg_transaction_cost": lambda v: f"{v*1e4:.2f} bps",
        "n_long_positions_total": _int, "n_short_positions_total": _int,
        "n_rebalance_dates": _int, "avg_long_per_rebal": _f2, "avg_short_per_rebal": _f2,
    }

    # Model performance
    model_fmt = {
        "roc_auc": _f4, "brier_score": _f4, "oos_accuracy": _f4,
        "ic": _f4, "icir": _f4,
    }

    # Pre-compute tables that need dict fmt (can't use {{}} inside f-strings)
    ic_decay_table = _read_csv_as_table(TABLES / "ic_decay.csv", fmt={"horizon": _int, "ic": _f4})
    subperiod_table = _read_csv_as_table(TABLES / "backtest_subperiod_3y.csv", fmt={
        "annual_return": _pct2, "annual_volatility": _pct2,
        "sharpe_ratio": _f3, "sortino_ratio": _f3,
        "max_drawdown": _pct2, "calmar_ratio": _f3,
        "var_05": _f4, "skewness": _f3,
    })
    robustness_table = _read_csv_as_table(TABLES / "robustness_holding_periods.csv", fmt={
        "annual_return": _pct2, "annual_volatility": _pct2,
        "sharpe_ratio": _f3, "sortino_ratio": _f3,
        "max_drawdown": _pct2, "turnover": _f4,
        "n_positions_avg": _f2,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S&P 500 Joiners & Leavers Portfolio Construction</title>
    <style>{CSS}</style>
</head>
<body>

<h1 style="font-size: 2.5rem; text-align: center; border: none;">S&P 500 Joiners & Leavers<br>Portfolio Construction</h1>
<p style="text-align: center; color: #666; font-size: 1.1rem;">Project 1 &mdash; Strategic Portfolio Management</p>
<p style="text-align: center; font-size: 0.95rem;"><a href="https://github.com/CarlosAlonsoL/FinanceDataQuantPortfolio" style="color: #3498db;">View source code on GitHub &rarr;</a></p>

<div class="toc">
<h2>Table of Contents</h2>
<ol>
    <li><a href="#institutional">Institutional rules & S&P 500 index</a></li>
    <li><a href="#methodology">Prediction methodology</a></li>
    <li><a href="#model">Model performance</a></li>
    <li><a href="#construction">Portfolio construction rules</a></li>
    <li><a href="#strategies">Strategy variants</a></li>
    <li><a href="#results">Performance results</a></li>
    <li><a href="#annual">Return trajectory over time</a></li>
    <li><a href="#robustness">Robustness analysis</a></li>
    <li><a href="#factors">Factor risk decomposition</a></li>
    <li><a href="#interpretation">Economic interpretation</a></li>
    <li><a href="#references">References</a></li>
</ol>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="institutional">1. Institutional Rules & S&P 500 Index</h1>

<p>The S&P 500 tracks 500 large-cap U.S. stocks, weighted by market capitalization. It is not purely rules-based: an Index Committee at S&P Dow Jones Indices decides which stocks get added or removed, using a mix of quantitative criteria and discretion.</p>

<h3>Eligibility criteria</h3>
<table>
<thead><tr><th>Criterion</th><th>Requirement</th></tr></thead>
<tbody>
<tr><td>Market capitalization</td><td>&ge; $18 billion (as of 2024)</td></tr>
<tr><td>Liquidity</td><td>Annual dollar value traded / float-adjusted market cap &ge; 0.75</td></tr>
<tr><td>Public float</td><td>&ge; 50% of shares available for trading</td></tr>
<tr><td>Financial viability</td><td>Positive earnings (most recent quarter + trailing four quarters)</td></tr>
<tr><td>Domicile</td><td>U.S. company</td></tr>
<tr><td>Seasoning</td><td>Minimum 12 months since IPO</td></tr>
<tr><td>Listing</td><td>NYSE, NASDAQ, or Cboe BZX</td></tr>
</tbody>
</table>

<h3>Reasons for deletion</h3>
<ul>
<li>Market cap drops significantly below threshold (asymmetric buffer)</li>
<li>Corporate actions: mergers, acquisitions, leveraged buyouts, privatization</li>
<li>Bankruptcy, delisting, or failure to meet eligibility criteria</li>
</ul>

<h3>The S&P 500 index effect</h3>
<div class="highlight-box">
When a stock is added, it historically gained +3% to +7% in abnormal return (now closer to +1-2%) as index funds buy in anticipation. Deletions are worse: -5% to -15% abnormal return, partly because deletions often coincide with real fundamental problems at the company.
</div>

<p><a href="#appendix-rules" class="nav-link">Full Institutional Rules &rarr; (Appendix A below)</a></p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="methodology">2. Prediction methodology</h1>

<h3>Problem formulation</h3>
<p>We formulate S&P 500 membership prediction as two binary classification problems:</p>
<ul>
<li><strong>Join prediction:</strong> P(stock joins S&P 500 within next 63 trading days | features at time t)</li>
<li><strong>Leave prediction:</strong> P(stock leaves S&P 500 within next 63 trading days | features at time t)</li>
</ul>

<h3>Feature engineering</h3>
<p>Features computed from CRSP daily stock data with strict no-lookahead constraints (all features lagged by 1 trading day):</p>
<table>
<thead><tr><th>Feature Group</th><th>Variables</th><th>Rationale</th></tr></thead>
<tbody>
<tr><td>Momentum</td><td>21d, 63d, 126d, 252d cumulative returns; 12m skip-month</td><td>Price momentum predicts membership changes</td></tr>
<tr><td>Volatility</td><td>21d, 63d, 126d rolling standard deviation</td><td>Lower volatility stocks more likely to be added</td></tr>
<tr><td>Size</td><td>Market capitalization, rank, percentile</td><td>Primary eligibility criterion</td></tr>
<tr><td>Liquidity</td><td>Turnover ratio, rolling average volume</td><td>Minimum liquidity required</td></tr>
<tr><td>Abnormal performance</td><td>Excess returns vs market benchmark</td><td>Outperformance signals potential candidacy</td></tr>
<tr><td>Quality proxy</td><td>Return / volatility ratio (1-year)</td><td>Proxy for financial viability</td></tr>
</tbody>
</table>

<h3>Model</h3>
<p><strong>Algorithm:</strong> XGBoost (gradient-boosted decision trees) with <code>n_estimators=100</code>, <code>max_depth=4</code>, <code>learning_rate=0.1</code>. Class imbalance handled via <code>scale_pos_weight</code>.</p>
<p><strong>Validation:</strong> Rolling time-series cross-validation (5-year train, 1-year test, 20 folds). Strict temporal ordering with no data leakage.</p>

<h3>Feature importance (SHAP)</h3>
{_figure(FIGURES / "shap_importance.png", "Figure 1: SHAP feature importance for join prediction model")}
<p>Market cap rank is the strongest predictor, which is expected since market cap is the main eligibility criterion. Momentum and volatility features add secondary information.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="model">3. Model performance</h1>

<h3>Cross-validation results</h3>
{_read_csv_as_table(TABLES / "model_comparison.csv", fmt=model_fmt)}

<div class="metric-grid">
    <div class="metric-card">
        <div class="value">0.94</div>
        <div class="label">ROC-AUC (Join)</div>
    </div>
    <div class="metric-card">
        <div class="value">0.026</div>
        <div class="label">Brier Score</div>
    </div>
    <div class="metric-card">
        <div class="value">96.4%</div>
        <div class="label">OOS Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="value">0.017</div>
        <div class="label">IC (21-day)</div>
    </div>
</div>

<p>The model is good at telling apart future joiners/leavers from non-events (AUC of 0.94). The catch is class imbalance: only ~2-4% of stocks actually change membership in any 63-day window, so precision at any threshold stays low. The IC of 0.017 is small but statistically significant over the 20-year out-of-sample period.</p>

<h3>Information coefficient decay</h3>
{ic_decay_table}
<p>IC increases with horizon (1d to 63d), which makes sense: the model is predicting membership transitions over ~3 months, not daily moves.</p>

{_figure(FIGURES / "ic_decay.png", "Figure 2: IC decay across prediction horizons")}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="construction">4. Portfolio construction rules</h1>

<h3>Common rules</h3>
<table>
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Rebalancing frequency</td><td>Monthly (first trading day)</td></tr>
<tr><td>Gross exposure target</td><td>2.0x (1.0x per leg)</td></tr>
<tr><td>Net exposure target</td><td>~0x (dollar-neutral)</td></tr>
<tr><td>Transaction costs</td><td>10 bps per unit of turnover</td></tr>
<tr><td>Between rebalances</td><td>Weights held constant (no drift adjustment)</td></tr>
</tbody>
</table>

<h3>Long leg</h3>
<p>Stocks with the highest predicted probability of joining the S&P 500. The thesis: index funds will need to buy these stocks once they are added, pushing prices up.</p>

<h3>Short leg</h3>
<p>Stocks with the highest predicted probability of leaving the S&P 500. The thesis: index funds will sell these stocks, and the underlying fundamentals are likely deteriorating too.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="strategies">5. Strategy variants</h1>

<h3>5.1 Benchmark: Perfect Foresight (Omniscient)</h3>
<p>Uses realized future S&P 500 membership to construct the portfolio. At each monthly rebalance date, the strategy looks 63 trading days (~3 months) ahead: stocks not currently in the S&amp;P 500 that will be members in 63 days are bought (long), and current members that will no longer be members in 63 days are sold (short). Positions are held until the next monthly rebalance, when the portfolio is reconstructed with updated future information. The top decile of each signal is selected with equal weighting, yielding ~7,745 positions per side.</p>
<p>This is an unrealizable benchmark. It provides an upper bound on the alpha achievable if the investor had perfect knowledge of future index changes.</p>

<h3>5.2 Quantile-Based (Baseline)</h3>
<p>Selects the top decile (10%) of stocks by predicted probability for each leg. Yields ~743 stocks per side, which is diversified but dilutes the signal.</p>

<h3>5.3 Top-N (Concentrated)</h3>
<p>Selects only the top N stocks by predicted probability. Calibration-independent: pure rank ordering. Tested with N = 5, 10, 20, 30, 50.</p>

<h3>5.4 Composite score</h3>
<p>Combines both signals: <code>score_long = p_join - &alpha; &middot; p_leave</code>. If a stock has high join probability but also high leave probability, something is off. This penalizes ambiguous cases and keeps only stocks where the two models agree. Tested with &alpha; = 0.25, 0.5, 0.75 and N = 5, 10, 20. See Composite-N rows in the <a href="#results">comparison table</a>.</p>

<h3>5.5 Asymmetric legs</h3>
<p>Different number of positions for each side (e.g., 5 long / 20 short). The idea: if the model is better at predicting one direction, concentrate there and diversify the other leg. See Asym rows in the <a href="#results">comparison table</a>.</p>

<h3>5.6 Volatility-scaled sizing</h3>
<p>Weights inversely proportional to stock volatility: <code>w<sub>i</sub> = score<sup>&gamma;</sup> / vol<sub>i</sub></code>. A high-conviction name that is very volatile gets less weight than one that is less volatile. The goal is equal risk per position rather than equal dollars. Tested with &gamma; = 0.0, 0.3, 0.5. See VolScaled rows in the <a href="#results">comparison table</a>.</p>

<h3>5.7 Momentum-filtered</h3>
<p>Only buys if the model says "join" and the stock is already trending up. Only shorts if the model says "leave" and the stock is already trending down. If model and price disagree, no trade. See MomFilter rows in the <a href="#results">comparison table</a>.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="results">6. Performance results</h1>

<!-- ── 6.1 Cumulative returns overview ─────────────────────────────── -->
<h3>6.1 Cumulative returns overview</h3>

<p>This chart shows how $1 invested grows over time for the best strategy in each category. Each line is a strategy; higher means more cumulative profit. A flat line means no returns; a dip means losses.</p>

{_figure(FIGURES / "cumulative_returns_comparison.png", "Figure 3: Cumulative returns comparison across key strategies")}

<p>The first thing that stands out is the gap between the Omniscient benchmark (blue, ~80x return) and everything else. The predictive strategies cluster near the bottom of the chart, but they are all positive. A few things to notice:</p>

<ul>
<li>The Omniscient (blue) has perfect foresight and still only gets a Sharpe of 0.97 with a 55% drawdown. This is a hard problem: even knowing exactly which stocks will join or leave, the price reactions are noisy and unpredictable.</li>
<li>Composite-5 (red) has the highest cumulative return among predictive strategies (~15x), but it also has the worst drawdown (81%). It nearly wipes out multiple times.</li>
<li>The predictive strategies only start generating meaningful returns around 2017, when the model's out-of-sample period begins and it has enough training data to produce useful signals.</li>
</ul>

<!-- ── 6.2 Which strategy is "best"? ──────────────────────────────── -->
<h3>6.2 Which strategy is "best"?</h3>

<p>It depends on what you care about. Different metrics point to different winners:</p>

<table>
<thead><tr><th>If you care about...</th><th>Best strategy</th><th>Value</th><th>Trade-off</th></tr></thead>
<tbody>
<tr><td><strong>Highest Sharpe ratio</strong> (risk-adjusted return)</td><td>Predictive (quantile)</td><td>0.609</td><td>Low return (3.8%), but very low volatility (6.5%) and drawdown (30%)</td></tr>
<tr><td><strong>Highest cumulative return</strong></td><td>Top-5 (probability)</td><td>9.6% annual</td><td>Extreme volatility (28%) and drawdown (70%)</td></tr>
<tr><td><strong>Lowest maximum drawdown</strong></td><td>Predictive (quantile)</td><td>29.9%</td><td>Broad diversification (~743 positions) dilutes signal</td></tr>
<tr><td><strong>Best composite signal</strong></td><td>Composite-5 (&alpha;=0.25)</td><td>9.2% annual, Sharpe 0.45</td><td>Worst drawdown of all strategies (81%)</td></tr>
</tbody>
</table>

<p>The pattern is clear: concentrated strategies (Top-5, Composite-5) make more money in absolute terms because they bet big on a few stocks. Diversified strategies (Quantile) have better risk-adjusted numbers because spreading across ~743 positions smooths out stock-specific noise. You pick based on how much drawdown you can tolerate.</p>

<!-- ── 6.3 Full comparison table ───────────────────────────────────── -->
<h3>6.3 Full strategy comparison</h3>

<p>All 33 strategies. The columns that matter most: annual_return (higher = more profit), sharpe_ratio (higher = better risk-adjusted), max_drawdown (lower = less painful losses).</p>

{_read_csv_as_table(TABLES / "strategy_comparison.csv", fmt=strat_fmt)}

<!-- ── 6.4 Best strategy deep-dive ────────────────────────────────── -->
<h3>6.4 Best risk-adjusted strategy up close</h3>

<p>The Predictive (quantile) strategy has the highest Sharpe (0.609). It takes the top 10% of stocks by predicted probability on each side, which means ~743 positions per leg, rebalanced monthly.</p>

<p>The return curve is slow but steady. Unlike the concentrated strategies, it does not spike or crash:</p>

{_figure(FIGURES / "cumulative_returns.png", "Figure 4: Cumulative returns for Quantile (top 10%)")}

<p>Its worst drawdown was 30%, during the COVID crash in March 2020. For context, the concentrated strategies (Top-5, Composite-5) drew down 70-81% in the same period:</p>

{_figure(FIGURES / "drawdown.png", "Figure 5: Drawdown profile for Quantile (top 10%)")}

<!-- ── 6.5 Omniscient benchmark ────────────────────────────────────── -->
<h3>6.5 The omniscient benchmark (the ceiling)</h3>

<p>The omniscient strategy knows the future. It buys stocks that will join and shorts stocks that will leave, with zero prediction error. And yet: Sharpe of 0.97, max drawdown of 55%. This tells you something important about the problem itself. Index reconstitution is not a clean signal. Stock prices around additions and deletions are noisy, and the timing of price moves is unpredictable even when the event itself is known.</p>

{_figure(FIGURES / "drawdown_omniscient.png", "Figure 6: Drawdown of the omniscient benchmark, which lost 55% peak-to-trough despite perfect information")}

<!-- ── 6.6 Operational metrics ─────────────────────────────────────── -->
<h3>6.6 Turnover and exposure</h3>

<p>Turnover is how much the portfolio changes at each rebalance. More turnover = more transaction costs eating into returns.</p>

{_figure(FIGURES / "turnover.png", "Figure 7: Portfolio turnover (21-day rolling average)")}

<p>Exposure tracks total long and short positions. The strategy targets gross exposure of 2.0 (1.0 long + 1.0 short) and net exposure of ~0 (dollar-neutral).</p>

{_figure(FIGURES / "exposure.png", "Figure 8: Gross and net exposure over time")}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="annual">7. Return trajectory over time</h1>

<p>The question any allocator asks: does the strategy work across different periods, or did it get lucky in one regime? These charts break down performance over time.</p>

<h3>7.1 Rolling 3-year annualised return</h3>
<p>Each bar is the annualised return over a 3-year window centered on that date. Green = positive, red = negative. Ideally the bars are all green and roughly the same height.</p>

{_figure(FIGURES / "annual_returns.png", "Figure 9: Rolling 3-year annualised return for best predictive strategy")}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h3>7.2 Rolling risk metrics</h3>

<p>Three panels showing how the risk profile changes over time. Top: Sharpe ratio (above 0 = making risk-adjusted money). Middle: max drawdown (deeper red = larger losses in that window). Bottom: volatility (spikes = market stress).</p>

{_figure(FIGURES / "rolling_metrics.png", "Figure 10: Rolling 3-year Sharpe, drawdown, and volatility")}

<h3>7.3 Subperiod Analysis (Rolling 3-Year Windows)</h3>
<p>Same data in table form. Each row is a 3-year window. Look at whether the numbers stay stable or swing wildly:</p>

{subperiod_table}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="robustness">8. Robustness analysis</h1>

<p>If the strategy only works with one specific parameter choice, it is probably overfit. We test the Top-N equal-weight strategy across two dimensions: holding period (1, 3, 6, 12 months) and number of positions (5, 10, 20, 30, 50). That gives 20 combinations.</p>

<p>In the heatmaps: green cells are good, red cells are bad. If the whole map is green, the strategy is robust to parameter choice. If only one corner is green, it is fragile.</p>

<h3>8.1 Robustness Grid</h3>
{robustness_table}

{_figure(FIGURES / "robustness_heatmap_sharpe_ratio.png", "Figure 11: Sharpe ratio across parameter combinations (brighter green = better)")}

{_figure(FIGURES / "robustness_heatmap_annual_return.png", "Figure 12: Annual return across parameter combinations (fewer positions = higher returns)")}

{_figure(FIGURES / "robustness_heatmap_max_drawdown.png", "Figure 13: Maximum drawdown across parameter combinations (darker red = larger losses)")}

<h3>8.2 What the robustness grid tells us</h3>

<p>Two clear patterns emerge from the grid:</p>

<p>First, the strategy only works with monthly rebalancing (holding period = 1 month). With quarterly, semi-annual, or annual rebalancing, returns drop to near zero or go negative. For example, Top-5 goes from 9.5% annual return with monthly rebalancing to -1.0% with quarterly. The model's signal decays fast: if you do not act on it within a month, it is already stale.</p>

<p>Second, fewer positions means better risk-adjusted returns. The Sharpe drops monotonically from 0.46 (N=5) to 0.14 (N=50) at monthly rebalancing. The model's predictive power is concentrated in its top-ranked stocks. Adding more positions just dilutes the signal with noise.</p>

<p>This is not a fully robust strategy in the traditional sense. It depends on two specific choices: monthly rebalancing and a small number of concentrated positions. But this is consistent with the nature of the signal. Index membership changes are infrequent events (~20 per year), and the anticipation premium is short-lived. A strategy designed to capture it needs to be nimble and selective.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="factors">9. Factor risk decomposition</h1>

<p>Are these returns real alpha, or just compensation for loading on known risk factors? A Fama-French 4-factor regression decomposes the portfolio's excess returns into:</p>

<table>
<thead><tr><th>Factor</th><th>Description</th></tr></thead>
<tbody>
<tr><td>MKT-RF</td><td>Market excess return</td></tr>
<tr><td>SMB</td><td>Small-minus-big (size)</td></tr>
<tr><td>HML</td><td>High-minus-low (value)</td></tr>
<tr><td>UMD</td><td>Up-minus-down (momentum)</td></tr>
</tbody>
</table>

<p>The regression model is:</p>
<p style="text-align:center;"><code>R<sub>p</sub> - R<sub>f</sub> = &alpha; + &beta;<sub>MKT</sub>(MKT-RF) + &beta;<sub>SMB</sub>(SMB) + &beta;<sub>HML</sub>(HML) + &beta;<sub>UMD</sub>(UMD) + &epsilon;</code></p>

<p>If the intercept (&alpha;) is positive and statistically significant, the strategy is generating returns that these four factors cannot explain. Since the portfolio is dollar-neutral by construction (equal long/short exposure), the market beta should be close to zero.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="interpretation">10. Economic interpretation</h1>

<h3>What we learned</h3>

<p>The omniscient benchmark, which knows every future S&amp;P 500 change before it happens, achieves a Sharpe of 0.97. That is not a great Sharpe for a strategy with perfect information. It tells us that index reconstitution is a noisy signal: stock prices around additions and deletions move unpredictably even when the event itself is certain.</p>

<p>Given that ceiling, the quantile-based strategy does well. Its Sharpe of 0.609 captures 63% of the omniscient's risk-adjusted return, with lower volatility (6.5% vs 16.3%) and a smaller drawdown (30% vs 55%). The diversification across ~743 positions per side helps a lot here.</p>

<p>Concentrated strategies (Top-5, Composite-5) earn more in absolute terms but with painful drawdowns (70-81%). The more positions you add, the worse the Sharpe gets, which confirms the model's signal is strongest at the top of the ranking and dilutes quickly.</p>

<p>The composite, vol-scaled, and momentum-filtered variants offer marginal improvements in specific dimensions but do not fundamentally change the picture. See the <a href="#results">comparison table</a> for all variants.</p>

<h3>The prediction-to-portfolio gap</h3>
<p>The model has a 0.94 AUC but only a 0.017 IC. Why? Class imbalance. Only ~2-4% of stocks change membership in any 63-day window. The model is good at identifying events vs non-events, but the cross-sectional ranking (which stocks will join first? which will leave first?) is much noisier, and it is the ranking that determines portfolio returns.</p>

<h3>Relation to academic literature</h3>
<p>Our results are consistent with the literature. The long leg captures the addition premium documented by Chen, Noronha & Singal (2004). The short leg profits from the deletion effect. Sharpe ratios are modest overall, in line with Petajisto (2011)'s finding that the index effect has diminished as markets have become more efficient.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="references">11. References</h1>
<ul>
<li>Chen, H., Noronha, G. & Singal, V. (2004). The price response to S&P 500 index additions and deletions. <em>Journal of Finance</em>, 59(4), 1901-1929.</li>
<li>Fama, E. & French, K. (1993). Common risk factors in the returns on stocks and bonds. <em>Journal of Financial Economics</em>, 33(1), 3-56.</li>
<li>Harris, L. & Gurel, E. (1986). Price and volume effects associated with changes in the S&P 500 list. <em>Journal of Finance</em>, 41(4), 815-829.</li>
<li>Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. <em>Journal of Finance</em>, 48(1), 65-91.</li>
<li>Petajisto, A. (2011). The index premium and its hidden cost for index funds. <em>Journal of Empirical Finance</em>, 18(2), 271-288.</li>
<li>Shleifer, A. (1986). Do demand curves for stocks slope down? <em>Journal of Finance</em>, 41(3), 579-590.</li>
<li>S&P Dow Jones Indices (2024). S&P U.S. Indices Methodology.</li>
</ul>

<hr style="margin: 3rem 0 1rem;">
<p style="text-align: center; color: #999; font-size: 0.85rem;">
Source code: <a href="https://github.com/CarlosAlonsoL/FinanceDataQuantPortfolio">GitHub Repository</a>
</p>

</body>
</html>"""

    return html


def build_institutional_rules_html() -> str:
    """Convert institutional rules markdown to HTML."""
    content = (DOCS / "SP500_INSTITUTIONAL_RULES.md").read_text()
    # Simple markdown-to-html conversion for this specific doc
    lines = content.split("\n")
    html_body = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            html_body.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_body.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("### "):
            html_body.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("| ") and "---" not in stripped:
            if not in_table:
                html_body.append("<table>")
                in_table = True
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                html_body.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
            else:
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                html_body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        elif stripped.startswith("|") and "---" in stripped:
            continue
        else:
            if in_table:
                html_body.append("</tbody></table>")
                in_table = False
            if stripped.startswith("- **"):
                html_body.append(f"<li>{stripped[2:]}</li>")
            elif stripped.startswith("- "):
                html_body.append(f"<li>{stripped[2:]}</li>")
            elif stripped:
                # Bold markdown
                import re
                processed = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
                processed = re.sub(r'\*(.+?)\*', r'<em>\1</em>', processed)
                html_body.append(f"<p>{processed}</p>")
    if in_table:
        html_body.append("</tbody></table>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>S&P 500 Institutional Rules</title>
    <style>{CSS}</style>
</head>
<body>
<p><a href="report.html" class="nav-link">&larr; Back to Main Report</a></p>
{"".join(html_body)}
<p><a href="report.html" class="nav-link">&larr; Back to Main Report</a></p>
</body>
</html>"""


def build_research_notes_html() -> str:
    """Convert research notes markdown to HTML."""
    content = (DOCS / "RESEARCH_NOTES.md").read_text()
    lines = content.split("\n")
    html_body = []
    in_table = False
    is_first_header_row = True
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            html_body.append("<hr>")
            continue
        if stripped.startswith("# "):
            html_body.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_body.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("### "):
            html_body.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("| ") and "---" not in stripped:
            if not in_table:
                html_body.append("<table>")
                in_table = True
                is_first_header_row = True
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                html_body.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
            else:
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                html_body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        elif stripped.startswith("|") and "---" in stripped:
            continue
        else:
            if in_table:
                html_body.append("</tbody></table>")
                in_table = False
            if stripped.startswith("- **"):
                html_body.append(f"<li>{stripped[2:]}</li>")
            elif stripped.startswith("- "):
                html_body.append(f"<li>{stripped[2:]}</li>")
            elif stripped:
                import re
                processed = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
                processed = re.sub(r'\*(.+?)\*', r'<em>\1</em>', processed)
                processed = re.sub(r'`(.+?)`', r'<code>\1</code>', processed)
                html_body.append(f"<p>{processed}</p>")
    if in_table:
        html_body.append("</tbody></table>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Research Notes & Academic Rationale</title>
    <style>{CSS}</style>
</head>
<body>
<p><a href="report.html" class="nav-link">&larr; Back to Main Report</a></p>
{"".join(html_body)}
<p><a href="report.html" class="nav-link">&larr; Back to Main Report</a></p>
</body>
</html>"""


def main():
    print("Generating HTML report...")

    report = build_main_report()
    out_path = RESULTS / "report.html"
    out_path.write_text(report)
    print(f"  Main report: {out_path}")

    inst = build_institutional_rules_html()
    inst_path = RESULTS / "institutional_rules.html"
    inst_path.write_text(inst)
    print(f"  Institutional rules: {inst_path}")

    research = build_research_notes_html()
    research_path = RESULTS / "research_notes.html"
    research_path.write_text(research)
    print(f"  Research notes: {research_path}")

    print("\nDone. Open results/report.html in a browser.")


if __name__ == "__main__":
    main()

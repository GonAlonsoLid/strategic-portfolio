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

<div class="toc">
<h2>Table of Contents</h2>
<ol>
    <li><a href="#institutional">Institutional Rules & S&P 500 Index</a></li>
    <li><a href="#methodology">Prediction Methodology</a></li>
    <li><a href="#model">Model Performance</a></li>
    <li><a href="#construction">Portfolio Construction Rules</a></li>
    <li><a href="#strategies">Strategy Variants</a></li>
    <li><a href="#results">Performance Results</a></li>
    <li><a href="#annual">Annual Returns Trajectory</a></li>
    <li><a href="#rolling">Rolling Performance Metrics</a></li>
    <li><a href="#robustness">Robustness Analysis</a></li>
    <li><a href="#interpretation">Economic Interpretation</a></li>
    <li><a href="#references">References</a></li>
</ol>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="institutional">1. Institutional Rules & S&P 500 Index</h1>

<p>The S&P 500 is a market-capitalization-weighted index of 500 large-cap U.S. equities maintained by S&P Dow Jones Indices. Unlike rules-based indices, the S&P 500 is managed by an <strong>Index Committee</strong> that exercises discretion in constituent selection.</p>

<h3>Eligibility Criteria for Inclusion</h3>
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

<h3>Reasons for Deletion</h3>
<ul>
<li>Market cap drops significantly below threshold (asymmetric buffer)</li>
<li>Corporate actions: mergers, acquisitions, leveraged buyouts, privatization</li>
<li>Bankruptcy, delisting, or failure to meet eligibility criteria</li>
</ul>

<h3>The S&P 500 Index Effect</h3>
<div class="highlight-box">
<strong>Addition effect:</strong> +3% to +7% abnormal return historically (diminished to ~1-2% in recent years). Driven by anticipated demand from index-tracking funds.<br>
<strong>Deletion effect:</strong> -5% to -15% abnormal return, larger and more persistent than additions. Often confounded with fundamental deterioration.
</div>

<p><a href="institutional_rules.html" class="nav-link">Full Institutional Rules Document &rarr;</a></p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="methodology">2. Prediction Methodology</h1>

<h3>Problem Formulation</h3>
<p>We formulate S&P 500 membership prediction as two binary classification problems:</p>
<ul>
<li><strong>Join prediction:</strong> P(stock joins S&P 500 within next 63 trading days | features at time t)</li>
<li><strong>Leave prediction:</strong> P(stock leaves S&P 500 within next 63 trading days | features at time t)</li>
</ul>

<h3>Feature Engineering</h3>
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

<h3>Feature Importance (SHAP)</h3>
{_figure(FIGURES / "shap_importance.png", "Figure 1: SHAP feature importance for join prediction model")}
<p>Market cap rank dominates, consistent with the primary eligibility criterion. Momentum and volatility features contribute secondary predictive power.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="model">3. Model Performance</h1>

<h3>Cross-Validation Results</h3>
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

<p>The model discriminates well between future joiners/leavers and non-events (high AUC), but the extreme class imbalance (~2-4% positive rate) means precision at any threshold is inherently low. The IC of 0.017 is modest but statistically significant across the 20-year out-of-sample period.</p>

<h3>Information Coefficient Decay</h3>
{_read_csv_as_table(TABLES / "ic_decay.csv", fmt={{"horizon": _int, "ic": _f4}})}
<p>IC increases with horizon (1d &rarr; 63d), confirming the model captures medium-term membership transitions rather than short-term noise.</p>

{_figure(FIGURES / "ic_decay.png", "Figure 2: IC decay across prediction horizons")}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="construction">4. Portfolio Construction Rules</h1>

<h3>Common Rules</h3>
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

<h3>Long Leg</h3>
<p>Stocks with the highest predicted probability of <strong>joining</strong> the S&P 500. These stocks are expected to benefit from anticipated index fund buying pressure and the signaling value of inclusion.</p>

<h3>Short Leg</h3>
<p>Stocks with the highest predicted probability of <strong>leaving</strong> the S&P 500. These stocks are expected to suffer from index fund selling pressure and the fundamental deterioration that often triggers deletion.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="strategies">5. Strategy Variants</h1>

<h3>5.1 Benchmark: Perfect Foresight (Omniscient)</h3>
<p>Uses realized future S&P 500 membership to construct the portfolio. Provides an upper bound on the alpha achievable if the investor had perfect knowledge of future index changes.</p>

<h3>5.2 Quantile-Based (Baseline)</h3>
<p>Selects the top decile (10%) of stocks by predicted probability for each leg. Yields ~743 stocks per side &mdash; diversified but diluted.</p>

<h3>5.3 Top-N (Concentrated)</h3>
<p>Selects only the top N stocks by predicted probability. Calibration-independent: pure rank ordering. Tested with N = 5, 10, 20, 30, 50.</p>

<h3>5.4 Composite Score (New)</h3>
<p>Combines join and leave signals: <code>score_long = p_join - &alpha; &middot; p_leave</code>. Penalizes stocks where both models give high probability (ambiguous signal). Filters out "confused" predictions, concentrating on stocks where models agree on direction.</p>

<h3>5.5 Asymmetric Legs (New)</h3>
<p>Uses different N for long vs short legs (e.g., 5 long / 30 short). Concentrates the long leg where signal is strongest, diversifies the short leg to reduce blow-up risk.</p>

<h3>5.6 Volatility-Scaled Sizing (New)</h3>
<p>Position sizes inversely proportional to stock volatility: <code>w<sub>i</sub> = score<sup>&gamma;</sup> / vol<sub>i</sub></code>. Targets equal risk contribution per position, reducing drawdowns from high-volatility names.</p>

<h3>5.7 Momentum-Filtered (New)</h3>
<p>Only enters long positions when recent momentum is positive (price confirming signal), short positions when momentum is negative. Filters out stocks where model prediction and price trend disagree.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="results">6. Performance Results</h1>

<h3>Full Strategy Comparison</h3>
{_read_csv_as_table(TABLES / "strategy_comparison.csv", fmt=strat_fmt)}

{_figure(FIGURES / "cumulative_returns_comparison.png", "Figure 3: Cumulative returns comparison across key strategies")}

{_figure(FIGURES / "cumulative_returns.png", "Figure 4: Cumulative returns for best predictive strategy")}

{_figure(FIGURES / "drawdown.png", "Figure 5: Drawdown profile of best predictive strategy")}

{_figure(FIGURES / "drawdown_omniscient.png", "Figure 6: Drawdown profile of omniscient benchmark")}

{_figure(FIGURES / "turnover.png", "Figure 7: Portfolio turnover over time")}

{_figure(FIGURES / "exposure.png", "Figure 8: Gross and net exposure over time")}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="annual">7. Annual Returns Trajectory</h1>

<p>Year-by-year performance reveals the strategy's behavior across different market regimes. Consistent risk-adjusted returns across time is the ultimate objective.</p>

{_figure(FIGURES / "annual_returns.png", "Figure 9: Annual returns by year for key strategies")}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="rolling">8. Rolling Performance Metrics</h1>

<p>Rolling 1-year Sharpe ratio and annual return show how the strategy's risk-adjusted performance evolves through different market environments.</p>

{_figure(FIGURES / "rolling_metrics.png", "Figure 10: Rolling 1-year Sharpe ratio and annual return")}

<h3>Subperiod Analysis (Rolling 3-Year Windows)</h3>
{_read_csv_as_table(TABLES / "backtest_subperiod_3y.csv", fmt={{
    "annual_return": _pct2, "annual_volatility": _pct2,
    "sharpe_ratio": _f3, "sortino_ratio": _f3,
    "max_drawdown": _pct2, "calmar_ratio": _f3,
    "var_05": _f4, "skewness": _f3,
}})}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="robustness">9. Robustness Analysis</h1>

<p>The strategy is tested across a grid of holding periods (1, 3, 6, 12 months) and position counts (5, 10, 20, 30, 50). This reveals which parameter combinations are robust and which are sensitive to specification.</p>

<h3>Robustness Grid</h3>
{_read_csv_as_table(TABLES / "robustness_holding_periods.csv", fmt={{
    "annual_return": _pct2, "annual_volatility": _pct2,
    "sharpe_ratio": _f3, "sortino_ratio": _f3,
    "max_drawdown": _pct2, "turnover": _f4,
    "n_positions_avg": _f2,
}})}

{_figure(FIGURES / "robustness_heatmap_sharpe_ratio.png", "Figure 11: Robustness heatmap &mdash; Sharpe ratio across holding period x position count")}

{_figure(FIGURES / "robustness_heatmap_annual_return.png", "Figure 12: Robustness heatmap &mdash; Annual return")}

{_figure(FIGURES / "robustness_heatmap_max_drawdown.png", "Figure 13: Robustness heatmap &mdash; Maximum drawdown")}

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h1 id="interpretation">10. Economic Interpretation</h1>

<h3>Key Findings</h3>

<div class="highlight-box">
<ol>
<li><strong>The predictive model generates positive risk-adjusted returns.</strong> The best strategy captures a meaningful fraction of the omniscient benchmark's Sharpe ratio, confirming that S&P 500 membership changes are partially predictable.</li>
<li><strong>Concentration increases returns but at the cost of higher risk.</strong> Top-5 delivers the highest annual return but with extreme volatility and drawdowns. Broader strategies offer better risk-return tradeoffs.</li>
<li><strong>Advanced portfolio construction improves performance.</strong> Composite scoring, volatility-scaled sizing, and momentum filtering each contribute incremental improvements by reducing noise and managing risk more efficiently.</li>
<li><strong>Signal quality degrades with portfolio size.</strong> Sharpe ratios decline monotonically as N increases, consistent with the model's signal being strongest for the highest-ranked stocks.</li>
</ol>
</div>

<h3>The Prediction-Implementation Gap</h3>
<p>Despite high AUC (0.94), the IC is modest (0.017). This gap arises from extreme class imbalance: only ~2-4% of stocks experience membership changes in any 63-day window. The model discriminates well between events and non-events, but the cross-sectional ranking information &mdash; which determines portfolio performance &mdash; is inherently noisier.</p>

<h3>Consistency with Academic Literature</h3>
<ul>
<li><strong>Addition effect:</strong> Long positions in predicted joiners capture the anticipation premium documented by Chen, Noronha & Singal (2004)</li>
<li><strong>Deletion effect:</strong> Short positions in predicted leavers profit from selling pressure and fundamental deterioration</li>
<li><strong>Signal decay with dilution:</strong> As portfolio size increases, signal-to-noise deteriorates, consistent with the information ratio literature</li>
<li><strong>Diminishing alpha:</strong> The modest Sharpe ratios reflect the well-documented decline in the index effect as markets have become more efficient (Petajisto 2011)</li>
</ul>

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
Supplementary: <a href="institutional_rules.html">Full Institutional Rules</a> |
<a href="research_notes.html">Research Notes & Academic Rationale</a>
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

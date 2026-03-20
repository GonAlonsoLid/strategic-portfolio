"""Build daily panel with returns, market cap, and S&P 500 membership."""
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.load_data import load_events, load_prices_chunked, load_config_paths, build_ticker_permno_bridge


def build_daily_panel(
    config: dict | None = None,
    *,
    output_path: str | Path | None = None,
    date_min: str = "1990-01-01",
    date_max: str | None = None,
    chunksize: int = 1_000_000,
    bridge_sample: int = 2_000_000,
    max_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """Build daily panel with date, permno, ticker, ret, market_cap, volume, is_sp500.

    - Loads events from Excel and daily.csv in chunks.
    - Builds Ticker-PERMNO bridge from prices.
    - Assigns is_sp500: True for constituents before 1995-01-01 (from first appearance in data)
      and toggles from events by effective_date.
    - Writes data/interim/daily_panel.parquet. If max_chunks is set, only that many
      chunks are read (for testing on large daily.csv).
    """
    from src.utils.config_loader import load_config
    cfg = config or load_config()
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent.parent
    interim = base / paths.get("interim", "data/interim")
    interim.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path or interim / "daily_panel.parquet")

    events = load_events(config=cfg)
    events["effective_date"] = pd.to_datetime(events["effective_date"]).dt.normalize()

    # Bridge: ticker (normalized) -> permno. Use latest date per permno/ticker.
    bridge = build_ticker_permno_bridge(config=cfg)
    # Map event ticker -> permno (take any permno that ever had this ticker_norm)
    ticker_to_permno = bridge.groupby("ticker")["permno"].first().to_dict()

    # Initial constituents: firms that appear in data before 1995-01-01 and are not in events as ADD later
    # We'll set is_sp500 = True for all that are in the index on each date.
    # Build index membership by date: set of permnos that are in SP500 on that date.
    # 1) Start with empty. 2) For each date in order, apply leaves then adds. 3) For dates before first event, we need "initial" set.
    # Simplified: get all ADD events and all DEL events. Initial = we don't have full history; assume we only track from 1995.
    # So: from 1995-01-01 onward, for each day, membership = previous day membership - leavers effective that day + joiners effective that day.
    # Initial membership at 1994-12-31: we need a list. Plan says "firms in the index before Jan 1 1995 should be treated as initial constituents".
    # We don't have a list of 1994 constituents. Approximate: take all permnos that ever appear in our daily data and that are NOT in the ADD list (they were already there). So initial = all permnos in data before 1995 minus those that joined after 1995. That's expensive. Simpler: initial = empty; then on each ADD we add, on each DEL we remove. So we only have membership from 1995 onward for firms that joined or left. For firms that were always there we never add them. So we need initial constituents. From CRSP/Refinitiv we'd have index membership. Here we don't. Alternative: treat "initial" as everyone who appears in the first year of data (1995) and is not in the ADD list in 1995. So: initial = set(permnos in 1995 data) - set(permnos that joined in 1995). Then apply DEL/ADD going forward.
    # Implement: collect all permnos from 1995 data; collect all permnos that are ADD in 1995. initial_constituents = permnos in 1995 - add_1995. Then for each date d >= 1995-01-01: members = prev_members - leavers_d + joiners_d.

    # Build event lists with permno
    events["permno"] = events["ticker"].map(ticker_to_permno)
    events = events.dropna(subset=["permno"]).copy()
    events["permno"] = events["permno"].astype(int)
    add_events = events[events["event_type"] == "ADD"][["effective_date", "permno"]].drop_duplicates()
    del_events = events[events["event_type"] == "DEL"][["effective_date", "permno"]].drop_duplicates()

    # We'll build panel in chunks and at the end compute is_sp500 per (date, permno)
    # First pass: stream daily data and collect (date, permno, ret, cap, volume, ticker)
    # Second pass: we need initial constituents. So first chunk we take 1995 only and get permnos; subtract ADD in 1995.
    chunks = []
    chunk_iter = load_prices_chunked(config=cfg, chunksize=chunksize, date_min=date_min, date_max=date_max)
    for i, ch in enumerate(chunk_iter):
        if max_chunks is not None and i >= max_chunks:
            break
        if "PERMNO" not in ch.columns:
            continue
        # Use "date" if loader already added it, else use DlyCalDt
        if "date" not in ch.columns and "DlyCalDt" in ch.columns:
            ch["date"] = pd.to_datetime(ch["DlyCalDt"], errors="coerce").dt.normalize()
        # Handle duplicate column names (e.g. from CSV)
        if isinstance(ch["date"], pd.DataFrame):
            ch["date"] = ch["date"].iloc[:, 0]
        ch = ch.rename(columns={
            "PERMNO": "permno",
            "DlyRet": "ret",
            "DlyCap": "market_cap",
            "DlyVol": "volume",
        })
        for c in ["ret", "market_cap", "volume"]:
            if c not in ch.columns:
                ch[c] = None
        if "Ticker" in ch.columns:
            ch["ticker"] = ch["Ticker"].astype(str).str.strip().str.upper()
        else:
            ch["ticker"] = ""
        ch = ch[["date", "permno", "ticker", "ret", "market_cap", "volume"]].copy()
        ch["permno"] = ch["permno"].astype(int)
        ch["date"] = pd.to_datetime(ch["date"], errors="coerce").dt.normalize()
        chunks.append(ch)
        if len(chunks) >= 100:
            panel = pd.concat(chunks, ignore_index=True)
            chunks = [panel]

    if chunks:
        panel = pd.concat(chunks, ignore_index=True)
    else:
        panel = pd.DataFrame(columns=["date", "permno", "ticker", "ret", "market_cap", "volume"])

    # Drop duplicates and sort
    panel = panel.drop_duplicates(subset=["date", "permno"]).sort_values(["date", "permno"]).reset_index(drop=True)

    # Initial constituents: the dataset includes 483 ADD events on 1994-12-30 representing
    # the initial S&P 500 members. Start with empty membership and let the event loop handle
    # everything (including those 1994-12-30 ADD events).
    # Build daily membership: for each date, set of permnos in index
    all_dates = panel["date"].unique()
    all_dates = sorted(all_dates)
    # Sparse: only store (date, permno) where is_sp500 True
    membership = set()
    date_members: dict[pd.Timestamp, set] = {}
    for d in all_dates:
        # Apply DEL on this date
        leavers = del_events[del_events["effective_date"] == d]["permno"].tolist()
        for p in leavers:
            membership.discard(p)
        # Apply ADD on this date
        joiners = add_events[add_events["effective_date"] == d]["permno"].tolist()
        for p in joiners:
            membership.add(p)
        date_members[d] = set(membership)

    # Map (date, permno) -> is_sp500
    panel["is_sp500"] = panel.apply(lambda r: r["permno"] in date_members.get(r["date"], set()), axis=1)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        panel.to_parquet(out_path, index=False)
    except ImportError:
        panel.to_csv(out_path.with_suffix(".csv"), index=False)

    return panel

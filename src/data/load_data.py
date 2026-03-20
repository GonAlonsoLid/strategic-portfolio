"""Load raw events (Excel) and daily prices (CSV)."""
from pathlib import Path
from typing import Iterator

import pandas as pd

# Excel sheet "L&J" has header at row 3: Status, Issuer, Code, Date
# Status = Joiner | Leaver, Code = ticker (e.g. CIEN.N), Date = event date
EVENTS_HEADER_ROW = 3
EVENTS_COL_MAP = {
    "Status": "event_type",
    "Issuer": "issuer",
    "Code": "ticker_raw",
    "Date": "event_date",
}


def load_config_paths(config: dict | None = None) -> dict:
    """Get paths from config; if None, load default config."""
    if config is None:
        from src.utils.config_loader import load_config
        config = load_config()
    return config.get("paths", {})


def load_events(
    path: str | Path | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Load S&P 500 joiners/leavers from Excel.

    Reads sheet 'L&J', header at row 3. Normalizes to:
    - event_type: ADD | DEL (from Joiner | Leaver)
    - ticker: normalized ticker (strip .N, .OQ, ^B26 etc. for matching)
    - event_date: datetime (used as both announcement and effective for lack of separate fields)
    - announcement_date, effective_date: same as event_date

    Args:
        path: Path to Excel file. If None, uses config paths.raw_events.
        config: Config dict; if None, load_config() is used.

    Returns:
        DataFrame with columns event_type, ticker, ticker_raw, event_date, announcement_date, effective_date.
    """
    if path is None:
        paths = load_config_paths(config)
        base = Path(__file__).resolve().parent.parent.parent
        path = base / paths.get("raw_events", "data/raw/SPX_index leavers & joiners_17-Feb-2026.xlsx")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    df = pd.read_excel(path, sheet_name="L&J", header=EVENTS_HEADER_ROW, engine="openpyxl")
    # Normalize column names (Excel may have different spacing)
    rename = {}
    for c in df.columns:
        cstr = str(c).strip()
        for orig, new in EVENTS_COL_MAP.items():
            if orig in cstr or cstr == orig:
                rename[c] = new
                break
    df = df.rename(columns=rename)

    # Require at least event_type and date-like column
    if "event_type" not in df.columns:
        # Try Status
        if "Status" in df.columns:
            df["event_type"] = df["Status"]
        else:
            raise ValueError("Events Excel must have Status (Joiner/Leaver) column.")
    if "event_date" not in df.columns:
        # Find first datetime column
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df["event_date"] = pd.to_datetime(df[c])
                break
        if "event_date" not in df.columns:
            raise ValueError("Events Excel must have a date column.")

    # Normalize event_type to ADD / DEL
    df["event_type"] = df["event_type"].astype(str).str.strip().str.upper()
    df.loc[df["event_type"].str.contains("JOIN", na=False), "event_type"] = "ADD"
    df.loc[df["event_type"].str.contains("LEAV", na=False), "event_type"] = "DEL"
    df = df[df["event_type"].isin(["ADD", "DEL"])].copy()

    # Ticker: strip exchange suffix for matching with CRSP (e.g. CIEN.N -> CIEN, VSNT.OQ -> VSNT)
    if "ticker_raw" not in df.columns and "Code" in df.columns:
        df["ticker_raw"] = df["Code"].astype(str)
    df["ticker_raw"] = df["ticker_raw"].astype(str).str.strip()
    df["ticker"] = df["ticker_raw"].str.replace(r"\.[A-Z]+$", "", regex=True).str.replace(r"\^.*$", "", regex=True)

    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    df["announcement_date"] = df["event_date"]
    df["effective_date"] = df["event_date"]

    out_cols = ["event_type", "ticker", "ticker_raw", "event_date", "announcement_date", "effective_date"]
    if "issuer" in df.columns:
        out_cols.insert(2, "issuer")
    return df[[c for c in out_cols if c in df.columns]].dropna(subset=["event_date", "ticker"]).reset_index(drop=True)


def load_prices_chunked(
    path: str | Path | None = None,
    config: dict | None = None,
    chunksize: int = 500_000,
    usecols: list[str] | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Iterate over daily.csv in chunks to avoid OOM.

    Default columns: PERMNO, DlyCalDt, DlyPrc, DlyRet, DlyCap, DlyVol, ShrOut, Ticker,
    vwretd, ewretd, sprtrn (for benchmark).
    """
    if path is None:
        paths = load_config_paths(config)
        base = Path(__file__).resolve().parent.parent.parent
        path = base / paths.get("raw_prices", "data/raw/daily.csv")
    path = Path(path)
    if usecols is None:
        usecols = [
            "PERMNO", "DlyCalDt", "DlyPrc", "DlyRet", "DlyCap", "DlyVol", "ShrOut",
            "Ticker", "vwretd", "ewretd", "sprtrn",
        ]
    # Only read columns that exist
    first = pd.read_csv(path, nrows=0)
    usecols = [c for c in usecols if c in first.columns]
    if "DlyCalDt" not in usecols and "DlyCalDt" in first.columns:
        usecols.append("DlyCalDt")

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, parse_dates=["DlyCalDt"], low_memory=False)
    for chunk in reader:
        if "DlyCalDt" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["DlyCalDt"]).dt.normalize()
        if date_min:
            chunk = chunk[chunk["date"] >= pd.Timestamp(date_min)]
        if date_max:
            chunk = chunk[chunk["date"] <= pd.Timestamp(date_max)]
        if len(chunk) == 0:
            continue
        yield chunk


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def build_ticker_permno_bridge(
    prices_path: str | Path | None = None,
    config: dict | None = None,
    sample_dates: int = 500,
) -> pd.DataFrame:
    """Build a Ticker <-> PERMNO bridge from daily.csv (one row per PERMNO with latest Ticker).

    Reads the full file in chunks to cover all PERMNOs. Returns columns: permno, ticker, date.
    """
    raw_path = prices_path or load_config_paths(config).get("raw_prices", "data/raw/daily.csv")
    base = _project_root() / raw_path if isinstance(raw_path, str) else Path(raw_path)
    usecols = ["PERMNO", "DlyCalDt", "Ticker"]
    first = pd.read_csv(base, nrows=0)
    usecols = [c for c in usecols if c in first.columns]
    # Read full file in chunks to collect all unique PERMNO-Ticker pairs across the entire dataset.
    # nrows=2_000_000 was insufficient (covered only ~780 of ~10,000+ PERMNOs in a 58M-row file).
    accumulated: list[pd.DataFrame] = []
    reader = pd.read_csv(base, usecols=usecols, chunksize=5_000_000, parse_dates=["DlyCalDt"], low_memory=False)
    for ch in reader:
        ch = ch.rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "Ticker": "ticker"})
        ch["ticker"] = ch["ticker"].astype(str).str.strip().str.upper()
        ch["ticker_norm"] = ch["ticker"].str.replace(r"\.[A-Z]+$", "", regex=True).str.replace(r"\^.*$", "", regex=True)
        accumulated.append(ch[["permno", "ticker_norm", "date"]].drop_duplicates(subset=["permno", "ticker_norm"], keep="last"))
    chunk = pd.concat(accumulated, ignore_index=True)
    bridge = chunk.drop_duplicates(subset=["permno", "ticker_norm"], keep="last").rename(columns={"ticker_norm": "ticker"})
    return bridge

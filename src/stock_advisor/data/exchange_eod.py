from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from io import BytesIO
import logging
import os
from typing import Any
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
EXCHANGE_EOD_ENABLED = os.getenv("EXCHANGE_EOD_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
EXCHANGE_EOD_LOOKBACK_DAYS = int(os.getenv("EXCHANGE_EOD_LOOKBACK_DAYS", "7"))
EXCHANGE_EOD_TIMEOUT_SECONDS = float(os.getenv("EXCHANGE_EOD_TIMEOUT_SECONDS", "20"))
NSE_BHAVCOPY_URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date:%Y%m%d}_F_0000.csv.zip"
BSE_BHAVCOPY_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{date:%Y%m%d}_F_0000.CSV"
EXCHANGE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,application/zip,application/octet-stream,*/*",
}


def get_latest_exchange_eod_rows(tickers: list[str] | tuple[str, ...], *, lookback_days: int | None = None) -> dict[str, pd.DataFrame]:
    """Return latest official NSE/BSE EOD rows for tickers keyed by input ticker.

    NSE is preferred for ``.NS`` tickers. BSE is used for ``.BO`` tickers and as a
    fallback when an NSE symbol is missing from the NSE bhavcopy.
    """
    if not EXCHANGE_EOD_ENABLED:
        return {}
    ticker_keys = list(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers if _normalize_ticker(ticker)))
    if not ticker_keys:
        return {}

    for trade_date in _candidate_trade_dates(lookback_days=lookback_days):
        rows = get_exchange_eod_rows_for_date(ticker_keys, trade_date)
        if rows:
            return rows
    return {}


def get_exchange_eod_rows_for_date(tickers: list[str] | tuple[str, ...], trade_date: date) -> dict[str, pd.DataFrame]:
    """Return official NSE/BSE EOD rows for a specific date keyed by input ticker.

    .BO tickers are keyed by BSE's numeric scrip code (FinInstrmId in the bhavcopy), not by
    TckrSymb — BSE's bhavcopy symbol column is always the short text symbol (e.g. "ABB"), which
    never matches a "544412.BO"-style ticker. Matching by scrip code instead of symbol is what
    makes the ~2,600 numeric-code BSE tickers in instrument_master resolvable here at all.
    """
    ticker_keys = list(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers if _normalize_ticker(ticker)))
    if not ticker_keys:
        return {}

    nse_symbols = {_ticker_symbol(ticker) for ticker in ticker_keys if not ticker.endswith(".BO")}
    bse_scrip_codes = {_ticker_symbol(ticker) for ticker in ticker_keys if ticker.endswith(".BO")}
    nse_rows = _bhavcopy_rows_by_key(_fetch_nse_bhavcopy(trade_date), nse_symbols, key_column="symbol") if nse_symbols else {}

    # NSE-missing fallback still looks up by text symbol (an NSE ticker's symbol, not a scrip
    # code), so it stays keyed on "symbol" — separate from the .BO scrip-code lookup above.
    missing_nse_symbols = {
        _ticker_symbol(ticker)
        for ticker in ticker_keys
        if not ticker.endswith(".BO") and _ticker_symbol(ticker) not in nse_rows
    }
    bse_rows_by_symbol = (
        _bhavcopy_rows_by_key(_fetch_bse_bhavcopy(trade_date), missing_nse_symbols, key_column="symbol")
        if missing_nse_symbols
        else {}
    )
    bse_rows_by_scrip_code = (
        _bhavcopy_rows_by_key(_fetch_bse_bhavcopy(trade_date), bse_scrip_codes, key_column="scrip_code")
        if bse_scrip_codes
        else {}
    )

    out: dict[str, pd.DataFrame] = {}
    for ticker in ticker_keys:
        symbol = _ticker_symbol(ticker)
        row = bse_rows_by_scrip_code.get(symbol) if ticker.endswith(".BO") else nse_rows.get(symbol) or bse_rows_by_symbol.get(symbol)
        if row is None:
            continue
        frame = pd.DataFrame([row])
        frame.attrs["provider"] = row["provider"]
        frame.attrs["selected_ticker"] = ticker
        frame.attrs["exchange_trade_date"] = str(row["date"].date())
        out[ticker] = frame[["date", "open", "high", "low", "close", "volume", "turnover"]]
        out[ticker].attrs.update(frame.attrs)
    return out


def bhavcopy_fetch_succeeded_for_date(trade_date: date) -> bool:
    """True if at least one exchange's bhavcopy actually returned data for this date.

    A day-level connectivity signal, deliberately independent of which tickers a caller happens
    to be filtering for — get_exchange_eod_rows_for_date's *filtered* result can legitimately be
    empty on a perfectly healthy day (none of the requested tickers traded, or none are in this
    exchange's file), and that must not be confused with the bhavcopy fetch itself having failed
    (network error, holiday, exchange outage). Callers doing a day-coverage sanity check (e.g.
    before treating a run of "no data" days as evidence of ticker dormancy) should use this, not
    the presence/absence of rows in a filtered result.

    Free to call after get_exchange_eod_rows_for_date for the same date — both underlying fetches
    are lru_cache'd, so this never triggers an extra network request.
    """
    return not _fetch_nse_bhavcopy(trade_date).empty or not _fetch_bse_bhavcopy(trade_date).empty


def clear_exchange_eod_fetch_cache() -> None:
    """Clear in-memory NSE/BSE bhavcopy fetch caches before a forced refresh."""
    _fetch_nse_bhavcopy.cache_clear()
    _fetch_bse_bhavcopy.cache_clear()


def _candidate_trade_dates(*, lookback_days: int | None = None) -> list[date]:
    days = max(1, int(lookback_days or EXCHANGE_EOD_LOOKBACK_DAYS))
    now = datetime.now(MARKET_TIMEZONE)
    start = now.date()
    if now.time() < time(17, 30):
        start -= timedelta(days=1)

    candidates: list[date] = []
    cursor = start
    while len(candidates) < days:
        if cursor.weekday() < 5:
            candidates.append(cursor)
        cursor -= timedelta(days=1)
    return candidates


@lru_cache(maxsize=16)
def _fetch_nse_bhavcopy(trade_date: date) -> pd.DataFrame:
    url = NSE_BHAVCOPY_URL.format(date=trade_date)
    try:
        response = requests.get(url, headers={**EXCHANGE_HEADERS, "Referer": "https://www.nseindia.com/"}, timeout=EXCHANGE_EOD_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        logger.info("NSE bhavcopy fetch failed for %s: %s", trade_date, exc)
        return pd.DataFrame()
    if response.status_code != 200 or not response.content.startswith(b"PK"):
        logger.info("NSE bhavcopy unavailable for %s: status=%s", trade_date, response.status_code)
        return pd.DataFrame()
    try:
        with ZipFile(BytesIO(response.content)) as archive:
            csv_name = next((name for name in archive.namelist() if name.lower().endswith(".csv")), None)
            if not csv_name:
                return pd.DataFrame()
            with archive.open(csv_name) as csv_file:
                raw = pd.read_csv(csv_file)
    except (BadZipFile, StopIteration, ValueError, OSError) as exc:
        logger.info("NSE bhavcopy parse failed for %s: %s", trade_date, exc)
        return pd.DataFrame()
    return _normalize_bhavcopy(raw, provider="nse_bhavcopy")


@lru_cache(maxsize=16)
def _fetch_bse_bhavcopy(trade_date: date) -> pd.DataFrame:
    url = BSE_BHAVCOPY_URL.format(date=trade_date)
    try:
        response = requests.get(url, headers={**EXCHANGE_HEADERS, "Referer": "https://www.bseindia.com/"}, timeout=EXCHANGE_EOD_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        logger.info("BSE bhavcopy fetch failed for %s: %s", trade_date, exc)
        return pd.DataFrame()
    if response.status_code != 200 or response.content.lstrip().startswith(b"<"):
        logger.info("BSE bhavcopy unavailable for %s: status=%s", trade_date, response.status_code)
        return pd.DataFrame()
    try:
        raw = pd.read_csv(BytesIO(response.content))
    except (ValueError, OSError) as exc:
        logger.info("BSE bhavcopy parse failed for %s: %s", trade_date, exc)
        return pd.DataFrame()
    return _normalize_bhavcopy(raw, provider="bse_bhavcopy")


def _normalize_bhavcopy(raw: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    required = {"TradDt", "TckrSymb", "OpnPric", "HghPric", "LwPric", "ClsPric"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["TradDt"], errors="coerce"),
            "symbol": raw["TckrSymb"].astype(str).str.strip().str.upper(),
            "scrip_code": raw.get("FinInstrmId", pd.Series([None] * len(raw))).map(_clean_scrip_code),
            "series": raw.get("SctySrs", pd.Series([""] * len(raw))).astype(str).str.strip().str.upper(),
            "instrument_type": raw.get("FinInstrmTp", pd.Series([""] * len(raw))).astype(str).str.strip().str.upper(),
            "open": pd.to_numeric(raw["OpnPric"], errors="coerce"),
            "high": pd.to_numeric(raw["HghPric"], errors="coerce"),
            "low": pd.to_numeric(raw["LwPric"], errors="coerce"),
            "close": pd.to_numeric(raw["ClsPric"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("TtlTradgVol", pd.Series([None] * len(raw))), errors="coerce"),
            "turnover": pd.to_numeric(raw.get("TtlTrfVal", pd.Series([None] * len(raw))), errors="coerce"),
            "provider": provider,
        }
    )
    out = out.dropna(subset=["date", "symbol", "open", "high", "low", "close"])
    if "instrument_type" in out.columns:
        out = out[out["instrument_type"].isin({"", "STK"})].copy()
    return out


def _bhavcopy_rows_by_key(df: pd.DataFrame, keys: set[str], *, key_column: str) -> dict[str, dict[str, Any]]:
    if df.empty or not keys or key_column not in df.columns:
        return {}
    selected = df[df[key_column].isin(keys)].copy()
    if selected.empty:
        return {}
    selected["series_rank"] = selected["series"].map({"EQ": 0, "BE": 1, "BZ": 2, "SM": 3, "ST": 4}).fillna(9)
    selected = selected.sort_values([key_column, "series_rank"]).drop_duplicates(key_column, keep="first")
    return {str(getattr(row, key_column)): row._asdict() for row in selected.itertuples(index=False)}


def _clean_scrip_code(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _ticker_symbol(ticker: str) -> str:
    return _normalize_ticker(ticker).removesuffix(".NS").removesuffix(".BO")

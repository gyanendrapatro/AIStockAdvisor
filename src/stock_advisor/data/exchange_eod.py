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
    """Return official NSE/BSE EOD rows for a specific date keyed by input ticker."""
    ticker_keys = list(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers if _normalize_ticker(ticker)))
    if not ticker_keys:
        return {}

    nse_symbols = {_ticker_symbol(ticker) for ticker in ticker_keys if not ticker.endswith(".BO")}
    bse_symbols = {_ticker_symbol(ticker) for ticker in ticker_keys if ticker.endswith(".BO")}
    nse_rows = _bhavcopy_rows_by_symbol(_fetch_nse_bhavcopy(trade_date), nse_symbols) if nse_symbols else {}

    missing_nse_symbols = {
        _ticker_symbol(ticker)
        for ticker in ticker_keys
        if not ticker.endswith(".BO") and _ticker_symbol(ticker) not in nse_rows
    }
    bse_lookup_symbols = bse_symbols | missing_nse_symbols
    bse_rows = _bhavcopy_rows_by_symbol(_fetch_bse_bhavcopy(trade_date), bse_lookup_symbols) if bse_lookup_symbols else {}

    out: dict[str, pd.DataFrame] = {}
    for ticker in ticker_keys:
        symbol = _ticker_symbol(ticker)
        row = bse_rows.get(symbol) if ticker.endswith(".BO") else nse_rows.get(symbol) or bse_rows.get(symbol)
        if row is None:
            continue
        frame = pd.DataFrame([row])
        frame.attrs["provider"] = row["provider"]
        frame.attrs["selected_ticker"] = ticker
        frame.attrs["exchange_trade_date"] = str(row["date"].date())
        out[ticker] = frame[["date", "open", "high", "low", "close", "volume"]]
        out[ticker].attrs.update(frame.attrs)
    return out


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
            "series": raw.get("SctySrs", pd.Series([""] * len(raw))).astype(str).str.strip().str.upper(),
            "instrument_type": raw.get("FinInstrmTp", pd.Series([""] * len(raw))).astype(str).str.strip().str.upper(),
            "open": pd.to_numeric(raw["OpnPric"], errors="coerce"),
            "high": pd.to_numeric(raw["HghPric"], errors="coerce"),
            "low": pd.to_numeric(raw["LwPric"], errors="coerce"),
            "close": pd.to_numeric(raw["ClsPric"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("TtlTradgVol", pd.Series([None] * len(raw))), errors="coerce"),
            "provider": provider,
        }
    )
    out = out.dropna(subset=["date", "symbol", "open", "high", "low", "close"])
    if "instrument_type" in out.columns:
        out = out[out["instrument_type"].isin({"", "STK"})].copy()
    return out


def _bhavcopy_rows_by_symbol(df: pd.DataFrame, symbols: set[str]) -> dict[str, dict[str, Any]]:
    if df.empty or not symbols:
        return {}
    selected = df[df["symbol"].isin(symbols)].copy()
    if selected.empty:
        return {}
    selected["series_rank"] = selected["series"].map({"EQ": 0, "BE": 1, "BZ": 2, "SM": 3, "ST": 4}).fillna(9)
    selected = selected.sort_values(["symbol", "series_rank"]).drop_duplicates("symbol", keep="first")
    return {str(row.symbol): row._asdict() for row in selected.itertuples(index=False)}


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _ticker_symbol(ticker: str) -> str:
    return _normalize_ticker(ticker).removesuffix(".NS").removesuffix(".BO")

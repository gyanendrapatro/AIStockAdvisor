from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from io import StringIO
import logging
import os
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
NSE_INDEX_ARCHIVE_ENABLED = os.getenv("NSE_INDEX_ARCHIVE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
NSE_INDEX_ARCHIVE_TIMEOUT_SECONDS = float(os.getenv("NSE_INDEX_ARCHIVE_TIMEOUT_SECONDS", "15"))
NSE_INDEX_ARCHIVE_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{date:%d%m%Y}.csv"
NSE_INDEX_ARCHIVE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,*/*",
    "Referer": "https://www.nseindia.com/",
}

NSE_INDEX_NAME_ALIASES = {
    "Nifty Consumption": "Nifty India Consumption",
    "Nifty Digital": "Nifty India Digital",
    "Nifty EV": "Nifty EV & New Age Automotive",
    "Nifty Healthcare": "Nifty Healthcare Index",
    "Nifty Manufacturing": "Nifty India Manufacturing",
    "Nifty Defence": "Nifty India Defence",
    "Nifty Metals & Mining": "Nifty Metal",
    "Nifty Tourism": "Nifty India Tourism",
}


def get_nse_index_price_history(index_name: str, *, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch official NSE index OHLC history from daily index-close archives."""
    if not NSE_INDEX_ARCHIVE_ENABLED:
        return pd.DataFrame()

    resolved_name = resolve_nse_index_name(index_name)
    rows = []
    for trade_date in _candidate_dates(period):
        daily = _fetch_daily_index_archive(trade_date)
        if daily.empty:
            continue
        match = daily[daily["index_name"].str.casefold() == resolved_name.casefold()]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "date": pd.to_datetime(row["date"], errors="coerce"),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    out = out.drop_duplicates("date", keep="last").reset_index(drop=True)
    out.attrs["provider"] = "nse_index_archive"
    out.attrs["selected_ticker"] = f"NSE_INDEX:{resolved_name}"
    out.attrs["nse_index_name"] = resolved_name
    if str(interval).strip().lower() in {"1wk", "1w", "weekly"}:
        out = _resample_index_history(out, "W-FRI")
    elif str(interval).strip().lower() in {"1mo", "1m", "monthly"}:
        out = _resample_index_history(out, "ME")
    return out


def resolve_nse_index_name(index_name: str) -> str:
    name = str(index_name or "").strip()
    return NSE_INDEX_NAME_ALIASES.get(name, name)


def _candidate_dates(period: str) -> list[date]:
    period_days = _period_days(period) or 365
    now = datetime.now(MARKET_TIMEZONE)
    end_date = now.date()
    if now.time() < time(17, 30):
        end_date -= timedelta(days=1)
    start_date = end_date - timedelta(days=period_days + 20)

    dates = []
    cursor = start_date
    while cursor <= end_date:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor += timedelta(days=1)
    return dates


@lru_cache(maxsize=512)
def _fetch_daily_index_archive(trade_date: date) -> pd.DataFrame:
    url = NSE_INDEX_ARCHIVE_URL.format(date=trade_date)
    try:
        response = requests.get(url, headers=NSE_INDEX_ARCHIVE_HEADERS, timeout=NSE_INDEX_ARCHIVE_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        logger.info("NSE index archive fetch failed for %s: %s", trade_date, exc)
        return pd.DataFrame()
    if response.status_code != 200 or not response.text.startswith("Index Name"):
        return pd.DataFrame()
    try:
        raw = pd.read_csv(StringIO(response.text))
    except Exception as exc:  # noqa: BLE001
        logger.info("NSE index archive parse failed for %s: %s", trade_date, exc)
        return pd.DataFrame()
    required = {"Index Name", "Index Date", "Open Index Value", "High Index Value", "Low Index Value", "Closing Index Value"}
    if not required.issubset(raw.columns):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "index_name": raw["Index Name"].astype(str).str.strip(),
            "date": pd.to_datetime(raw["Index Date"], format="%d-%m-%Y", errors="coerce"),
            "open": pd.to_numeric(raw["Open Index Value"], errors="coerce"),
            "high": pd.to_numeric(raw["High Index Value"], errors="coerce"),
            "low": pd.to_numeric(raw["Low Index Value"], errors="coerce"),
            "close": pd.to_numeric(raw["Closing Index Value"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("Volume", pd.Series([0] * len(raw))), errors="coerce").fillna(0),
        }
    ).dropna(subset=["index_name", "date", "open", "high", "low", "close"])


def _resample_index_history(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    indexed = df.copy()
    indexed["date"] = pd.to_datetime(indexed["date"], errors="coerce")
    indexed = indexed.dropna(subset=["date"]).set_index("date").sort_index()
    sampled = indexed.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    out = sampled.reset_index()
    out.attrs.update(df.attrs)
    return out


def _period_days(period: str) -> int | None:
    value = str(period or "").strip().lower()
    if value in {"max", "all"}:
        return None
    units = [
        ("mo", 31),
        ("wk", 7),
        ("yr", 365),
        ("w", 7),
        ("y", 365),
        ("m", 31),
        ("d", 1),
    ]
    for unit, multiplier in units:
        if value.endswith(unit):
            raw = value[: -len(unit)].strip()
            try:
                return max(1, int(float(raw) * multiplier))
            except ValueError:
                return None
    return None

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from io import StringIO
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

STOOQ_URL = "https://stooq.com/q/d/l/"


def get_stooq_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch free Stooq OHLCV data when a free Stooq API key is configured."""
    api_key = os.getenv("STOOQ_API_KEY")
    if not api_key:
        return pd.DataFrame()
    symbol = _to_stooq_symbol(ticker)
    freq = _to_stooq_interval(interval)
    if symbol is None or freq is None:
        return pd.DataFrame()

    start = _period_start(period)
    params = {
        "s": symbol,
        "i": freq,
        "d1": start.strftime("%Y%m%d"),
        "d2": date.today().strftime("%Y%m%d"),
        "apikey": api_key,
    }
    url = f"{STOOQ_URL}?{urlencode(params)}"
    try:
        request = Request(url, headers={"User-Agent": "ai-stock-advisor-mcp/0.1"})
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
    except (OSError, URLError) as exc:
        logger.warning("Stooq price history fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if not payload.strip() or "No data" in payload:
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(payload))
    except Exception as exc:
        logger.warning("Stooq price history parse failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    out = df.rename(columns={c: str(c).lower().replace(" ", "_") for c in df.columns})
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(out.columns):
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out.attrs["provider"] = "stooq"
    return out


def _to_stooq_symbol(ticker: str) -> str | None:
    normalized = ticker.strip().lower()
    if not normalized or normalized.endswith((".ns", ".bo")):
        return None
    if "." in normalized:
        return normalized
    return f"{normalized}.us"


def _to_stooq_interval(interval: str) -> str | None:
    return {"1d": "d", "1wk": "w", "1mo": "m"}.get(interval)


def _period_start(period: str) -> date:
    today = date.today()
    value = period.strip().lower()
    if value == "max":
        return today - timedelta(days=365 * 30)
    if value == "ytd":
        return date(today.year, 1, 1)
    try:
        amount = int(value[:-2] if value.endswith("mo") else value[:-1])
    except ValueError:
        return today - timedelta(days=365)
    if value.endswith("mo"):
        return today - timedelta(days=31 * amount)
    if value.endswith("y"):
        return today - timedelta(days=365 * amount)
    if value.endswith("d"):
        return today - timedelta(days=amount)
    return today - timedelta(days=365)

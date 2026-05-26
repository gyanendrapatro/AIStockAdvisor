from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def get_yahoo_chart_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo's chart endpoint without yfinance cookie handling."""
    params = {
        "range": _to_yahoo_range(period),
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }
    url = f"{YAHOO_CHART_URL.format(ticker=quote(ticker, safe=''))}?{urlencode(params)}"
    try:
        request = Request(url, headers={"User-Agent": "ai-stock-advisor-mcp/0.1"})
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.info("Yahoo chart price history fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    try:
        result = payload["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        quote_data = result.get("indicators", {}).get("quote", [{}])[0]
    except (KeyError, IndexError, TypeError) as exc:
        logger.info("Yahoo chart response missing price history for %s: %s", ticker, exc)
        return pd.DataFrame()

    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "open": quote_data.get("open", []),
            "high": quote_data.get("high", []),
            "low": quote_data.get("low", []),
            "close": quote_data.get("close", []),
            "volume": quote_data.get("volume", []),
        }
    )
    rows = rows.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame()
    rows.attrs["provider"] = "yahoo_chart"
    return rows


def _to_yahoo_range(period: str) -> str:
    value = period.strip().lower()
    allowed = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    return value if value in allowed else "1y"

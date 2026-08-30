from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Personal, non-commercial, read-only use: derives shares_outstanding = market_cap / price
# from screener.in's free company pages, rate-limited, one ticker at a time. This is a one-time
# (or occasional) backfill only -- see scripts/backfill_shares_outstanding.py, the only caller.
# Never wired into any automatic refresh; shares outstanding barely changes (buybacks, new
# issuance, splits, bonus issues), so market cap itself is recomputed daily for free afterward
# from this figure and the already-cached bhavcopy/yfinance close price -- see
# market_data.recompute_market_cap_from_shares_outstanding.
SCREENER_BASE_URL = "https://www.screener.in/company/{slug}/"
SCREENER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}
DEFAULT_REQUEST_DELAY_SECONDS = 1.5


def screener_slug(ticker: str) -> str | None:
    """screener.in company-page slug for a ticker: the bare NSE symbol for .NS tickers, the
    numeric BSE scrip code for .BO tickers. None for anything else (US benchmarks, indices)."""
    value = str(ticker or "").strip().upper()
    if value.endswith(".NS"):
        return value[: -len(".NS")] or None
    if value.endswith(".BO"):
        return value[: -len(".BO")] or None
    return None


_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")


def _first_number(text: str) -> float | None:
    """Extract the first Indian-comma-grouped number (e.g. "17,45,629" or "93.3") from text.
    Deliberately narrower than stripping all non-digit characters: a trailing "Cr." has its own
    period (e.g. "93.3 Cr." has two periods total), which would otherwise get glued onto the
    decimal point and make the result unparseable."""
    match = _NUMBER_RE.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _parse_crore_rupees(text: str) -> float | None:
    """Parse a "₹ 17,45,629 Cr." style value into plain rupees. screener.in's equity Market
    Cap figure is always in Cr. (1 Cr. = 1e7)."""
    number = _first_number(text)
    return number * 1e7 if number is not None else None


def _parse_rupees(text: str) -> float | None:
    """Parse a "₹ 1,290" style value into plain rupees (no Cr. multiplier)."""
    return _first_number(text)


def fetch_screener_shares_outstanding(
    ticker: str, *, session: requests.Session | None = None, timeout: float = 20.0
) -> float | None:
    """Fetch one ticker's shares_outstanding = market_cap / current_price from screener.in.

    Returns None if the page can't be found, the ratios panel is missing, or the company isn't
    currently trading -- screener.in shows a blank Market Cap/Current Price for those (which
    independently corroborates this app's own bhavcopy-based dormancy detection: tickers already
    marked instrument_master.active=0 consistently show blank here too).
    """
    slug = screener_slug(ticker)
    if not slug:
        return None
    http = session or requests
    try:
        response = http.get(SCREENER_BASE_URL.format(slug=slug), headers=SCREENER_HEADERS, timeout=timeout)
    except requests.RequestException as exc:
        logger.info("Screener fetch failed for %s: %s", ticker, exc)
        return None
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    ratios = soup.find("ul", id="top-ratios")
    if not ratios:
        return None

    values: dict[str, str] = {}
    for item in ratios.find_all("li"):
        name_el = item.find("span", class_="name")
        value_el = item.find("span", class_="value")
        if name_el and value_el:
            values[name_el.get_text(strip=True)] = value_el.get_text(" ", strip=True)

    market_cap = _parse_crore_rupees(values.get("Market Cap", ""))
    current_price = _parse_rupees(values.get("Current Price", ""))
    if not market_cap or not current_price:
        return None
    return round(market_cap / current_price, 2)

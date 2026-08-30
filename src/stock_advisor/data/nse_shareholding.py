from __future__ import annotations

import logging
import re
from datetime import date

import requests

logger = logging.getLogger(__name__)

# Official, free, unauthenticated -- NSE's own bulk quarterly SEBI shareholding-pattern filing
# index, verified live this session. Each record links to the filing's XBRL, which carries the
# grand-total number of fully paid-up equity shares outstanding with a real as-of date. Sits on
# the same reliable archive infrastructure this app already trusts for bhavcopy, unlike the
# `quote-equity` API (already blocked, already the reason this app's classification data goes
# stale -- see universe.py).
NSE_HOME = "https://www.nseindia.com"
NSE_SHAREHOLDING_MASTER_API = "https://www.nseindia.com/api/corporate-share-holdings-master"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}

# The XBRL taxonomy's grand-total context ID changed between the 2022-09-30 and 2025-10-31
# revisions (verified this session against real filings for the same company years apart) --
# same tag, different context suffix. Tried in order; append future revisions here rather than
# replacing what's here, so older filings keep parsing.
_GRAND_TOTAL_CONTEXT_IDS = ("ShareholdingPattern_ContextI", "ShareholdingPatternI")
_TOTAL_SHARES_RE = re.compile(
    r'<in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="(?P<context>[^"]+)"[^>]*>(?P<value>[^<]+)<'
)


def nse_session() -> requests.Session:
    """A session warmed with NSE's cookies -- required by corporate-share-holdings-master (unlike
    the public archive/bhavcopy endpoints). Self-contained here rather than importing
    universe.py's private _nse_session(), matching this codebase's convention of each data-source
    module owning its own session/header setup (see exchange_eod.py)."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    session.get(NSE_HOME, timeout=15)
    return session


def fetch_shareholding_master(
    session: requests.Session,
    *,
    universe: str = "equities",
    from_date: str | None = None,
    to_date: str | None = None,
    timeout: float = 20.0,
) -> list[dict]:
    """One call, whole exchange: every company's latest (or date-ranged) quarterly shareholding
    filing. universe="equities" for mainboard, "sme" for the SME platform.

    For universe="sme", NSE's default (no date range) lookback is too narrow for a real backfill
    -- verified this session: it returns ~20 records vs. 960 with a several-month window. Callers
    doing a full SME sweep should always pass an explicit from_date/to_date.

    dates are DD-MM-YYYY strings, matching NSE's own format.
    """
    params: dict[str, str] = {"index": universe}
    if from_date:
        params["from_date"] = from_date
    if to_date:
        params["to_date"] = to_date
    try:
        response = session.get(NSE_SHAREHOLDING_MASTER_API, params=params, timeout=timeout)
    except requests.RequestException as exc:
        logger.info("NSE shareholding master fetch failed (index=%s): %s", universe, exc)
        return []
    if response.status_code != 200:
        logger.info("NSE shareholding master unavailable (index=%s): status=%s", universe, response.status_code)
        return []
    try:
        data = response.json()
    except ValueError:
        logger.info("NSE shareholding master returned non-JSON (index=%s)", universe)
        return []
    return data if isinstance(data, list) else []


def extract_total_paid_up_shares(xbrl_text: str) -> float | None:
    """Parse in-bse-shp:NumberOfFullyPaidUpEquityShares at the grand-total context out of a
    shareholding-pattern XBRL filing's raw text. Tries every known context-ID naming convention
    (see _GRAND_TOTAL_CONTEXT_IDS) so both current and multi-year-old filings parse."""
    matches = {m.group("context"): m.group("value") for m in _TOTAL_SHARES_RE.finditer(xbrl_text)}
    for context_id in _GRAND_TOTAL_CONTEXT_IDS:
        raw = matches.get(context_id)
        if raw is None:
            continue
        try:
            return float(raw.strip())
        except ValueError:
            continue
    return None


def build_shareholding_index(records: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """From a list of shareholding-master records (possibly spanning several quarters per
    company, if fetched with a wide date range), build (by_symbol, by_isin) lookups holding only
    each company's *latest* filing -- picking a stale one instead would be a real correctness
    bug, not just a missed optimization."""
    by_symbol: dict[str, dict] = {}
    by_isin: dict[str, dict] = {}

    def _keep_latest(index: dict[str, dict], key: str, record: dict) -> None:
        if not key:
            return
        existing = index.get(key)
        if existing is None or str(record.get("date") or "") > str(existing.get("date") or ""):
            index[key] = record

    for record in records:
        symbol = str(record.get("symbol") or "").strip().upper()
        isin = str(record.get("isin") or "").strip().upper()
        _keep_latest(by_symbol, symbol, record)
        _keep_latest(by_isin, isin, record)
    return by_symbol, by_isin


def default_sme_lookback_from_date(months_back: int = 9) -> str:
    """DD-MM-YYYY string ~months_back months before today, for the wide from_date the sme
    universe needs (see fetch_shareholding_master's docstring)."""
    today = date.today()
    year = today.year
    month = today.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, 1).strftime("%d-%m-%Y")

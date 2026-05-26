from __future__ import annotations

from functools import lru_cache
import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "ai-stock-advisor-mcp/0.1 local-research@example.com")

USD_UNITS = ("USD", "USD/shares", "shares", "pure")

FACT_TAGS = {
    "sec_revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "sec_net_income": ("NetIncomeLoss", "ProfitLoss"),
    "sec_assets": ("Assets",),
    "sec_liabilities": ("Liabilities", "LiabilitiesCurrent"),
    "sec_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "sec_cash": ("CashAndCashEquivalentsAtCarryingValue", "Cash"),
    "sec_operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "sec_eps_diluted": ("EarningsPerShareDiluted",),
}


def get_sec_fundamentals(ticker: str) -> dict[str, Any]:
    """Return official free SEC EDGAR facts for US-listed tickers."""
    if "." in ticker:
        return {}
    cik = get_cik_for_ticker(ticker)
    if cik is None:
        return {}
    facts = _get_company_facts(cik)
    if not facts:
        return {}

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    result: dict[str, Any] = {
        "_sources": ["sec_edgar"],
        "sec_cik": f"{cik:010d}",
        "sec_entity_name": facts.get("entityName"),
    }
    for key, tags in FACT_TAGS.items():
        fact = _latest_fact(us_gaap, tags)
        if fact:
            result[key] = fact.get("val")
            result[f"{key}_end"] = fact.get("end")
            result[f"{key}_form"] = fact.get("form")
            result[f"{key}_filed"] = fact.get("filed")
    return result


@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, int]:
    try:
        payload = _get_json(SEC_TICKERS_URL)
    except Exception as exc:
        logger.info("SEC ticker map fetch failed: %s", exc)
        return {}
    fields = payload.get("fields", [])
    ticker_index = fields.index("ticker")
    cik_index = fields.index("cik")
    return {str(row[ticker_index]).upper(): int(row[cik_index]) for row in payload.get("data", [])}


def get_cik_for_ticker(ticker: str) -> int | None:
    return _ticker_map().get(ticker.strip().upper())


@lru_cache(maxsize=512)
def _get_company_facts(cik: int) -> dict[str, Any]:
    try:
        return _get_json(SEC_COMPANY_FACTS_URL.format(cik=cik))
    except Exception as exc:
        logger.info("SEC company facts fetch failed for CIK %s: %s", cik, exc)
        return {}


def _get_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(str(exc)) from exc


def _latest_fact(us_gaap: dict[str, Any], tags: tuple[str, ...]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for tag in tags:
        units = us_gaap.get(tag, {}).get("units", {})
        for unit in USD_UNITS:
            candidates.extend(units.get(unit, []))
        if candidates:
            break
    if not candidates:
        return None
    candidates = [item for item in candidates if item.get("val") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (str(item.get("filed", "")), str(item.get("end", ""))))

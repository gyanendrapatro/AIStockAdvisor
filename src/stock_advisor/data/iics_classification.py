from __future__ import annotations

from typing import Any
import io
import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# One-time, personal/non-commercial backfill (see scripts/build_iics_classification.py, the only
# caller) -- same framing as screener.py's shares-outstanding scraper. Not wired into any
# automatic refresh: run manually, produces data/iics_classification.csv, which
# market_data.load_iics_classification_seed replays into a fresh DB with zero network calls.
#
# The official India Industry Classification Structure (IICS) -- the common taxonomy NSE and BSE
# jointly adopted in 2022 -- has 4 levels: Macro-Economic Sector -> Sector -> Industry -> Basic
# Industry (12 -> 22 -> 59 -> 197 nodes). BSE publishes the full code tree as a flat, parseable
# PDF table; screener.in's own company pages resolve every listed company (NSE and BSE alike) to
# its IICS basic-industry leaf, and that leaf's own page is a *listing* page -- fetching it
# returns every company in that basic industry, not just one. So only ~197 requests (one per
# basic-industry leaf, taken straight from the PDF's code list) are needed for the whole
# NSE+BSE universe, instead of one request per company.
BSE_IICS_TAXONOMY_PDF_URL = "https://www.bseindia.com/Downloads1/India_Industry_Classification_Structure.pdf"
SCREENER_MARKET_URL = "https://www.screener.in/market/{macro_code}/{sector_code}/{industry_code}/{basic_industry_code}/"
IICS_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}
DEFAULT_REQUEST_DELAY_SECONDS = 1.5

# Matches one IICS row's 4 nested codes in the PDF's extracted text: IN01 (macro, 2 digits) ->
# IN0101 (sector, 4 digits) -> IN010101 (industry, 6 digits) -> IN010101001 (basic industry, 9
# digits). \D*? (non-digit, non-greedy) skips over the name/definition text between codes without
# needing to correctly segment where a name ends and a definition begins -- that text wraps
# unpredictably across PDF lines, but the codes themselves never do. Verified live: 197 matches,
# 197 distinct basic-industry codes, correctly nested.
_IICS_CODE_ROW_RE = re.compile(r"(IN\d{2})\D*?(IN\d{4})\D*?(IN\d{6})\D*?(IN\d{9})")

_COMPANY_LINK_RE = re.compile(r"/company/([A-Za-z0-9]+)/")
_PAGE_INFO_RE = re.compile(r"Showing page (\d+) of (\d+)")


def fetch_iics_taxonomy_pdf(*, timeout: float = 30.0) -> bytes:
    """Download BSE's official IICS code-tree PDF. Free, no auth."""
    response = requests.get(BSE_IICS_TAXONOMY_PDF_URL, headers=IICS_REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_iics_taxonomy_pdf(pdf_bytes: bytes) -> list[dict[str, str]]:
    """Extract the ~197 basic-industry leaf code-paths from BSE's IICS PDF.

    Only the codes are extracted here (not names/definitions) -- deliberately: the PDF's text
    layout wraps names/definitions across lines unpredictably, making that text hard to segment
    reliably, but every basic-industry leaf's screener.in page already shows a clean breadcrumb
    with all 4 levels' names (see fetch_iics_basic_industry_companies) -- no need to parse it
    twice from two different, differently-messy sources.
    """
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    full_text = " ".join(page.extract_text() or "" for page in reader.pages)
    full_text = re.sub(r"\s+", " ", full_text)

    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for macro_code, sector_code, industry_code, basic_industry_code in _IICS_CODE_ROW_RE.findall(full_text):
        if basic_industry_code in seen:
            continue
        seen.add(basic_industry_code)
        rows.append(
            {
                "macro_code": macro_code,
                "sector_code": sector_code,
                "industry_code": industry_code,
                "basic_industry_code": basic_industry_code,
            }
        )
    return rows


def fetch_iics_basic_industry_companies(
    macro_code: str,
    sector_code: str,
    industry_code: str,
    basic_industry_code: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Fetch one basic-industry leaf's screener.in listing page.

    Returns {"names": {"macro_sector": ..., "sector": ..., "industry": ..., "basic_industry": ...},
    "slugs": [...], "page": 1, "total_pages": 1}. slugs are screener.in company-page slugs (bare
    NSE symbol, or numeric BSE scrip code for a BSE-only listing) -- same convention as
    screener.screener_slug, just discovered in bulk here instead of looked up one ticker at a
    time. Only page 1 is ever fetched -- screener.in's robots.txt disallows crawling `?page=`, so
    a basic industry with more companies than fit on one page is only partially covered; total_pages
    surfaces that so the caller can log it rather than silently under-counting.
    """
    url = SCREENER_MARKET_URL.format(
        macro_code=macro_code, sector_code=sector_code, industry_code=industry_code, basic_industry_code=basic_industry_code
    )
    http = session or requests
    response = http.get(url, headers=IICS_REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    breadcrumb_items = soup.select(".breadcrumb li")
    # Item 0 is the static "Industries" root; items 1-3 are macro/sector/industry (each an <a>
    # whose text is the level's name); the last item (basic industry) is the current page, so
    # screener.in renders it as plain text, not a link.
    names_list = [item.get_text(strip=True) for item in breadcrumb_items[1:]]
    names = {
        "macro_sector": names_list[0] if len(names_list) > 0 else None,
        "sector": names_list[1] if len(names_list) > 1 else None,
        "industry": names_list[2] if len(names_list) > 2 else None,
        "basic_industry": names_list[3] if len(names_list) > 3 else None,
    }

    slugs = sorted(dict.fromkeys(_COMPANY_LINK_RE.findall(response.text)))

    page, total_pages = 1, 1
    page_match = _PAGE_INFO_RE.search(response.text)
    if page_match:
        page, total_pages = int(page_match.group(1)), int(page_match.group(2))

    return {"names": names, "slugs": slugs, "page": page, "total_pages": total_pages}


def resolve_iics_slug_to_ticker(slug: str) -> str | None:
    """Inverse of screener.screener_slug: a purely numeric slug is a BSE scrip code (-> .BO), an
    alphabetic slug is an NSE symbol (-> .NS)."""
    value = str(slug or "").strip().upper()
    if not value:
        return None
    return f"{value}.BO" if value.isdigit() else f"{value}.NS"


def build_iics_classification_mapping(
    taxonomy_rows: list[dict[str, str]],
    *,
    delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Walk every basic-industry leaf and build the full ticker -> IICS classification mapping.

    Returns {"rows": [...], "basic_industries_walked": int, "basic_industries_failed": int,
    "incomplete_basic_industries": [...], "tickers_resolved": int}. rows are shaped for
    market_data.store_security_classification (ticker + all 4 levels' codes/names).
    """
    import time

    session = requests.Session()
    session.headers.update(IICS_REQUEST_HEADERS)

    rows: list[dict[str, Any]] = []
    failed = 0
    incomplete: list[str] = []
    total = len(taxonomy_rows)

    for index, taxonomy_row in enumerate(taxonomy_rows):
        macro_code = taxonomy_row["macro_code"]
        sector_code = taxonomy_row["sector_code"]
        industry_code = taxonomy_row["industry_code"]
        basic_industry_code = taxonomy_row["basic_industry_code"]
        try:
            result = fetch_iics_basic_industry_companies(macro_code, sector_code, industry_code, basic_industry_code, session=session)
        except requests.RequestException as exc:
            logger.info("IICS basic-industry fetch failed for %s: %s", basic_industry_code, exc)
            failed += 1
            result = None

        if result:
            if result["total_pages"] > 1:
                incomplete.append(basic_industry_code)
            names = result["names"]
            for slug in result["slugs"]:
                ticker = resolve_iics_slug_to_ticker(slug)
                if not ticker:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "macro_sector_code": macro_code,
                        "macro_sector": names.get("macro_sector"),
                        "sector_code": sector_code,
                        "sector": names.get("sector"),
                        "industry_code": industry_code,
                        "industry": names.get("industry"),
                        "basic_industry_code": basic_industry_code,
                        "basic_industry": names.get("basic_industry"),
                    }
                )

        if progress_callback:
            progress_callback({"completed": index + 1, "total": total, "basic_industry_code": basic_industry_code})
        if delay_seconds and index < total - 1:
            time.sleep(max(0.0, float(delay_seconds)))

    # A company can legitimately appear once (one basic industry each); dedupe defensively by
    # ticker in case a slug ever showed up under two leaves.
    deduped = list({row["ticker"]: row for row in rows}.values())
    return {
        "rows": deduped,
        "basic_industries_walked": total - failed,
        "basic_industries_failed": failed,
        "incomplete_basic_industries": incomplete,
        "tickers_resolved": len(deduped),
    }

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import logging
import os
from typing import Any

import requests

from stock_advisor.analysis.pipeline import sanitize_for_json
from stock_advisor.data.universe import load_stock_universe

logger = logging.getLogger(__name__)

NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements"
BSE_ANNOUNCEMENTS_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
BSE_ATTACHMENT_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("EXCHANGE_ANNOUNCEMENT_TIMEOUT_SECONDS", "12"))
PDF_PARSE_ENABLED = os.getenv("EXCHANGE_PDF_PARSE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}

MATERIAL_KEYWORDS = {
    "quarterly_results": ("financial results", "results", "quarter", "audited", "unaudited", "earnings"),
    "order_book": ("order", "contract", "loa", "letter of award", "purchase order", "work order"),
    "expansion_capex": ("expansion", "capex", "capacity", "plant", "facility", "greenfield", "brownfield"),
    "innovation_new_area": ("innovation", "product launch", "new product", "technology", "ai", "data center", "semiconductor"),
    "fundraising_debt": ("fund raising", "fundraising", "qip", "preferential", "debt", "loan", "ncd"),
    "legal_regulatory": ("sebi", "court", "tribunal", "litigation", "penalty", "show cause", "regulatory"),
    "corporate_action": ("dividend", "split", "bonus", "buyback", "record date"),
    "management_governance": ("resignation", "appointment", "director", "auditor", "promoter", "pledge"),
}


def get_exchange_announcements(
    ticker: str,
    *,
    limit: int = 12,
    days: int = 180,
    parse_pdfs: bool = True,
    max_pdf_documents: int = 2,
    max_pdf_chars: int = 3500,
) -> dict[str, Any]:
    """Return latest official NSE/BSE announcements and optional PDF snippets.

    NSE data is queried by symbol. BSE data is queried when a BSE security code is
    discoverable from the local NSE+BSE master. PDF parsing is best-effort and
    works when the optional ``pypdf`` package is installed.
    """
    normalized = _normalize_ticker(ticker)
    symbol = _symbol(normalized)
    identifiers = _exchange_identifiers(normalized)
    warnings: list[str] = []

    rows: list[dict[str, Any]] = []
    nse_rows, nse_warnings = _get_nse_announcements(symbol=symbol, limit=limit)
    rows.extend(nse_rows)
    warnings.extend(nse_warnings)

    bse_security_id = identifiers.get("bse_security_id")
    if bse_security_id:
        bse_rows, bse_warnings = _get_bse_announcements(str(bse_security_id), limit=limit, days=days)
        rows.extend(bse_rows)
        warnings.extend(bse_warnings)
    else:
        warnings.append("BSE announcement lookup skipped because no BSE security id was found in the local master.")

    rows = _dedupe_announcements(rows)
    rows = sorted(rows, key=lambda row: str(row.get("published_at") or ""), reverse=True)[: max(1, int(limit))]
    for row in rows:
        row["material_categories"] = _classify_announcement(row)
        row["materiality_score"] = _materiality_score(row)

    parsed_count = 0
    if parse_pdfs and PDF_PARSE_ENABLED:
        for row in sorted(rows, key=lambda item: item.get("materiality_score", 0), reverse=True):
            if parsed_count >= max(0, int(max_pdf_documents)):
                break
            attachment_url = row.get("attachment_url")
            if not attachment_url or not str(attachment_url).lower().endswith(".pdf"):
                continue
            text, parse_warning = _extract_pdf_text(str(attachment_url), max_chars=max_pdf_chars)
            if parse_warning:
                warnings.append(parse_warning)
                continue
            if text:
                row["attachment_text_excerpt"] = text
                parsed_count += 1
    elif parse_pdfs and not PDF_PARSE_ENABLED:
        warnings.append("Exchange PDF parsing is disabled by EXCHANGE_PDF_PARSE_ENABLED=0.")

    category_counts: dict[str, int] = {}
    for row in rows:
        for category in row.get("material_categories", []):
            category_counts[category] = category_counts.get(category, 0) + 1

    return sanitize_for_json(
        {
            "ticker": normalized,
            "symbol": symbol,
            "identifiers": identifiers,
            "announcements": rows,
            "category_counts": dict(sorted(category_counts.items())),
            "parsed_pdf_count": parsed_count,
            "providers": sorted({row.get("exchange") for row in rows if row.get("exchange")}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": list(dict.fromkeys(warnings)),
        }
    )


def _get_nse_announcements(*, symbol: str, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    if not symbol:
        return [], ["NSE announcement lookup skipped because no symbol was available."]
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
        }
    )
    warnings: list[str] = []
    try:
        session.get("https://www.nseindia.com/", timeout=REQUEST_TIMEOUT_SECONDS)
        response = session.get(
            NSE_ANNOUNCEMENTS_URL,
            params={"index": "equities", "symbol": symbol},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.info("NSE announcements fetch failed for %s: %s", symbol, exc)
        return [], [f"NSE announcements fetch failed for {symbol}: {exc}"]

    if isinstance(payload, dict):
        raw_rows = payload.get("data") or payload.get("rows") or []
    else:
        raw_rows = payload if isinstance(payload, list) else []

    rows: list[dict[str, Any]] = []
    for item in raw_rows[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        headline = _first_text(item, "desc", "subject", "announcement", "sm_name")
        rows.append(
            {
                "exchange": "NSE",
                "symbol": _first_text(item, "symbol") or symbol,
                "company_name": _first_text(item, "sm_name", "companyName"),
                "headline": headline,
                "summary": _first_text(item, "details", "remarks"),
                "published_at": _first_text(item, "dt", "an_dt", "sort_date"),
                "category": _first_text(item, "desc", "attchmntText"),
                "attachment_url": _first_text(item, "attchmntFile"),
                "source_url": _first_text(item, "attchmntFile"),
            }
        )
    return [row for row in rows if row.get("headline") or row.get("attachment_url")], warnings


def _get_bse_announcements(security_id: str, *, limit: int, days: int) -> tuple[list[dict[str, Any]], list[str]]:
    end = datetime.now().date()
    start = end - timedelta(days=max(1, int(days)))
    params = {
        "pageno": 1,
        "strCat": "-1",
        "strPrevDate": start.strftime("%Y%m%d"),
        "strScrip": security_id,
        "strSearch": "P",
        "strToDate": end.strftime("%Y%m%d"),
        "strType": "C",
        "subcategory": "-1",
        "strIndustry": "-1",
        "segment": "Equity",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.bseindia.com/corporates/ann.html",
    }
    try:
        response = requests.get(BSE_ANNOUNCEMENTS_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.info("BSE announcements fetch failed for %s: %s", security_id, exc)
        return [], [f"BSE announcements fetch failed for {security_id}: {exc}"]

    raw_rows = payload.get("Table") if isinstance(payload, dict) else []
    if not isinstance(raw_rows, list):
        return [], []

    rows: list[dict[str, Any]] = []
    for item in raw_rows[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        attachment = _first_text(item, "ATTACHMENTNAME", "NSURL")
        rows.append(
            {
                "exchange": "BSE",
                "symbol": _first_text(item, "SCRIP_CD") or security_id,
                "company_name": _first_text(item, "SLONGNAME", "COMPANYNAME"),
                "headline": _first_text(item, "HEADLINE", "MORE"),
                "summary": _first_text(item, "MORE"),
                "published_at": _first_text(item, "NEWS_DT", "DT_TM"),
                "category": _first_text(item, "CATEGORYNAME", "SUBCATNAME"),
                "attachment_url": _bse_attachment_url(attachment),
                "source_url": _bse_attachment_url(attachment),
            }
        )
    return [row for row in rows if row.get("headline") or row.get("attachment_url")], []


def _exchange_identifiers(ticker: str) -> dict[str, Any]:
    try:
        universe = load_stock_universe(universe="all_india")
    except Exception as exc:  # noqa: BLE001
        logger.info("All India universe load failed for exchange identifiers: %s", exc)
        universe = None
    if universe is None or universe.empty:
        return {}

    symbol = _symbol(ticker)
    candidates = universe[
        (universe.get("ticker").astype(str).str.upper() == ticker)
        | (universe.get("nse_ticker").astype(str).str.upper() == ticker)
        | (universe.get("bse_ticker").astype(str).str.upper() == ticker)
        | (universe.get("symbol").astype(str).str.upper() == symbol)
    ].copy()
    if candidates.empty:
        return {}
    row = candidates.iloc[0].to_dict()
    return {
        "isin": _clean_identifier(row.get("isin")),
        "nse_ticker": _clean_identifier(row.get("nse_ticker")),
        "bse_ticker": _clean_identifier(row.get("bse_ticker")),
        "nse_security_id": _clean_identifier(row.get("nse_security_id")),
        "bse_security_id": _clean_identifier(row.get("bse_security_id") or row.get("security_id")),
        "company_name": _clean_identifier(row.get("name")),
    }


def _extract_pdf_text(url: str, *, max_chars: int) -> tuple[str | None, str | None]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None, "PDF text extraction skipped because optional package pypdf is not installed."

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        reader = PdfReader(BytesIO(response.content))
        chunks = []
        for page in reader.pages[:6]:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text.strip())
            if sum(len(chunk) for chunk in chunks) >= max_chars:
                break
    except Exception as exc:  # noqa: BLE001
        logger.info("PDF extraction failed for %s: %s", url, exc)
        return None, f"PDF text extraction failed for {url}: {exc}"

    text = "\n".join(chunks).strip()
    if not text:
        return None, None
    return text[: max(0, int(max_chars))], None


def _classify_announcement(row: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("headline", "summary", "category", "attachment_text_excerpt")
    ).lower()
    categories = [
        category
        for category, keywords in MATERIAL_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    return categories or ["general"]


def _materiality_score(row: dict[str, Any]) -> int:
    weights = {
        "quarterly_results": 30,
        "order_book": 28,
        "expansion_capex": 24,
        "innovation_new_area": 22,
        "legal_regulatory": 22,
        "fundraising_debt": 18,
        "corporate_action": 16,
        "management_governance": 14,
        "general": 5,
    }
    score = sum(weights.get(category, 0) for category in row.get("material_categories", []))
    if row.get("attachment_text_excerpt"):
        score += 10
    if row.get("attachment_url"):
        score += 3
    return score


def _dedupe_announcements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for row in rows:
        key = (
            str(row.get("exchange") or ""),
            str(row.get("published_at") or ""),
            str(row.get("headline") or row.get("attachment_url") or "").strip().lower(),
        )
        if not key[2] or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _first_text(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None or str(value).strip() == "":
            continue
        return str(value).strip()
    return None


def _bse_attachment_url(attachment: str | None) -> str | None:
    if not attachment:
        return None
    if str(attachment).startswith("http"):
        return str(attachment)
    return BSE_ATTACHMENT_URL.format(attachment=str(attachment).strip())


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker or "").strip().upper()
    if value and "." not in value:
        value = f"{value}.NS"
    return value


def _symbol(ticker: str) -> str:
    return ticker.removesuffix(".NS").removesuffix(".BO").strip().upper()


def _clean_identifier(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text

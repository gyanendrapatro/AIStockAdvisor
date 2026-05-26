from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import yfinance as yf

logger = logging.getLogger(__name__)

POSITIVE_TERMS = {
    "beat",
    "beats",
    "bullish",
    "buy",
    "growth",
    "higher",
    "outperform",
    "profit",
    "profits",
    "raise",
    "raises",
    "record",
    "strong",
    "surge",
    "upgrade",
    "upside",
}

NEGATIVE_TERMS = {
    "bearish",
    "cut",
    "cuts",
    "decline",
    "downgrade",
    "falls",
    "loss",
    "losses",
    "miss",
    "misses",
    "probe",
    "risk",
    "sell",
    "slump",
    "weak",
    "warning",
}


def get_news(ticker: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return free Yahoo Finance and GDELT headlines with local sentiment estimates."""
    if limit <= 0:
        return []
    yahoo_rows = _get_yahoo_news(ticker, limit)
    gdelt_rows = _get_gdelt_news(ticker, max(2, limit // 2))
    return _merge_news(yahoo_rows, gdelt_rows, limit)


def search_gdelt_articles(query: str, *, limit: int = 10, timespan: str = "30d") -> list[dict[str, Any]]:
    """Search GDELT's free news index and return locally classified article rows."""
    return _fetch_gdelt_articles(query, limit=limit, timespan=timespan)


def search_google_news(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    """Search Google News RSS for recent public web news without an API key."""
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en",
    }
    url = f"https://news.google.com/rss/search?{urlencode(params)}"
    try:
        request = Request(url, headers={"User-Agent": "ai-stock-advisor-mcp/0.1"})
        with urlopen(request, timeout=8) as response:
            payload = response.read()
        root = ET.fromstring(payload)
    except (OSError, HTTPError, URLError, TimeoutError, ET.ParseError) as exc:
        logger.info("Google News RSS fetch failed for query %r: %s", query, exc)
        return []

    rows = []
    for item in root.findall("./channel/item")[:limit]:
        title = _element_text(item, "title")
        summary = _strip_html(_element_text(item, "description"))
        score = _estimate_sentiment_score(" ".join(filter(None, [title, summary])))
        source = item.find("source")
        rows.append(
            _with_recency(
                {
                    "title": title,
                    "summary": summary,
                    "url": _element_text(item, "link"),
                    "time_published": _element_text(item, "pubDate"),
                    "overall_sentiment_score": score,
                    "overall_sentiment_label": _sentiment_label(score),
                    "provider": "google_news",
                    "source": source.text if source is not None else None,
                }
            )
        )
    return [row for row in rows if row.get("title") or row.get("url")]


def _get_yahoo_news(ticker: str, limit: int) -> list[dict[str, Any]]:
    try:
        articles = yf.Ticker(ticker).get_news(count=limit, tab="news")
    except Exception as exc:
        logger.info("Yahoo Finance news fetch failed for %s: %s", ticker, exc)
        return []

    rows: list[dict[str, Any]] = []
    for item in articles[:limit]:
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        title = item.get("title") or content.get("title")
        summary = item.get("summary") or content.get("summary") or content.get("description")
        score = _estimate_sentiment_score(" ".join(filter(None, [title, summary])))
        rows.append(
            _with_recency(
                {
                    "title": title,
                    "summary": summary,
                    "url": _extract_url(item, content),
                    "time_published": _extract_time(item, content),
                    "overall_sentiment_score": score,
                    "overall_sentiment_label": _sentiment_label(score),
                    "provider": "yahoo_finance",
                    "source": _extract_source(item, content),
                }
            )
        )
    return [row for row in rows if row.get("title") or row.get("summary")]


def _get_gdelt_news(ticker: str, limit: int) -> list[dict[str, Any]]:
    query = _gdelt_query(ticker)
    return _fetch_gdelt_articles(query, limit=limit, timespan="1week")


def _fetch_gdelt_articles(query: str, *, limit: int, timespan: str) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sort": "datedesc",
        "timespan": timespan,
        "maxrecords": max(1, min(limit, 25)),
    }
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?{urlencode(params)}"
    try:
        request = Request(url, headers={"User-Agent": "ai-stock-advisor-mcp/0.1"})
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.info("GDELT news fetch failed for query %r: %s", query, exc)
        return []

    rows = []
    for item in payload.get("articles", [])[:limit]:
        title = item.get("title")
        summary = item.get("seendate")
        score = _estimate_sentiment_score(title or "")
        rows.append(
            _with_recency(
                {
                    "title": title,
                    "summary": summary,
                    "url": item.get("url"),
                    "time_published": item.get("seendate"),
                    "overall_sentiment_score": score,
                    "overall_sentiment_label": _sentiment_label(score),
                    "provider": "gdelt",
                    "source": item.get("domain"),
                    "source_country": item.get("sourceCountry"),
                    "language": item.get("language"),
                }
            )
        )
    return [row for row in rows if row.get("title") or row.get("url")]


def _merge_news(primary: list[dict[str, Any]], secondary: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_len = max(len(primary), len(secondary))
    for idx in range(max_len):
        for rows in (primary, secondary):
            if idx >= len(rows):
                continue
            row = rows[idx]
            key = (row.get("url") or row.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(row)
            if len(merged) >= limit:
                return merged
    return merged


def _gdelt_query(ticker: str) -> str:
    base = ticker.split(".", 1)[0].strip().upper()
    if not base:
        return "stock market"
    return f'("{base}" OR "{base} stock" OR "{base} shares")'


def _extract_url(item: dict[str, Any], content: dict[str, Any]) -> str | None:
    for value in (
        item.get("url"),
        item.get("link"),
        _nested_url(content.get("canonicalUrl")),
        _nested_url(content.get("clickThroughUrl")),
    ):
        if value:
            return str(value)
    return None


def _extract_time(item: dict[str, Any], content: dict[str, Any]) -> str | None:
    value = item.get("providerPublishTime") or item.get("pubDate") or content.get("pubDate") or content.get("displayTime")
    if value is None:
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return str(value)


def _extract_source(item: dict[str, Any], content: dict[str, Any]) -> str | None:
    provider = item.get("publisher") or item.get("provider") or content.get("provider")
    if isinstance(provider, dict):
        return provider.get("displayName") or provider.get("name")
    if provider:
        return str(provider)
    return None


def _nested_url(value: Any) -> str | None:
    if isinstance(value, dict):
        url = value.get("url")
        return str(url) if url else None
    return str(value) if value else None


def _estimate_sentiment_score(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0.0
    positive = sum(1 for word in words if word in POSITIVE_TERMS)
    negative = sum(1 for word in words if word in NEGATIVE_TERMS)
    total = positive + negative
    if total == 0:
        return 0.0
    return round(max(-0.35, min(0.35, (positive - negative) / max(total, 3))), 3)


def _sentiment_label(score: float) -> str:
    if score >= 0.1:
        return "positive"
    if score <= -0.1:
        return "negative"
    return "neutral"


def _with_recency(row: dict[str, Any]) -> dict[str, Any]:
    published = _parse_time(row.get("time_published"))
    if not published:
        row["recency_bucket"] = "unknown"
        return row
    now = datetime.now(timezone.utc)
    days_old = max(0.0, (now - published).total_seconds() / 86400)
    row["published_at"] = published.isoformat()
    row["days_old"] = round(days_old, 2)
    if days_old <= 2:
        row["recency_bucket"] = "last_2_days"
    elif days_old <= 7:
        row["recency_bucket"] = "last_7_days"
    elif days_old <= 30:
        row["recency_bucket"] = "last_30_days"
    else:
        row["recency_bucket"] = "stale"
    return row


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        parsed = parsedate_to_datetime(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError, OverflowError):
        pass
    try:
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _element_text(parent: ET.Element, name: str) -> str | None:
    element = parent.find(name)
    return element.text if element is not None else None


def _strip_html(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()

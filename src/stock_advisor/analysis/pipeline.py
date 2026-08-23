from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import math
import os
from typing import Any

import pandas as pd

from stock_advisor.analysis.chart_patterns import chart_pattern_indicator_fields, detect_chart_patterns
from stock_advisor.analysis.scoring import score_stock, to_dict
from stock_advisor.analysis.indicators import latest_indicators
from stock_advisor.config.settings import load_watchlists, settings
from stock_advisor.data.analyst_events import get_analyst_insights, get_stock_events
from stock_advisor.data.company_intelligence import get_company_intelligence
from stock_advisor.data.market_data import CACHE_REFRESH_HINT, get_basic_fundamentals, get_price_history
from stock_advisor.data.news import get_news

# Analyzing a ticker is I/O-bound (price/fundamentals/news are all network fetches), so a
# multi-ticker scan or comparison fans them out across a small thread pool instead of
# looping serially. Rate-limit-friendly default; override via env if needed.
SCAN_MAX_WORKERS = int(os.getenv("STOCK_SCAN_MAX_WORKERS", "8"))


def analyze_stock(
    ticker: str,
    period: str | None = None,
    interval: str | None = None,
    *,
    include_news: bool = True,
    include_intelligence: bool = False,
    include_analyst_events: bool = False,
    intelligence_days: int = 30,
    intelligence_strategic_days: int = 365,
) -> dict[str, Any]:
    """Analyze one stock and return JSON-safe scoring output."""
    normalized_ticker = normalize_ticker(ticker)
    effective_period = period or settings.default_period
    effective_interval = interval or settings.default_interval

    warnings: list[str] = []
    prices = get_price_history(normalized_ticker, effective_period, effective_interval)
    if prices.empty:
        warnings.append(f"No price history is cached for {normalized_ticker}. {CACHE_REFRESH_HINT}")

    indicators = latest_indicators(prices)
    chart_patterns = detect_chart_patterns(prices)
    indicators.update(chart_pattern_indicator_fields(chart_patterns))
    fundamentals = get_basic_fundamentals(normalized_ticker)
    has_fundamentals = _has_fundamental_data(fundamentals)
    if not has_fundamentals:
        warnings.append("No fundamental data returned by market data provider.")

    news = get_news(normalized_ticker) if include_news else []
    if include_news and not news:
        warnings.append("No news sentiment data returned or news provider is not configured.")

    company_intelligence = (
        get_company_intelligence(
            normalized_ticker,
            fundamentals=fundamentals,
            indicators=indicators,
            seed_articles=news,
            days=intelligence_days,
            strategic_days=intelligence_strategic_days,
        )
        if include_intelligence
        else {}
    )
    if include_intelligence:
        warnings.extend(company_intelligence.get("warnings", [])[:2])

    analyst_insights = get_analyst_insights(normalized_ticker) if include_analyst_events else {}
    stock_events = get_stock_events(normalized_ticker) if include_analyst_events else {}
    if include_analyst_events:
        warnings.extend(analyst_insights.get("warnings", [])[:1])
        warnings.extend(stock_events.get("warnings", [])[:1])

    score = score_stock(
        normalized_ticker,
        indicators,
        fundamentals,
        news,
        intelligence=company_intelligence if include_intelligence else None,
    )
    result = to_dict(score)
    result["fundamentals"] = fundamentals
    result["latest_indicators"] = indicators
    result["chart_patterns"] = chart_patterns
    result["news"] = news
    result["company_intelligence"] = company_intelligence
    result["analyst_insights"] = analyst_insights
    result["stock_events"] = stock_events
    result["metadata"] = {
        "ticker": normalized_ticker,
        "period": effective_period,
        "interval": effective_interval,
        "data_points": int(len(prices)),
        "price_provider": prices.attrs.get("provider") if not prices.empty else None,
        "fundamental_providers": fundamentals.get("_sources", []),
        "news_providers": sorted({str(item.get("provider")) for item in news if item.get("provider")}),
        "intelligence_providers": company_intelligence.get("providers", []),
        "analyst_event_providers": sorted(
            {
                str(provider)
                for provider in [
                    *analyst_insights.get("providers", []),
                    *stock_events.get("providers", []),
                ]
                if provider
            }
        ),
        "has_price_history": not prices.empty,
        "has_fundamentals": has_fundamentals,
        "has_news": bool(news),
        "has_chart_patterns": bool(chart_patterns.get("patterns")),
        "has_company_intelligence": bool(company_intelligence.get("evidence")),
        "has_analyst_insights": bool(analyst_insights.get("providers")),
        "has_stock_events": bool(stock_events.get("providers")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
    }
    return sanitize_for_json(result)


def research_stock(
    ticker: str,
    period: str | None = None,
    interval: str | None = None,
    *,
    intelligence_days: int = 30,
    intelligence_strategic_days: int = 365,
) -> dict[str, Any]:
    """Run the deeper company-intelligence analysis for one ticker."""
    return analyze_stock(
        ticker,
        period=period,
        interval=interval,
        include_news=True,
        include_intelligence=True,
        include_analyst_events=True,
        intelligence_days=intelligence_days,
        intelligence_strategic_days=intelligence_strategic_days,
    )


def _analyze_stock_safe(ticker: str, **kwargs: Any) -> dict[str, Any]:
    """analyze_stock, but never raises — one bad ticker shouldn't sink a batch scan."""
    try:
        return analyze_stock(ticker, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "ticker": ticker,
            "error": str(exc),
            "final_score": 0,
            "signal": "Error",
            "confidence": 0,
            "reasons": [],
            "risks": ["Analysis failed before scoring."],
            "metadata": {
                "warnings": [str(exc)],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }


def _analyze_stocks_concurrently(tickers: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    """Fan out analyze_stock across a thread pool (I/O-bound: price/fundamentals/news fetches)."""
    if not tickers:
        return []
    if len(tickers) == 1:
        return [_analyze_stock_safe(tickers[0], **kwargs)]
    with ThreadPoolExecutor(max_workers=min(SCAN_MAX_WORKERS, len(tickers))) as executor:
        futures = [executor.submit(_analyze_stock_safe, ticker, **kwargs) for ticker in tickers]
        return [future.result() for future in as_completed(futures)]


def rank_watchlist(
    group: str | None = None,
    *,
    limit: int | None = None,
    period: str | None = None,
    interval: str | None = None,
    include_news: bool = True,
    include_intelligence: bool = False,
) -> list[dict[str, Any]]:
    """Analyze and rank all tickers in a watchlist group."""
    tickers = get_watchlist_tickers(group)
    results = _analyze_stocks_concurrently(
        tickers,
        period=period,
        interval=interval,
        include_news=include_news,
        include_intelligence=include_intelligence,
        include_analyst_events=include_intelligence,
    )
    ranked = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)
    if limit is not None:
        return ranked[: max(0, limit)]
    return ranked


def compare_stocks(
    tickers: Iterable[str],
    *,
    period: str | None = None,
    interval: str | None = None,
    include_news: bool = True,
    include_intelligence: bool = False,
) -> dict[str, Any]:
    """Analyze an explicit ticker list and return ranked comparison output."""
    unique_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers))
    rows = _analyze_stocks_concurrently(
        unique_tickers,
        period=period,
        interval=interval,
        include_news=include_news,
        include_intelligence=include_intelligence,
        include_analyst_events=include_intelligence,
    )
    ranked = sorted(rows, key=lambda x: x.get("final_score", 0), reverse=True)
    return {
        "count": len(ranked),
        "best": ranked[0] if ranked else None,
        "rows": ranked,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def get_watchlist_tickers(group: str | None = None) -> list[str]:
    watchlists = load_watchlists()
    if group:
        normalized_group = group.strip().lower()
        if normalized_group not in watchlists:
            raise ValueError(f"Unknown watchlist group: {group}. Expected one of: {', '.join(sorted(watchlists))}")
        return [normalize_ticker(ticker) for ticker in watchlists.get(normalized_group, [])]

    tickers: list[str] = []
    for values in watchlists.values():
        tickers.extend(normalize_ticker(ticker) for ticker in values)
    return list(dict.fromkeys(tickers))


def normalize_ticker(ticker: str) -> str:
    normalized = (ticker or "").strip().upper()
    if not normalized:
        raise ValueError("Ticker is required.")
    return normalized


def _has_fundamental_data(fundamentals: dict[str, Any]) -> bool:
    return any(value is not None for key, value in fundamentals.items() if not key.startswith("_"))


def sanitize_for_json(value: Any) -> Any:
    """Convert provider/library scalar types into MCP/JSON friendly values."""
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp | datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return sanitize_for_json(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

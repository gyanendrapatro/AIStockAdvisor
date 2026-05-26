from __future__ import annotations

from collections import Counter
import re
from typing import Any

import yfinance as yf

from stock_advisor.data.news import get_news, search_gdelt_articles, search_google_news


MATERIAL_EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings_results": (
        "earnings",
        "results",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
        "profit",
        "revenue",
        "ebitda",
        "margin",
        "concall",
        "conference call",
        "investor presentation",
    ),
    "expansion_capex": (
        "expansion",
        "capex",
        "capital expenditure",
        "investment",
        "invest",
        "capacity",
        "project",
        "launch",
        "data center",
        "datacenter",
        "development",
    ),
    "order_book": (
        "order book",
        "order inflow",
        "order win",
        "wins large",
        "wins order",
        "win order",
        "won order",
        "new order",
        "large order",
        "contract win",
        "backlog",
        "letter of award",
        "loa",
    ),
    "orders_contracts": (
        "order",
        "contract",
        "deal",
        "agreement",
        "partnership",
        "lease",
        "signed",
        "client",
    ),
    "new_business_area": (
        "new business",
        "new segment",
        "new vertical",
        "foray",
        "enters",
        "entry into",
        "diversification",
        "adjacent business",
        "new market",
    ),
    "innovation_product": (
        "innovation",
        "innovative",
        "technology",
        "patent",
        "r&d",
        "research and development",
        "launches",
        "unveils",
        "prototype",
        "platform",
        "ai",
        "automation",
        "advanced",
    ),
    "partnerships_alliances": (
        "partnership",
        "joint venture",
        "collaboration",
        "alliance",
        "mou",
        "memorandum of understanding",
        "tie-up",
        "tie up",
        "strategic partner",
    ),
    "debt_fundraising": (
        "debt",
        "loan",
        "fundraise",
        "fund raising",
        "qip",
        "rights issue",
        "preferential",
        "pledge",
        "credit",
    ),
    "legal_regulatory": (
        "case",
        "court",
        "lawsuit",
        "litigation",
        "tribunal",
        "nclt",
        "sebi",
        "penalty",
        "fine",
        "probe",
        "investigation",
        "tax demand",
        "regulatory",
    ),
    "corporate_actions": (
        "dividend",
        "split",
        "bonus",
        "buyback",
        "record date",
        "board meeting",
        "agm",
        "egm",
    ),
    "governance_management": (
        "resignation",
        "appointment",
        "director",
        "auditor",
        "promoter",
        "shareholding",
        "pledged",
        "management",
    ),
    "ratings_analyst": (
        "rating",
        "upgrade",
        "downgrade",
        "target price",
        "coverage",
        "recommendation",
    ),
}

BUSINESS_AREA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "real_estate_development": ("real estate", "residential", "commercial", "property", "township", "land", "development"),
    "data_centers_digital_infrastructure": ("data center", "datacenter", "cloud", "digital infrastructure", "server", "ai infrastructure"),
    "hospitality": ("hotel", "hospitality", "resort"),
    "leasing_rentals": ("lease", "leasing", "rental", "annuity"),
    "infrastructure_projects": ("infrastructure", "industrial park", "project", "construction"),
    "renewable_energy": ("solar", "renewable", "green energy", "wind"),
    "defence_security": ("defence", "defense", "drone", "missile", "shipbuilding", "aerospace", "radar"),
    "ev_battery_mobility": ("ev", "electric vehicle", "battery", "charging", "mobility", "lithium"),
    "water_environment": ("water treatment", "wastewater", "pollution control", "recycling", "effluent"),
    "manufacturing_electronics": ("electronics", "manufacturing", "semiconductor", "ems", "component"),
}

POSITIVE_EVENT_CATEGORIES = {
    "earnings_results",
    "expansion_capex",
    "order_book",
    "orders_contracts",
    "new_business_area",
    "innovation_product",
    "partnerships_alliances",
    "ratings_analyst",
}
RISK_EVENT_CATEGORIES = {"legal_regulatory", "debt_fundraising", "governance_management"}


def get_company_intelligence(
    ticker: str,
    *,
    fundamentals: dict[str, Any] | None = None,
    indicators: dict[str, Any] | None = None,
    seed_articles: list[dict[str, Any]] | None = None,
    days: int = 30,
    strategic_days: int = 365,
    limit: int = 12,
) -> dict[str, Any]:
    """Build a free-source company intelligence brief for a stock."""
    normalized_ticker = ticker.strip().upper()
    profile = _get_company_profile(normalized_ticker, fundamentals or {})
    aliases = _company_aliases(normalized_ticker, profile)

    articles = _collect_evidence(
        normalized_ticker,
        aliases,
        days=days,
        strategic_days=strategic_days,
        limit=limit,
        seed_articles=seed_articles,
    )
    classified = [_classify_article(article) for article in articles]
    classified = sorted(classified, key=lambda row: row.get("materiality_score", 0), reverse=True)

    category_counts = Counter(category for row in classified for category in row.get("categories", []))
    area_counts = Counter(area for row in classified for area in row.get("business_areas", []))
    profile_areas = _classify_business_areas(profile.get("business_summary") or "")
    for area in profile_areas:
        area_counts[area] += 2

    positive_catalysts = _top_titles(classified, POSITIVE_EVENT_CATEGORIES, max_days=90)
    order_book_updates = _top_titles(classified, {"order_book", "orders_contracts"}, max_days=180)
    innovation_updates = _top_titles(classified, {"innovation_product", "new_business_area", "partnerships_alliances"}, max_days=180)
    risk_flags = _top_titles(classified, RISK_EVENT_CATEGORIES)
    freshness = _freshness(classified)
    sector_fit = _sector_fit(profile, fundamentals or {}, indicators or {}, area_counts, category_counts)

    warnings = []
    if freshness["last_7_days"] == 0:
        warnings.append("No company-specific evidence found in the last 7 days from configured free sources.")
    if not classified:
        warnings.append("No company intelligence evidence found from Yahoo/GDELT; manual exchange filings review is still required.")
    warnings.append("Coverage is news-index based and does not replace official NSE/BSE filings, court databases, or paid corporate feeds.")

    return {
        "ticker": normalized_ticker,
        "profile": profile,
        "aliases_used": aliases,
        "business_areas": [area for area, _ in area_counts.most_common()],
        "material_event_counts": dict(category_counts),
        "freshness": freshness,
        "positive_catalysts": positive_catalysts,
        "order_book_updates": order_book_updates,
        "innovation_updates": innovation_updates,
        "risk_flags": risk_flags,
        "sector_fit": sector_fit,
        "evidence": classified[:limit],
        "monitoring_focus": _monitoring_focus(profile, area_counts, category_counts),
        "warnings": warnings,
        "providers": sorted({row.get("provider") for row in classified if row.get("provider")}),
    }


def _get_company_profile(ticker: str, fundamentals: dict[str, Any]) -> dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    return {
        "name": fundamentals.get("shortName") or info.get("shortName") or info.get("longName"),
        "sector": fundamentals.get("sector") or info.get("sector"),
        "industry": fundamentals.get("industry") or info.get("industry"),
        "website": info.get("website"),
        "country": info.get("country"),
        "market_cap": fundamentals.get("marketCap") or info.get("marketCap"),
        "business_summary": info.get("longBusinessSummary"),
    }


def _collect_evidence(
    ticker: str,
    aliases: list[str],
    *,
    days: int,
    strategic_days: int,
    limit: int,
    seed_articles: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = list(seed_articles) if seed_articles is not None else get_news(ticker, limit=min(limit, 8))

    google_query = _google_company_query(aliases)
    if google_query:
        rows.extend(search_google_news(google_query, limit=limit))

    company_query = _company_query(aliases)
    if company_query:
        rows.extend(search_gdelt_articles(company_query, limit=limit, timespan=f"{max(1, days)}d"))

    material_query = _company_query(aliases, MATERIAL_EVENT_KEYWORDS_FLAT)
    if material_query:
        rows.extend(search_gdelt_articles(material_query, limit=limit, timespan=f"{max(days, strategic_days)}d"))

    unique = _dedupe_articles(rows)
    return [row for row in unique if _is_company_relevant(row, aliases)]


MATERIAL_EVENT_KEYWORDS_FLAT = tuple(
    dict.fromkeys(keyword for values in MATERIAL_EVENT_KEYWORDS.values() for keyword in values)
)


def _company_aliases(ticker: str, profile: dict[str, Any]) -> list[str]:
    aliases = []
    base = ticker.split(".", 1)[0]
    for value in (profile.get("name"), base, base.replace("-", " ")):
        if value:
            aliases.append(str(value).strip())
    if profile.get("name"):
        cleaned = re.sub(r"\b(limited|ltd|inc|corp|corporation|company)\b\.?", "", str(profile["name"]), flags=re.I)
        if cleaned.strip():
            aliases.append(cleaned.strip())
    return list(dict.fromkeys(alias for alias in aliases if len(alias) >= 3))[:4]


def _company_query(aliases: list[str], terms: tuple[str, ...] | None = None) -> str:
    if not aliases:
        return ""
    alias_query = " OR ".join(f'"{alias}"' for alias in aliases[:3])
    if not terms:
        return f"({alias_query})"
    compact_terms = terms[:18]
    term_query = " OR ".join(f'"{term}"' if " " in term else term for term in compact_terms)
    return f"({alias_query}) ({term_query})"


def _google_company_query(aliases: list[str]) -> str:
    if not aliases:
        return ""
    primary = aliases[-1] if len(aliases) > 1 else aliases[0]
    return f'"{primary}" stock earnings results expansion "data center" court SEBI'


def _dedupe_articles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("url") or row.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _is_company_relevant(row: dict[str, Any], aliases: list[str]) -> bool:
    text = " ".join(str(row.get(key) or "") for key in ("title", "summary")).lower()
    compact_text = re.sub(r"[^a-z0-9]", "", text)
    for alias in aliases:
        normalized = alias.lower()
        compact_alias = re.sub(r"[^a-z0-9]", "", normalized)
        if normalized in text or compact_alias in compact_text:
            return True
    return False


def _classify_article(article: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(article.get(key) or "") for key in ("title", "summary", "source")).lower()
    categories = _match_keyword_groups(text, MATERIAL_EVENT_KEYWORDS)
    business_areas = _classify_business_areas(text)
    materiality_score = _materiality_score(article, categories)
    return {
        **article,
        "categories": categories,
        "business_areas": business_areas,
        "materiality_score": materiality_score,
    }


def _match_keyword_groups(text: str, groups: dict[str, tuple[str, ...]]) -> list[str]:
    matches = []
    for group, keywords in groups.items():
        if any(keyword in text for keyword in keywords):
            matches.append(group)
    return matches


def _classify_business_areas(text: str) -> list[str]:
    return _match_keyword_groups(text.lower(), BUSINESS_AREA_KEYWORDS)


def _materiality_score(article: dict[str, Any], categories: list[str]) -> float:
    score = 20 + 10 * len(categories)
    try:
        days_old = float(article.get("days_old"))
    except (TypeError, ValueError):
        days_old = None
    if days_old is not None:
        if days_old <= 2:
            score += 30
        elif days_old <= 7:
            score += 22
        elif days_old <= 30:
            score += 12
        elif days_old <= 180:
            score += 5
    try:
        sentiment = float(article.get("overall_sentiment_score") or 0)
    except (TypeError, ValueError):
        sentiment = 0
    score += abs(sentiment) * 20
    return round(min(100, score), 2)


def _freshness(rows: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"last_2_days": 0, "last_7_days": 0, "last_30_days": 0, "stale_or_unknown": 0}
    for row in rows:
        try:
            days_old = float(row.get("days_old"))
        except (TypeError, ValueError):
            buckets["stale_or_unknown"] += 1
            continue
        if days_old <= 2:
            buckets["last_2_days"] += 1
        if days_old <= 7:
            buckets["last_7_days"] += 1
        if days_old <= 30:
            buckets["last_30_days"] += 1
        if days_old > 30:
            buckets["stale_or_unknown"] += 1
    return buckets


def _top_titles(
    rows: list[dict[str, Any]],
    category_set: set[str],
    limit: int = 5,
    max_days: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        categories = set(row.get("categories", []))
        if not categories.intersection(category_set):
            continue
        if max_days is not None:
            try:
                if float(row.get("days_old")) > max_days:
                    continue
            except (TypeError, ValueError):
                continue
        selected.append(
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "provider": row.get("provider"),
                "days_old": row.get("days_old"),
                "categories": row.get("categories", []),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _sector_fit(
    profile: dict[str, Any],
    fundamentals: dict[str, Any],
    indicators: dict[str, Any],
    area_counts: Counter,
    category_counts: Counter,
) -> dict[str, Any]:
    score = 50
    reasons = []
    sector = profile.get("sector")
    industry = profile.get("industry")
    if sector or industry:
        reasons.append(f"Classified as {sector or 'unknown sector'} / {industry or 'unknown industry'}")

    if _positive_number(fundamentals.get("revenueGrowth"), threshold=0.10):
        score += 10
        reasons.append("Revenue growth is above 10%")
    if _positive_number(fundamentals.get("earningsGrowth"), threshold=0.10):
        score += 8
        reasons.append("Earnings growth is above 10%")
    if _positive_number(fundamentals.get("returnOnEquity"), threshold=0.10):
        score += 5
        reasons.append("ROE is above 10%")

    close = indicators.get("close")
    sma50 = indicators.get("sma_50")
    sma200 = indicators.get("sma_200")
    if close and sma50:
        if close > sma50:
            score += 6
            reasons.append("Price is above the 50-day average")
        else:
            score -= 4
            reasons.append("Price is below the 50-day average")
    if close and sma200:
        if close > sma200:
            score += 8
            reasons.append("Price confirms longer-term trend above the 200-day average")
        else:
            score -= 8
            reasons.append("Price is still below the 200-day average")

    if category_counts.get("expansion_capex"):
        score += 8
        reasons.append("Evidence mentions expansion, capex, projects, or investment")
    if category_counts.get("order_book"):
        score += 8
        reasons.append("Evidence mentions order book, order wins, backlog, or LOA")
    if category_counts.get("new_business_area"):
        score += 6
        reasons.append("Evidence mentions entry into a new business area or segment")
    if category_counts.get("innovation_product"):
        score += 6
        reasons.append("Evidence mentions innovation, technology, product launch, or R&D")
    if category_counts.get("partnerships_alliances"):
        score += 5
        reasons.append("Evidence mentions partnership, JV, collaboration, or MoU")
    if category_counts.get("earnings_results"):
        score += 5
        reasons.append("Evidence includes earnings/results context")
    if area_counts.get("data_centers_digital_infrastructure"):
        score += 8
        reasons.append("Evidence links the company to data centers or digital infrastructure")
    if category_counts.get("legal_regulatory"):
        score -= 10
        reasons.append("Legal or regulatory keywords appeared in evidence")
    if category_counts.get("debt_fundraising"):
        score -= 4
        reasons.append("Debt/fundraising keywords appeared in evidence")

    bounded = max(0, min(100, round(score, 2)))
    return {
        "score": bounded,
        "label": _fit_label(bounded),
        "sector": sector,
        "industry": industry,
        "reasons": reasons[:8],
    }


def _positive_number(value: Any, *, threshold: float) -> bool:
    try:
        return float(value) > threshold
    except (TypeError, ValueError):
        return False


def _fit_label(score: float) -> str:
    if score >= 75:
        return "strong fit"
    if score >= 60:
        return "moderate fit"
    if score >= 45:
        return "mixed fit"
    return "weak fit"


def _monitoring_focus(profile: dict[str, Any], area_counts: Counter, category_counts: Counter) -> list[str]:
    focus = []
    if category_counts.get("earnings_results"):
        focus.append("Track next quarterly revenue, margin, cash-flow, and management commentary.")
    if category_counts.get("expansion_capex"):
        focus.append("Verify whether expansion/capex plans convert into signed revenue and returns on capital.")
    if category_counts.get("order_book"):
        focus.append("Track order book conversion, margins on new orders, execution timeline, and client concentration.")
    if category_counts.get("new_business_area"):
        focus.append("Check whether the new business area has credible revenue visibility, capital allocation discipline, and management capability.")
    if category_counts.get("innovation_product"):
        focus.append("Validate product/technology adoption, pricing power, competitive moat, and commercialization timeline.")
    if category_counts.get("partnerships_alliances"):
        focus.append("Verify partnership economics, exclusivity, revenue share, and whether announcements convert into orders.")
    if area_counts.get("data_centers_digital_infrastructure"):
        focus.append("Monitor data-center capacity additions, utilization, customer wins, and power/capex funding.")
    if category_counts.get("legal_regulatory"):
        focus.append("Review official filings and court/regulatory updates before position sizing.")
    if category_counts.get("debt_fundraising"):
        focus.append("Watch leverage, pledge, QIP/preferential issuance, and interest-cost changes.")
    if not focus:
        sector = profile.get("sector") or "the company"
        focus.append(f"Monitor fresh filings, sector news, and price-volume confirmation for {sector}.")
    return focus[:5]

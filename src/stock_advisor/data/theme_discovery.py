from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re
from typing import Any

from stock_advisor.config.settings import load_watchlists
from stock_advisor.data.dhan import get_dhan_portfolio_summary
from stock_advisor.data.news import search_gdelt_articles, search_google_news


THEMES: list[dict[str, Any]] = [
    {
        "id": "power_grid_electrification",
        "title": "Power grid, electrification, and equipment",
        "query": 'India power demand transmission transformer smart meter capex order book order win stock',
        "keywords": ("power demand", "transmission", "transformer", "smart meter", "grid", "electrification", "order", "order book", "order win", "capex"),
        "beneficiary_sectors": ("Capital goods", "Electrical equipment", "Power infrastructure", "Smart metering"),
        "portfolio_symbols": ("TDPOWERSYS", "TECHNOE", "GENUSPOWER", "BBL", "KPIL", "TRITURBINE"),
        "stocks_to_check": ("TDPOWERSYS.NS", "TECHNOE.NS", "GENUSPOWER.NS", "KPIL.NS", "BBL.NS", "POWERGRID.NS"),
        "thesis": "Rising electricity demand and grid capex can benefit electrical equipment, EPC, and smart-metering names.",
    },
    {
        "id": "renewable_energy",
        "title": "Renewable energy and clean-tech capex",
        "query": 'India renewable energy solar wind green hydrogen capex order book innovation stock',
        "keywords": ("renewable", "solar", "wind", "green hydrogen", "energy transition", "module", "turbine", "capacity", "order book", "innovation"),
        "beneficiary_sectors": ("Solar", "Wind", "Renewable EPC", "Industrial machinery"),
        "portfolio_symbols": ("INOXWIND", "SWSOLAR", "SWELECTES", "SHAKTIPUMP", "PRAJIND", "KPIL"),
        "stocks_to_check": ("INOXWIND.NS", "SWSOLAR.NS", "SWELECTES.NS", "SHAKTIPUMP.NS", "PRAJIND.NS", "KPIL.NS"),
        "thesis": "Renewable project awards and capex can benefit wind, solar, pumps, ethanol/bioenergy, and EPC suppliers.",
    },
    {
        "id": "data_centers_ai_infra",
        "title": "Data centers, AI infrastructure, and digital capex",
        "query": 'India data center AI infrastructure cloud capex power cooling innovation new business stock',
        "keywords": ("data center", "datacenter", "ai infrastructure", "cloud", "server", "digital infrastructure", "cooling", "innovation", "new business"),
        "beneficiary_sectors": ("Data centers", "Power EPC", "IT services", "Digital infrastructure", "Real estate"),
        "portfolio_symbols": ("TECHNOE", "PROTEAN", "INFY", "KPIL"),
        "stocks_to_check": ("TECHNOE.NS", "ANANTRAJ.NS", "NETWEB.NS", "PROTEAN.NS", "INFY.NS", "TCS.NS"),
        "thesis": "AI/cloud growth can benefit data-center operators, power/cooling contractors, digital platforms, and IT services.",
    },
    {
        "id": "infrastructure_capex",
        "title": "Infrastructure, roads, railways, and public capex",
        "query": 'India infrastructure capex railways roads EPC order book order win construction stock',
        "keywords": ("infrastructure", "railway", "roads", "highway", "metro", "epc", "construction", "order book", "order win", "contract win"),
        "beneficiary_sectors": ("Infrastructure EPC", "Engineering", "Construction", "Capital goods"),
        "portfolio_symbols": ("KPIL", "TECHNOE", "PRAJIND", "LT", "TRITURBINE"),
        "stocks_to_check": ("LT.NS", "KPIL.NS", "TECHNOE.NS", "PRAJIND.NS", "RVNL.NS", "IRCON.NS"),
        "thesis": "Public/private capex and project ordering can benefit EPC, construction, and capital-goods companies.",
    },
    {
        "id": "gas_lng_cgd",
        "title": "Gas, LNG, and city gas distribution",
        "query": 'India natural gas LNG city gas distribution CNG PNG price policy stock',
        "keywords": ("natural gas", "lng", "city gas", "cng", "png", "gas distribution", "gas price"),
        "beneficiary_sectors": ("City gas distribution", "Gas utilities", "LNG", "Energy distribution"),
        "portfolio_symbols": ("MGL",),
        "stocks_to_check": ("MGL.NS", "IGL.NS", "GUJGASLTD.NS", "GAIL.NS", "PETRONET.NS"),
        "thesis": "Gas policy, LNG prices, and CNG/PNG demand can move city-gas distributors and gas utilities.",
    },
    {
        "id": "water_environment",
        "title": "Water, waste treatment, and environmental infrastructure",
        "query": 'India water treatment waste recycling pollution control infrastructure order innovation stock',
        "keywords": ("water treatment", "wastewater", "pollution control", "recycling", "environment", "effluent", "desalination", "order", "innovation"),
        "beneficiary_sectors": ("Water treatment", "Pollution controls", "Environmental infrastructure"),
        "portfolio_symbols": ("IONEXCHANG", "GANECOS"),
        "stocks_to_check": ("IONEXCHANG.NS", "GANECOS.NS", "VAWATER.NS"),
        "thesis": "Water scarcity, industrial treatment, recycling, and environmental rules can benefit treatment and recycling companies.",
    },
    {
        "id": "banking_credit_rates",
        "title": "Banking, credit growth, and rates",
        "query": 'India banks credit growth RBI rates liquidity deposit loan growth stock',
        "keywords": ("credit growth", "rbi", "interest rate", "liquidity", "deposit", "loan growth", "bank"),
        "beneficiary_sectors": ("Banks", "NBFCs", "Credit services", "Financial services"),
        "portfolio_symbols": ("RECLTD", "HDFCBANK", "ICICIBANK", "SBIN"),
        "stocks_to_check": ("HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "RECLTD.NS", "PFC.NS"),
        "thesis": "Credit growth, liquidity, and rate-cycle changes can affect banks, PSU financiers, and NBFCs.",
    },
    {
        "id": "ev_batteries_auto_ancillaries",
        "title": "EV, batteries, and auto ancillaries",
        "query": 'India EV battery auto ancillary lithium cell manufacturing demand innovation new product stock',
        "keywords": ("ev", "electric vehicle", "battery", "lithium", "cell manufacturing", "auto ancillary", "charging", "innovation", "new product"),
        "beneficiary_sectors": ("Batteries", "Auto ancillaries", "EV components", "Charging infrastructure"),
        "portfolio_symbols": ("EXIDEIND", "SHAKTIPUMP"),
        "stocks_to_check": ("EXIDEIND.NS", "ARE&M.NS", "TMPV.NS", "OLECTRA.NS"),
        "thesis": "EV adoption and battery localization can benefit battery makers, auto ancillaries, and charging ecosystem names.",
    },
    {
        "id": "it_ai_services",
        "title": "IT services, AI spending, and outsourcing",
        "query": 'India IT services AI cloud outsourcing deal innovation new platform stock Infosys TCS',
        "keywords": ("it services", "outsourcing", "ai", "cloud", "deal", "digital transformation", "technology spending", "innovation", "platform"),
        "beneficiary_sectors": ("IT services", "Cloud", "Digital engineering", "Cybersecurity"),
        "portfolio_symbols": ("INFY", "TCS"),
        "stocks_to_check": ("INFY.NS", "TCS.NS", "HCLTECH.NS", "LTM.NS", "PERSISTENT.NS"),
        "thesis": "Enterprise AI/cloud spending and outsourcing budgets can support IT services, but margin and disruption risk matter.",
    },
    {
        "id": "consumer_jewellery",
        "title": "Consumer demand, gold, and jewellery retail",
        "query": 'India jewellery retail gold demand festive wedding consumption stock',
        "keywords": ("jewellery", "gold", "wedding", "festive", "retail", "consumption", "store expansion"),
        "beneficiary_sectors": ("Jewellery", "Retail", "Consumer discretionary"),
        "portfolio_symbols": ("KALYANKJIL",),
        "stocks_to_check": ("KALYANKJIL.NS", "TITAN.NS", "SENCO.NS"),
        "thesis": "Wedding/festive demand, gold prices, and store expansion can drive jewellery and discretionary retail names.",
    },
    {
        "id": "geopolitics_defence_security",
        "title": "Geopolitics, war risk, and defence/security",
        "query": 'India geopolitics war conflict border defence security missile drone shipbuilding order book innovation sanctions stock',
        "keywords": (
            "geopolitics",
            "war",
            "conflict",
            "border",
            "defence",
            "defense",
            "security",
            "missile",
            "drone",
            "shipbuilding",
            "sanctions",
            "order book",
            "innovation",
        ),
        "beneficiary_sectors": ("Defence", "Aerospace", "Shipbuilding", "Electronics", "Cybersecurity"),
        "portfolio_symbols": (),
        "stocks_to_check": ("HAL.NS", "BEL.NS", "BDL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "DATAPATTNS.NS"),
        "thesis": "Geopolitical tension and security spending can benefit defence manufacturers, shipbuilders, electronics, and cybersecurity suppliers.",
    },
    {
        "id": "trade_tariffs_export_bans",
        "title": "Trade policy, tariffs, export bans, and import restrictions",
        "query": 'India tariff export ban import restriction anti dumping duty trade policy China US stock',
        "keywords": (
            "tariff",
            "export ban",
            "import restriction",
            "anti-dumping",
            "anti dumping",
            "duty",
            "trade policy",
            "china",
            "us",
            "sanctions",
        ),
        "beneficiary_sectors": ("Domestic manufacturing", "Specialty chemicals", "Textiles", "Metals", "Electronics manufacturing"),
        "portfolio_symbols": ("GANECOS",),
        "stocks_to_check": ("DIXON.NS", "KAYNES.NS", "PGEL.NS", "AARTIIND.NS", "SRF.NS", "WELSPUNLIV.NS"),
        "thesis": "Tariffs, bans, and import substitution can shift demand toward domestic manufacturers, chemicals, electronics, and exporters.",
    },
    {
        "id": "government_policy_regulation",
        "title": "Government policy, regulation, PLI, and budget decisions",
        "query": 'India government policy regulation PLI budget subsidy approval tender order book new business stock sector',
        "keywords": (
            "government policy",
            "regulation",
            "pli",
            "budget",
            "subsidy",
            "approval",
            "tender",
            "order book",
            "new business",
            "incentive",
            "scheme",
            "allocation",
        ),
        "beneficiary_sectors": ("Manufacturing", "Infrastructure", "Renewables", "Electronics", "Railways"),
        "portfolio_symbols": ("KPIL", "TECHNOE", "SWSOLAR", "PRAJIND", "GENUSPOWER"),
        "stocks_to_check": ("DIXON.NS", "KAYNES.NS", "RVNL.NS", "IRCON.NS", "BHEL.NS", "BEML.NS"),
        "thesis": "Policy changes, PLI schemes, budget allocations, and tenders can create sector-level catalysts before company results show them.",
    },
    {
        "id": "oil_gas_commodities",
        "title": "Oil, gas, commodities, and input-cost shocks",
        "query": 'India crude oil gas LNG commodity price inflation OPEC supply disruption stock',
        "keywords": (
            "crude oil",
            "oil price",
            "gas",
            "lng",
            "commodity",
            "opec",
            "supply disruption",
            "input cost",
            "inflation",
        ),
        "beneficiary_sectors": ("Oil and gas", "City gas", "Upstream energy", "Commodity producers", "Paints/chemicals risk"),
        "portfolio_symbols": ("MGL",),
        "stocks_to_check": ("ONGC.NS", "OIL.NS", "GAIL.NS", "PETRONET.NS", "IGL.NS", "GUJGASLTD.NS"),
        "thesis": "Oil/gas and commodity shocks can help producers but hurt input-cost sensitive consumers and manufacturers.",
    },
    {
        "id": "currency_rates_inflation",
        "title": "Currency, inflation, RBI rates, and liquidity",
        "query": 'India rupee dollar inflation RBI rate cut rate hike liquidity bond yield stock market',
        "keywords": (
            "rupee",
            "dollar",
            "inflation",
            "rbi",
            "rate cut",
            "rate hike",
            "liquidity",
            "bond yield",
            "forex",
        ),
        "beneficiary_sectors": ("Banks", "NBFCs", "IT exporters", "Pharma exporters", "Rate-sensitive consumption"),
        "portfolio_symbols": ("INFY", "RECLTD"),
        "stocks_to_check": ("HCLTECH.NS", "PERSISTENT.NS", "SUNPHARMA.NS", "CIPLA.NS", "PFC.NS", "BAJFINANCE.NS"),
        "thesis": "Currency, inflation, and rate-cycle moves can affect exporters, lenders, consumption, and valuation multiples.",
    },
    {
        "id": "agri_weather_monsoon",
        "title": "Weather, monsoon, agriculture, and rural demand",
        "query": 'India monsoon rainfall heatwave agriculture rural demand fertilizer irrigation pump innovation stock',
        "keywords": (
            "monsoon",
            "rainfall",
            "heatwave",
            "agriculture",
            "rural demand",
            "fertilizer",
            "irrigation",
            "pump",
            "crop",
            "innovation",
        ),
        "beneficiary_sectors": ("Agri inputs", "Irrigation", "Pumps", "Fertilizers", "Rural consumption"),
        "portfolio_symbols": ("SHAKTIPUMP", "EXIDEIND"),
        "stocks_to_check": ("COROMANDEL.NS", "CHAMBLFERT.NS", "VSTTILLERS.NS", "KSB.NS", "KIRLOSBROS.NS"),
        "thesis": "Monsoon and weather trends can affect rural demand, irrigation, pumps, fertilizers, and agri-input companies.",
    },
    {
        "id": "order_book_momentum",
        "title": "Order-book momentum and large contract wins",
        "query": 'India listed company order book order win contract win letter of award LOA backlog stock',
        "keywords": (
            "order book",
            "order win",
            "wins order",
            "contract win",
            "letter of award",
            "loa",
            "backlog",
            "large order",
            "new order",
        ),
        "beneficiary_sectors": ("Capital goods", "Infrastructure EPC", "Railways", "Defence", "Industrial machinery"),
        "portfolio_symbols": ("KPIL", "TECHNOE", "GENUSPOWER", "TDPOWERSYS", "TRITURBINE", "PRAJIND"),
        "stocks_to_check": ("BHEL.NS", "BEML.NS", "RVNL.NS", "IRCON.NS", "BEL.NS", "SIEMENS.NS", "ABB.NS"),
        "thesis": "Large order wins and improving order books can precede revenue acceleration in EPC, capital goods, railways, and defence names.",
    },
    {
        "id": "innovation_new_business_models",
        "title": "Innovation, product launches, and new business areas",
        "query": 'India listed company innovation new product launch new business segment foray platform technology stock',
        "keywords": (
            "innovation",
            "new product",
            "launch",
            "new business",
            "new segment",
            "foray",
            "platform",
            "technology",
            "r&d",
            "patent",
        ),
        "beneficiary_sectors": ("Technology", "Electronics", "EV/Batteries", "Digital platforms", "Specialty manufacturing"),
        "portfolio_symbols": ("PROTEAN", "INFY", "EXIDEIND", "TECHNOE", "GENUSPOWER"),
        "stocks_to_check": ("NETWEB.NS", "KAYNES.NS", "DIXON.NS", "TATAELXSI.NS", "KPITTECH.NS", "CYIENT.NS"),
        "thesis": "New product launches, platform shifts, and credible entry into adjacent businesses can create fresh growth vectors.",
    },
]


def discover_market_themes(
    *,
    market: str = "india",
    days: int = 7,
    limit_per_theme: int = 6,
    max_themes: int = 8,
    include_portfolio: bool = True,
    include_watchlists: bool = True,
) -> dict[str, Any]:
    """Discover news-driven market themes and stocks to check from free sources."""
    market_normalized = market.strip().lower() or "india"
    days = max(1, min(int(days), 30))
    limit_per_theme = max(1, min(int(limit_per_theme), 12))
    max_themes = max(1, min(int(max_themes), len(THEMES)))
    portfolio_symbols = _portfolio_symbols() if include_portfolio else set()
    watchlist_symbols = _watchlist_symbols() if include_watchlists else set()
    known_symbols = portfolio_symbols | watchlist_symbols

    themes = []
    all_evidence: list[dict[str, Any]] = []
    for theme in THEMES:
        evidence = _fetch_theme_evidence(theme, market_normalized, days=days, limit=limit_per_theme)
        all_evidence.extend(evidence)
        scored = _score_theme(theme, evidence, portfolio_symbols, watchlist_symbols, known_symbols)
        if scored["evidence_count"] or scored["portfolio_matches"] or scored["watchlist_matches"]:
            themes.append(scored)

    ranked = sorted(themes, key=lambda row: row["score"], reverse=True)[:max_themes]
    return {
        "market": market_normalized,
        "days": days,
        "theme_count": len(ranked),
        "themes": ranked,
        "top_sectors": _top_sectors(ranked),
        "top_new_stocks_to_check": _top_stocks_to_check(ranked, new_only=True),
        "top_stocks_to_check": _top_stocks_to_check(ranked, new_only=False),
        "known_portfolio_symbols": sorted(portfolio_symbols),
        "known_watchlist_symbols": sorted(watchlist_symbols),
        "evidence_count": len(_dedupe_articles(all_evidence)),
        "providers": sorted({str(row.get("provider")) for row in all_evidence if row.get("provider")}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": [
            "Theme discovery is news-index based and should be validated against official filings and price-volume confirmation.",
            "top_new_stocks_to_check excludes symbols already present in the Dhan portfolio or configured watchlists.",
            "This tool identifies stocks to check; it is not a buy/sell recommendation.",
        ],
    }


def _fetch_theme_evidence(theme: dict[str, Any], market: str, *, days: int, limit: int) -> list[dict[str, Any]]:
    query = f'{market} {theme["query"]}'.strip()
    rows = []
    rows.extend(search_google_news(query, limit=limit))
    rows.extend(search_gdelt_articles(query, limit=limit, timespan=f"{days}d"))
    unique = _dedupe_articles(rows)
    return [_classify_theme_article(row, theme) for row in unique]


def _classify_theme_article(article: dict[str, Any], theme: dict[str, Any]) -> dict[str, Any]:
    text = _article_text(article)
    matched = [keyword for keyword in theme["keywords"] if keyword in text]
    recency_score = _recency_score(article.get("days_old"))
    sentiment = _number(article.get("overall_sentiment_score"))
    impact_score = 25 + len(matched) * 8 + recency_score + max(-10, min(10, sentiment * 20))
    return {
        "title": article.get("title"),
        "summary": article.get("summary"),
        "url": article.get("url"),
        "provider": article.get("provider"),
        "source": article.get("source"),
        "days_old": article.get("days_old"),
        "sentiment": article.get("overall_sentiment_label"),
        "matched_keywords": matched[:8],
        "impact_score": round(max(0, min(100, impact_score)), 2),
    }


def _score_theme(
    theme: dict[str, Any],
    evidence: list[dict[str, Any]],
    portfolio_symbols: set[str],
    watchlist_symbols: set[str],
    known_symbols: set[str],
) -> dict[str, Any]:
    evidence_count = len(evidence)
    keyword_counter = Counter(keyword for row in evidence for keyword in row.get("matched_keywords", []))
    freshness = _freshness(evidence)
    portfolio_matches = _matches(theme["portfolio_symbols"], portfolio_symbols)
    watchlist_matches = _matches(theme["portfolio_symbols"], watchlist_symbols)
    avg_impact = sum(_number(row.get("impact_score")) for row in evidence) / evidence_count if evidence_count else 0
    score = min(100, avg_impact + min(20, evidence_count * 4) + min(10, len(portfolio_matches) * 2))
    stocks_to_check = list(dict.fromkeys((*portfolio_matches, *watchlist_matches, *theme["stocks_to_check"])))
    new_stocks_to_check = [
        ticker for ticker in stocks_to_check if _symbol_from_ticker(ticker) not in known_symbols
    ]

    return {
        "theme_id": theme["id"],
        "title": theme["title"],
        "thesis": theme["thesis"],
        "score": round(score, 2),
        "confidence": _confidence(score, evidence_count),
        "beneficiary_sectors": list(theme["beneficiary_sectors"]),
        "freshness": freshness,
        "evidence_count": evidence_count,
        "matched_keywords": [keyword for keyword, _ in keyword_counter.most_common(8)],
        "portfolio_matches": portfolio_matches,
        "watchlist_matches": watchlist_matches,
        "new_stocks_to_check": new_stocks_to_check[:12],
        "stocks_to_check": stocks_to_check[:12],
        "evidence": sorted(evidence, key=lambda row: _number(row.get("impact_score")), reverse=True)[:5],
    }


def _portfolio_symbols() -> set[str]:
    try:
        summary = get_dhan_portfolio_summary(include_market_values=False)
    except Exception:
        return set()
    return {
        str(row.get("tradingSymbol") or row.get("symbol") or "").strip().upper()
        for row in summary.get("holdings", [])
        if row.get("tradingSymbol") or row.get("symbol")
    }


def _watchlist_symbols() -> set[str]:
    symbols = set()
    for tickers in load_watchlists().values():
        for ticker in tickers:
            symbols.add(_symbol_from_ticker(ticker))
    return symbols


def _matches(theme_symbols: tuple[str, ...], owned_or_watched: set[str]) -> list[str]:
    rows = []
    for symbol in theme_symbols:
        normalized = _symbol_from_ticker(symbol)
        if normalized in owned_or_watched:
            rows.append(_ticker_for_symbol(normalized))
    return rows


def _top_sectors(themes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for theme in themes:
        for sector in theme.get("beneficiary_sectors", []):
            scores[sector] = scores.get(sector, 0.0) + _number(theme.get("score"))
    return [
        {"sector": sector, "score": round(score, 2)}
        for sector, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]
    ]


def _top_stocks_to_check(themes: list[dict[str, Any]], *, new_only: bool) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for theme in themes:
        key = "new_stocks_to_check" if new_only else "stocks_to_check"
        for ticker in theme.get(key, []):
            scores[ticker] = scores.get(ticker, 0.0) + _number(theme.get("score"))
            reasons.setdefault(ticker, []).append(theme["title"])
    return [
        {
            "ticker": ticker,
            "score": round(score, 2),
            "themes": reasons[ticker][:4],
            "already_known": False if new_only else None,
        }
        if new_only
        else {"ticker": ticker, "score": round(score, 2), "themes": reasons[ticker][:4]}
        for ticker, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:15]
    ]


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


def _freshness(rows: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"last_2_days": 0, "last_7_days": 0, "last_30_days": 0, "stale_or_unknown": 0}
    for row in rows:
        days_old = row.get("days_old")
        try:
            days = float(days_old)
        except (TypeError, ValueError):
            buckets["stale_or_unknown"] += 1
            continue
        if days <= 2:
            buckets["last_2_days"] += 1
        if days <= 7:
            buckets["last_7_days"] += 1
        if days <= 30:
            buckets["last_30_days"] += 1
        if days > 30:
            buckets["stale_or_unknown"] += 1
    return buckets


def _recency_score(days_old: Any) -> float:
    try:
        days = float(days_old)
    except (TypeError, ValueError):
        return 0
    if days <= 2:
        return 30
    if days <= 7:
        return 22
    if days <= 30:
        return 10
    return 0


def _confidence(score: float, evidence_count: int) -> float:
    return round(max(35, min(92, 40 + score * 0.35 + evidence_count * 4)), 1)


def _article_text(article: dict[str, Any]) -> str:
    return " ".join(str(article.get(key) or "") for key in ("title", "summary", "source")).lower()


def _symbol_from_ticker(ticker: str) -> str:
    return re.sub(r"\.(NS|BO)$", "", str(ticker).strip().upper())


def _ticker_for_symbol(symbol: str) -> str:
    if "." in symbol:
        return symbol.upper()
    return f"{symbol.upper()}.NS"


def _number(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from fastmcp import FastMCP

from stock_advisor.analysis.chart_patterns import detect_chart_patterns as _detect_chart_patterns
from stock_advisor.analysis.indicators import add_indicators, latest_indicators
from stock_advisor.analysis.market_analytics import (
    get_industry_analytics as _get_industry_analytics,
    get_market_breadth as _get_market_breadth,
    get_market_indices as _get_market_indices,
    get_relative_rotation_graph as _get_relative_rotation_graph,
    get_sector_analytics as _get_sector_analytics,
    get_top_gainers as _get_top_gainers,
    list_industry_definitions as _list_industry_definitions,
    rank_industry_stocks as _rank_industry_stocks,
)
from stock_advisor.analysis.pipeline import (
    analyze_stock as _analyze_stock,
    compare_stocks as _compare_stocks,
    rank_watchlist as _rank_watchlist,
    research_stock as _research_stock,
    sanitize_for_json,
)
from stock_advisor.analysis.portfolio import analyze_portfolio_holdings as _analyze_portfolio_holdings
from stock_advisor.analysis.sector_rotation import (
    discover_sector_opportunities as _discover_sector_opportunities,
    get_sector_rotation as _get_sector_rotation,
    list_sector_definitions as _list_sector_definitions,
    rank_sector_stocks as _rank_sector_stocks,
)
from stock_advisor.agents.sector_rotation_workflow import run_sector_rotation_workflow as _run_sector_rotation_workflow
from stock_advisor.config.settings import PROJECT_ROOT, load_watchlists, settings
from stock_advisor.data.analyst_events import (
    get_analyst_insights as _get_analyst_insights,
    get_stock_events as _get_stock_events,
)
from stock_advisor.data.company_intelligence import get_company_intelligence as _get_company_intelligence
from stock_advisor.data.dhan import (
    dhan_config_status,
    get_dhan_fund_limits as _get_dhan_fund_limits,
    get_dhan_holdings as _get_dhan_holdings,
    get_dhan_portfolio_summary as _get_dhan_portfolio_summary,
    get_dhan_positions as _get_dhan_positions,
    get_dhan_profile as _get_dhan_profile,
)
from stock_advisor.data.market_data import (
    get_basic_fundamentals as _get_basic_fundamentals,
    get_price_cache_status as _get_price_cache_status,
    get_price_history as _get_price_history,
    refresh_latest_exchange_eod_cache as _refresh_latest_exchange_eod_cache,
    warm_price_history_cache as _warm_price_history_cache,
)
from stock_advisor.data.daily_refresh import run_daily_market_data_refresh as _run_daily_market_data_refresh
from stock_advisor.data.news import get_news as _get_news
from stock_advisor.data.ownership import (
    get_ownership_fundamentals as _get_ownership_fundamentals,
    ownership_data_file_exists,
)
from stock_advisor.data.theme_discovery import discover_market_themes as _discover_market_themes
from stock_advisor.data.universe import (
    list_sector_constituents as _list_sector_constituents,
    list_stock_universe as _list_stock_universe,
    load_stock_universe as _load_stock_universe,
    refresh_bse_stock_universe as _refresh_bse_stock_universe,
    refresh_full_stock_universe as _refresh_full_stock_universe,
    refresh_india_stock_universe as _refresh_india_stock_universe,
    refresh_stock_universe as _refresh_stock_universe,
)
from stock_advisor.reporting.daily_report import generate_daily_report as _generate_daily_report

mcp = FastMCP("AI Stock Advisor")


@mcp.tool("server_info")
def server_info() -> dict[str, Any]:
    """Return MCP server capabilities and current default configuration."""
    watchlists = load_watchlists()
    return {
        "name": "AI Stock Advisor",
        "version": "0.1.0",
        "purpose": "Free-only educational stock analysis and watchlist triage. Not financial advice.",
        "default_period": settings.default_period,
        "default_interval": settings.default_interval,
        "data_model": "No paid API keys. Uses yfinance/Yahoo Finance, Yahoo Chart fallback prices, optional free-key Stooq fallback, SEC EDGAR US filing facts, Yahoo/Google News RSS/GDELT evidence, local technical indicators, local chart-pattern detection, local scoring, and local sentiment.",
        "watchlist_groups": sorted(watchlists),
        "tools": [
            "server_info",
            "health_check",
            "get_watchlist",
            "analyze_stock",
            "research_stock",
            "rank_watchlist",
            "compare_stocks",
            "get_stock_profile",
            "get_stock_fundamentals",
            "get_stock_news",
            "get_latest_technical_indicators",
            "get_chart_patterns",
            "get_analyst_insights",
            "get_stock_events",
            "discover_market_themes",
            "list_sector_constituents",
            "list_stock_universe",
            "refresh_bse_stock_universe",
            "refresh_full_stock_universe",
            "refresh_india_stock_universe",
            "refresh_stock_universe",
            "refresh_latest_exchange_eod_cache",
            "run_daily_market_data_refresh",
            "list_sector_definitions",
            "get_sector_rotation",
            "rank_sector_stocks",
            "discover_sector_opportunities",
            "run_sector_rotation_workflow",
            "get_sector_analytics",
            "list_industry_definitions",
            "get_industry_analytics",
            "rank_industry_stocks",
            "get_market_indices",
            "get_market_breadth",
            "get_top_gainers",
            "get_relative_rotation_graph",
            "get_company_intelligence",
            "get_ownership_fundamentals",
            "get_dhan_profile",
            "get_dhan_holdings",
            "get_dhan_positions",
            "get_dhan_fund_limits",
            "get_dhan_portfolio_stocks",
            "get_dhan_portfolio_summary",
            "analyze_dhan_portfolio",
            "get_price_history",
            "get_historical_prices",
            "market_snapshot",
            "generate_daily_report",
        ],
    }


@mcp.tool("health_check")
def health_check() -> dict[str, Any]:
    """Return configuration and provider readiness for the stock advisor."""
    watchlists = load_watchlists()
    return {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configured_watchlists": {group: len(tickers) for group, tickers in watchlists.items()},
        "providers": {
            "yfinance_market_data": {"configured": True, "required": True, "cost": "free"},
            "yahoo_chart_price_fallback": {"configured": True, "required": False, "cost": "free"},
            "stooq_price_fallback": {"configured": bool(os.getenv("STOOQ_API_KEY")), "required": False, "cost": "free"},
            "sec_edgar_us_fundamentals": {"configured": True, "required": False, "cost": "free"},
            "local_ownership_governance": {"configured": ownership_data_file_exists(), "required": False, "cost": "free"},
            "yahoo_finance_news": {"configured": True, "required": False, "cost": "free"},
            "gdelt_news": {"configured": True, "required": False, "cost": "free"},
            "gdelt_company_intelligence": {"configured": True, "required": False, "cost": "free"},
            "dhan_trading_api_read_only": dhan_config_status(),
            "local_chart_patterns": {"configured": True, "required": True, "cost": "free"},
            "local_sector_rotation": {"configured": True, "required": True, "cost": "free"},
            "local_market_analytics": {"configured": True, "required": True, "cost": "free"},
            "nse_total_market_universe": {"configured": True, "required": False, "cost": "free"},
            "nse_full_equity_universe": {"configured": True, "required": False, "cost": "free"},
            "bse_full_equity_universe": {"configured": True, "required": False, "cost": "free"},
            "india_nse_bse_equity_universe": {"configured": True, "required": False, "cost": "free"},
            "local_sentiment": {"configured": True, "required": True, "cost": "free"},
            "local_commentary": {"configured": True, "required": True, "cost": "free"},
        },
        "report_dir": str(settings.report_dir),
    }


@mcp.tool("get_watchlist")
def get_watchlist(group: str | None = None) -> dict[str, list[str]] | list[str]:
    """Return all configured watchlists or one group such as india/us."""
    watchlists = load_watchlists()
    if group is None:
        return watchlists
    normalized_group = group.strip().lower()
    if normalized_group not in watchlists:
        raise ValueError(f"Unknown watchlist group: {group}. Expected one of: {', '.join(sorted(watchlists))}")
    return watchlists[normalized_group]


@mcp.tool("analyze_stock")
def analyze_stock(
    ticker: str,
    period: str | None = None,
    interval: str | None = None,
    include_news: bool = True,
    include_intelligence: bool = False,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Analyze one ticker using technicals, fundamentals, news sentiment, risk, and liquidity."""
    return _analyze_stock(
        ticker,
        period=period,
        interval=interval,
        include_news=include_news,
        include_intelligence=include_intelligence,
        force_refresh_prices=force_refresh_prices,
    )


@mcp.tool("research_stock")
def research_stock(
    ticker: str,
    period: str | None = None,
    interval: str | None = None,
    intelligence_days: int = 30,
    intelligence_strategic_days: int = 365,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Run deep stock research with company intelligence, recent events, themes, and sector fit."""
    return _research_stock(
        ticker,
        period=period,
        interval=interval,
        intelligence_days=intelligence_days,
        intelligence_strategic_days=intelligence_strategic_days,
        force_refresh_prices=force_refresh_prices,
    )


@mcp.tool("rank_watchlist")
def rank_watchlist(
    group: str | None = None,
    limit: int | None = None,
    period: str | None = None,
    interval: str | None = None,
    include_news: bool = True,
    include_intelligence: bool = False,
    force_refresh_prices: bool = True,
) -> list[dict[str, Any]]:
    """Rank all configured tickers or one configured watchlist group."""
    return _rank_watchlist(
        group,
        limit=limit,
        period=period,
        interval=interval,
        include_news=include_news,
        include_intelligence=include_intelligence,
        force_refresh_prices=force_refresh_prices,
    )


@mcp.tool("compare_stocks")
def compare_stocks(
    tickers: list[str],
    period: str | None = None,
    interval: str | None = None,
    include_news: bool = True,
    include_intelligence: bool = False,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Analyze and rank an explicit ticker list for side-by-side comparison."""
    return _compare_stocks(
        tickers,
        period=period,
        interval=interval,
        include_news=include_news,
        include_intelligence=include_intelligence,
        force_refresh_prices=force_refresh_prices,
    )


@mcp.tool("get_stock_profile")
def get_stock_profile(ticker: str) -> dict[str, Any]:
    """Return compact company profile fields from free fundamental providers."""
    fundamentals = _get_basic_fundamentals(ticker)
    return sanitize_for_json(
        {
            "ticker": ticker.strip().upper(),
            "name": fundamentals.get("shortName"),
            "sector": fundamentals.get("sector"),
            "industry": fundamentals.get("industry"),
            "market_cap": fundamentals.get("marketCap"),
            "beta": fundamentals.get("beta"),
            "dividend_yield": fundamentals.get("dividendYield"),
            "providers": fundamentals.get("_sources", []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


@mcp.tool("get_stock_fundamentals")
def get_stock_fundamentals(ticker: str) -> dict[str, Any]:
    """Return free fundamentals plus optional local ownership/governance fields."""
    fundamentals = _get_basic_fundamentals(ticker)
    return sanitize_for_json(
        {
            "ticker": ticker.strip().upper(),
            "fundamentals": fundamentals,
            "providers": fundamentals.get("_sources", []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


@mcp.tool("get_stock_news")
def get_stock_news(ticker: str, limit: int = 10) -> dict[str, Any]:
    """Return free Yahoo Finance/GDELT news rows and local sentiment estimates."""
    capped_limit = max(0, min(int(limit), 25))
    rows = _get_news(ticker, limit=capped_limit)
    return sanitize_for_json(
        {
            "ticker": ticker.strip().upper(),
            "limit": capped_limit,
            "count": len(rows),
            "providers": sorted({str(row.get("provider")) for row in rows if row.get("provider")}),
            "articles": rows,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


@mcp.tool("get_latest_technical_indicators")
def get_latest_technical_indicators(
    ticker: str,
    period: str | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    """Return the latest calculated technical indicator snapshot for a ticker."""
    effective_period = period or settings.default_period
    effective_interval = interval or settings.default_interval
    prices = _get_price_history(ticker, period=effective_period, interval=effective_interval)
    indicators = latest_indicators(prices)
    chart_patterns = _detect_chart_patterns(prices)
    return sanitize_for_json(
        {
            "ticker": ticker.strip().upper(),
            "period": effective_period,
            "interval": effective_interval,
            "price_provider": prices.attrs.get("provider") if not prices.empty else None,
            "data_points": int(len(prices)),
            "indicators": indicators,
            "chart_patterns": chart_patterns,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": [] if indicators else ["No technical indicator data could be calculated."],
        }
    )


@mcp.tool("get_chart_patterns")
def get_chart_patterns(
    ticker: str,
    period: str | None = None,
    interval: str | None = None,
    lookback: int = 120,
) -> dict[str, Any]:
    """Return deterministic OHLCV chart-pattern detections for a ticker."""
    effective_period = period or settings.default_period
    effective_interval = interval or settings.default_interval
    prices = _get_price_history(ticker, period=effective_period, interval=effective_interval)
    chart_patterns = _detect_chart_patterns(prices, lookback=lookback)
    return sanitize_for_json(
        {
            "ticker": ticker.strip().upper(),
            "period": effective_period,
            "interval": effective_interval,
            "lookback": max(40, min(int(lookback or 120), 300)),
            "price_provider": prices.attrs.get("provider") if not prices.empty else None,
            "data_points": int(len(prices)),
            "chart_patterns": chart_patterns,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": chart_patterns.get("warnings", []),
        }
    )


@mcp.tool("get_analyst_insights")
def get_analyst_insights(ticker: str, max_rows: int = 20) -> dict[str, Any]:
    """Return analyst consensus, price targets, recommendation summary, and rating changes."""
    return _get_analyst_insights(ticker, max_rows=max_rows)


@mcp.tool("get_stock_events")
def get_stock_events(ticker: str, max_rows: int = 20) -> dict[str, Any]:
    """Return corporate calendar, dividends, splits, actions, and earnings history/dates."""
    return _get_stock_events(ticker, max_rows=max_rows)


@mcp.tool("discover_market_themes")
def discover_market_themes(
    market: str = "india",
    days: int = 7,
    limit_per_theme: int = 6,
    max_themes: int = 8,
    include_portfolio: bool = True,
    include_watchlists: bool = True,
) -> dict[str, Any]:
    """Scan latest broad news and map likely beneficiary sectors/stocks to check."""
    return _discover_market_themes(
        market=market,
        days=days,
        limit_per_theme=limit_per_theme,
        max_themes=max_themes,
        include_portfolio=include_portfolio,
        include_watchlists=include_watchlists,
    )


@mcp.tool("list_stock_universe")
def list_stock_universe(
    universe: str = "broad",
    sector: str | None = None,
    industry: str | None = None,
    limit: int = 500,
    refresh_universe: bool = False,
) -> dict[str, Any]:
    """Return broad NSE stock universe metadata grouped by sector and basic industry."""
    return sanitize_for_json(
        _list_stock_universe(
            universe=universe,
            sector=sector,
            industry=industry,
            limit=limit,
            refresh=refresh_universe,
        )
    )


@mcp.tool("list_sector_constituents")
def list_sector_constituents(
    universe: str = "broad",
    sector: str | None = None,
    refresh_universe: bool = False,
) -> dict[str, Any]:
    """Return the exact sector -> industry -> stock constituents used by broad analytics."""
    return sanitize_for_json(
        _list_sector_constituents(
            universe=universe,
            sector=sector,
            refresh=refresh_universe,
        )
    )


@mcp.tool("refresh_stock_universe")
def refresh_stock_universe(index_name: str = "NIFTY TOTAL MARKET") -> dict[str, Any]:
    """Refresh the local broad NSE universe CSV from free NSE index constituent data."""
    return sanitize_for_json(_refresh_stock_universe(index_name=index_name))


@mcp.tool("refresh_full_stock_universe")
def refresh_full_stock_universe(max_symbols: int | None = None, symbols: list[str] | None = None) -> dict[str, Any]:
    """Refresh the full NSE equity universe CSV from public NSE equity master and quote metadata."""
    return sanitize_for_json(_refresh_full_stock_universe(max_symbols=max_symbols, symbols=symbols))


@mcp.tool("refresh_bse_stock_universe")
def refresh_bse_stock_universe() -> dict[str, Any]:
    """Refresh the full BSE equity universe CSV from Dhan's free public instrument master."""
    return sanitize_for_json(_refresh_bse_stock_universe())


@mcp.tool("refresh_india_stock_universe")
def refresh_india_stock_universe() -> dict[str, Any]:
    """Refresh a combined NSE+BSE active equity universe CSV from free public instrument data."""
    return sanitize_for_json(_refresh_india_stock_universe())


@mcp.tool("refresh_latest_exchange_eod_cache")
def refresh_latest_exchange_eod_cache(
    universe: str = "full_nse",
    max_universe_stocks: int | None = None,
    interval: str = "1d",
) -> dict[str, Any]:
    """Fetch latest official NSE/BSE bhavcopy rows and store them in the local SQLite cache."""
    universe_df = _load_stock_universe(universe=universe, max_stocks=max_universe_stocks)
    tickers = list(universe_df["ticker"]) if not universe_df.empty else []
    result = _refresh_latest_exchange_eod_cache(tickers, interval=interval)
    result["universe"] = universe
    result["universe_stock_count"] = int(len(universe_df))
    return sanitize_for_json(result)


@mcp.tool("run_daily_market_data_refresh")
def run_daily_market_data_refresh(
    universe: str = "full_nse",
    period: str = "2y",
    interval: str = "1d",
    refresh_universes: bool = True,
    refresh_full_nse_universe: bool = True,
    refresh_broad_universe: bool = True,
    refresh_bse_universe: bool = False,
    refresh_india_universe: bool = False,
    warm_price_cache: bool = True,
    refresh_exchange_eod: bool = True,
    max_universe_symbols: int | None = None,
    max_price_symbols: int | None = None,
    chunk_size: int = 80,
    retry_attempts: int = 2,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Run the daily public NSE/BSE refresh job and return the JSON report."""
    return sanitize_for_json(
        _run_daily_market_data_refresh(
            refresh_universes=refresh_universes,
            refresh_broad_universe=refresh_broad_universe,
            refresh_full_nse_universe=refresh_full_nse_universe,
            refresh_bse_universe=refresh_bse_universe,
            refresh_india_universe=refresh_india_universe,
            warm_price_cache=warm_price_cache,
            refresh_exchange_eod=refresh_exchange_eod,
            warm_universe=universe,
            period=period,
            interval=interval,
            max_universe_symbols=max_universe_symbols,
            max_price_symbols=max_price_symbols,
            chunk_size=chunk_size,
            retry_attempts=retry_attempts,
            force_refresh_prices=force_refresh_prices,
        )
    )


@mcp.tool("get_price_cache_status")
def get_price_cache_status(
    universe: str = "full_nse",
    interval: str = "1d",
    max_universe_stocks: int | None = None,
) -> dict[str, Any]:
    """Return SQLite OHLCV cache coverage for the selected stock universe."""
    universe_df = _load_stock_universe(universe=universe, max_stocks=max_universe_stocks)
    tickers = list(universe_df["ticker"]) if not universe_df.empty else []
    return sanitize_for_json(_get_price_cache_status(tickers=tickers, interval=interval))


@mcp.tool("warm_price_history_cache")
def warm_price_history_cache(
    universe: str = "full_nse",
    period: str = "2y",
    interval: str = "1d",
    max_universe_stocks: int | None = None,
    chunk_size: int = 80,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Fetch and store OHLCV candles for a universe so later analytics avoid repeated provider downloads."""
    universe_df = _load_stock_universe(universe=universe, max_stocks=max_universe_stocks)
    tickers = list(universe_df["ticker"]) if not universe_df.empty else []
    result = _warm_price_history_cache(
        tickers,
        period=period,
        interval=interval,
        chunk_size=chunk_size,
        force_refresh=force_refresh_prices,
    )
    result["universe"] = universe
    result["universe_stock_count"] = int(len(universe_df))
    return sanitize_for_json(result)


@mcp.tool("list_sector_definitions")
def list_sector_definitions() -> dict[str, Any]:
    """Return configured sector indices and stock universes used for sector rotation."""
    return _list_sector_definitions()


@mcp.tool("get_sector_rotation")
def get_sector_rotation(
    period: str = "1y",
    interval: str = "1d",
    max_sectors: int | None = None,
    include_breadth: bool = True,
    max_breadth_stocks: int = 8,
) -> dict[str, Any]:
    """Rank sectors by index movement, relative strength, trend, acceleration, and breadth."""
    return _get_sector_rotation(
        period=period,
        interval=interval,
        max_sectors=max_sectors,
        include_breadth=include_breadth,
        max_breadth_stocks=max_breadth_stocks,
    )


@mcp.tool("rank_sector_stocks")
def rank_sector_stocks(
    sector: str,
    period: str = "1y",
    interval: str = "1d",
    max_stocks: int = 10,
    include_fundamentals: bool = True,
) -> dict[str, Any]:
    """Rank stock candidates inside a sector by relative strength, trend, pattern, volume, and risk."""
    return _rank_sector_stocks(
        sector,
        period=period,
        interval=interval,
        max_stocks=max_stocks,
        include_fundamentals=include_fundamentals,
    )


@mcp.tool("discover_sector_opportunities")
def discover_sector_opportunities(
    period: str = "1y",
    interval: str = "1d",
    top_sectors: int = 3,
    stocks_per_sector: int = 5,
    include_fundamentals: bool = True,
) -> dict[str, Any]:
    """Find leading/emerging sectors and the strongest stock setups inside them."""
    return _discover_sector_opportunities(
        period=period,
        interval=interval,
        top_sectors=top_sectors,
        stocks_per_sector=stocks_per_sector,
        include_fundamentals=include_fundamentals,
    )


@mcp.tool("run_sector_rotation_workflow")
def run_sector_rotation_workflow(
    period: str = "1y",
    interval: str = "1d",
    auto_period: bool = False,
    max_sectors: int = 10,
    max_breadth_stocks: int = 6,
    stocks_per_sector: int = 6,
    include_fundamentals: bool = True,
    selected_sector: str | None = None,
) -> dict[str, Any]:
    """Run a fresh sector-rotation workflow that chains sector ranking and stock ranking."""
    return _run_sector_rotation_workflow(
        period=period,
        interval=interval,
        auto_period=auto_period,
        max_sectors=max_sectors,
        max_breadth_stocks=max_breadth_stocks,
        stocks_per_sector=stocks_per_sector,
        include_fundamentals=include_fundamentals,
        selected_sector=selected_sector,
    )


@mcp.tool("get_sector_analytics")
def get_sector_analytics(
    mode: str = "near_52w_high",
    period: str = "1y",
    interval: str = "1d",
    ma_period: int = 200,
    ma_type: str | None = None,
    rs_cutoff: float = 80,
    near_high_pct: float = 5,
    selected_sector: str | None = None,
    max_stocks: int = 12,
    universe: str = "broad",
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
    max_universe_stocks: int | None = None,
) -> dict[str, Any]:
    """Run sector breadth analytics with MA, RS, or near-52w-high filters and drill-down rows."""
    return _get_sector_analytics(
        mode=mode,
        period=period,
        interval=interval,
        ma_period=ma_period,
        ma_type=ma_type,
        rs_cutoff=rs_cutoff,
        near_high_pct=near_high_pct,
        selected_sector=selected_sector,
        max_stocks=max_stocks,
        universe=universe,
        refresh_universe=refresh_universe,
        force_refresh_prices=force_refresh_prices,
        max_universe_stocks=max_universe_stocks,
    )


@mcp.tool("list_industry_definitions")
def list_industry_definitions(universe: str = "broad", refresh_universe: bool = False) -> dict[str, Any]:
    """Return configured industry groups and stock universes."""
    return _list_industry_definitions(universe=universe, refresh_universe=refresh_universe)


@mcp.tool("get_industry_analytics")
def get_industry_analytics(
    period: str = "1y",
    interval: str = "1d",
    min_stocks: int = 3,
    weighting: str = "equal",
    include_fundamentals: bool = False,
    universe: str = "broad",
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
    max_universe_stocks: int | None = None,
) -> dict[str, Any]:
    """Rank industries across 1D, 1W, 1M, and 3M movement for top-down analysis."""
    return _get_industry_analytics(
        period=period,
        interval=interval,
        min_stocks=min_stocks,
        weighting=weighting,
        include_fundamentals=include_fundamentals,
        universe=universe,
        refresh_universe=refresh_universe,
        force_refresh_prices=force_refresh_prices,
        max_universe_stocks=max_universe_stocks,
    )


@mcp.tool("rank_industry_stocks")
def rank_industry_stocks(
    industry: str,
    period: str = "1y",
    interval: str = "1d",
    max_stocks: int = 10,
    include_fundamentals: bool = True,
    universe: str = "broad",
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Rank stocks inside one industry by relative strength, trend, pattern, volume, fundamentals, and risk."""
    return _rank_industry_stocks(
        industry,
        period=period,
        interval=interval,
        max_stocks=max_stocks,
        include_fundamentals=include_fundamentals,
        universe=universe,
        refresh_universe=refresh_universe,
        force_refresh_prices=force_refresh_prices,
    )


@mcp.tool("get_market_indices")
def get_market_indices(
    period: str = "1y",
    interval: str = "1d",
    max_indices: int | None = None,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Return broad and sector index performance, trend, and RS metrics."""
    return _get_market_indices(period=period, interval=interval, max_indices=max_indices, force_refresh_prices=force_refresh_prices)


@mcp.tool("get_market_breadth")
def get_market_breadth(
    period: str = "1y",
    interval: str = "1d",
    max_stocks: int | None = None,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Return market-health breadth across configured NSE stock universe."""
    return _get_market_breadth(period=period, interval=interval, max_stocks=max_stocks, force_refresh_prices=force_refresh_prices)


@mcp.tool("get_top_gainers")
def get_top_gainers(
    period: str = "1y",
    interval: str = "1d",
    return_window: str = "1d",
    min_return_pct: float = 5.0,
    market_cap_min: float = 1000.0,
    min_industry_stocks: int = 3,
    max_rows: int = 50,
    max_industries: int = 20,
    universe: str = "full_nse",
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
    max_universe_stocks: int | None = None,
) -> dict[str, Any]:
    """Rank top gaining stocks and summarize which industries are driving the move."""
    return _get_top_gainers(
        period=period,
        interval=interval,
        return_window=return_window,
        min_return_pct=min_return_pct,
        market_cap_min=market_cap_min,
        min_industry_stocks=min_industry_stocks,
        max_rows=max_rows,
        max_industries=max_industries,
        universe=universe,
        refresh_universe=refresh_universe,
        force_refresh_prices=force_refresh_prices,
        max_universe_stocks=max_universe_stocks,
    )


@mcp.tool("get_relative_rotation_graph")
def get_relative_rotation_graph(
    period: str = "1y",
    interval: str = "1d",
    benchmark: str = "^NSEI",
    trail_length: int = 5,
    selected_sectors: list[str] | None = None,
    zone: str | None = None,
    max_sectors: int | None = None,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Return a ChartsMaze-style sector RRG with benchmark, zone filters, and rotation trails."""
    return _get_relative_rotation_graph(
        period=period,
        interval=interval,
        benchmark=benchmark,
        trail_length=trail_length,
        selected_sectors=selected_sectors,
        zone=zone,
        max_sectors=max_sectors,
        force_refresh_prices=force_refresh_prices,
    )


@mcp.tool("get_company_intelligence")
def get_company_intelligence(
    ticker: str,
    days: int = 30,
    strategic_days: int = 365,
    limit: int = 12,
) -> dict[str, Any]:
    """Fetch company profile, event evidence, strategic themes, risks, and sector fit."""
    return _get_company_intelligence(ticker, days=days, strategic_days=strategic_days, limit=limit)


@mcp.tool("get_ownership_fundamentals")
def get_ownership_fundamentals(ticker: str) -> dict[str, Any]:
    """Return optional local ownership/governance data for a ticker."""
    return _get_ownership_fundamentals(ticker)


@mcp.tool("get_dhan_profile")
def get_dhan_profile() -> dict[str, Any]:
    """Return Dhan profile/token status. Read-only; requires DHAN_ACCESS_TOKEN."""
    return _get_dhan_profile()


@mcp.tool("get_dhan_holdings")
def get_dhan_holdings() -> list[dict[str, Any]]:
    """Return Dhan demat holdings. Read-only; requires DHAN_ACCESS_TOKEN."""
    return _get_dhan_holdings()


@mcp.tool("get_dhan_positions")
def get_dhan_positions() -> list[dict[str, Any]]:
    """Return Dhan open/carry-forward positions. Read-only; requires DHAN_ACCESS_TOKEN."""
    return _get_dhan_positions()


@mcp.tool("get_dhan_fund_limits")
def get_dhan_fund_limits() -> dict[str, Any]:
    """Return Dhan available fund limits. Read-only; requires DHAN_ACCESS_TOKEN."""
    return _get_dhan_fund_limits()


@mcp.tool("get_dhan_portfolio_stocks")
def get_dhan_portfolio_stocks(include_market_values: bool = True) -> dict[str, Any]:
    """Return normalized stock symbols/tickers currently present in the Dhan portfolio."""
    summary = _get_dhan_portfolio_summary(include_market_values=include_market_values)
    holdings = summary.get("holdings", [])
    stocks = [_compact_dhan_holding(row) for row in holdings]
    return sanitize_for_json(
        {
            "provider": "dhan",
            "count": len(stocks),
            "stocks": stocks,
            "total_cost_value": summary.get("holding_cost_value"),
            "total_current_value": summary.get("holding_current_value"),
            "total_unrealized_pnl": summary.get("holding_unrealized_pnl"),
            "warnings": summary.get("warnings", []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


@mcp.tool("get_dhan_portfolio_summary")
def get_dhan_portfolio_summary(include_market_values: bool = True) -> dict[str, Any]:
    """Return Dhan portfolio exposure/P&L summary. Read-only; requires DHAN_ACCESS_TOKEN."""
    return _get_dhan_portfolio_summary(include_market_values=include_market_values)


@mcp.tool("analyze_dhan_portfolio")
def analyze_dhan_portfolio(
    period: str | None = None,
    interval: str | None = None,
    include_news: bool = True,
    include_intelligence: bool = False,
    include_market_values: bool = True,
    max_holdings: int | None = None,
    include_full_analysis: bool = False,
) -> dict[str, Any]:
    """Fetch Dhan holdings and return portfolio-aware add/hold/reduce action buckets."""
    dhan_summary = _get_dhan_portfolio_summary(include_market_values=include_market_values)
    portfolio_analysis = _analyze_portfolio_holdings(
        dhan_summary.get("holdings", []),
        period=period,
        interval=interval,
        include_news=include_news,
        include_intelligence=include_intelligence,
        max_holdings=max_holdings,
        include_full_analysis=include_full_analysis,
    )
    try:
        fund_limits = _get_dhan_fund_limits()
    except Exception as exc:
        fund_limits = {"error": str(exc)}
    return sanitize_for_json(
        {
            "provider": "dhan",
            "dhan_summary": {
                key: value
                for key, value in dhan_summary.items()
                if key not in {"holdings", "positions"}
            },
            "fund_limits": fund_limits,
            "portfolio_analysis": portfolio_analysis,
            "positions": dhan_summary.get("positions", []),
            "warnings": [
                "Read-only Dhan Trading API workflow. No orders are placed by this MCP server.",
                *dhan_summary.get("warnings", []),
            ],
        }
    )


@mcp.tool("get_price_history")
def get_price_history(
    ticker: str,
    period: str | None = None,
    interval: str | None = None,
    include_indicators: bool = True,
    max_rows: int = 120,
    force_refresh: bool = True,
) -> dict[str, Any]:
    """Return recent OHLCV price rows, optionally enriched with technical indicators."""
    effective_period = period or settings.default_period
    effective_interval = interval or settings.default_interval
    df = _get_price_history(ticker, period=effective_period, interval=effective_interval, force_refresh=force_refresh)
    if include_indicators:
        df = add_indicators(df)
    capped = df.tail(max(0, min(max_rows, 500)))
    return sanitize_for_json(
        {
            "ticker": ticker.strip().upper(),
            "period": effective_period,
            "interval": effective_interval,
            "price_provider": df.attrs.get("provider") if not df.empty else None,
            "row_count": int(len(df)),
            "returned_rows": int(len(capped)),
            "rows": capped.to_dict(orient="records") if not capped.empty else [],
        }
    )


@mcp.tool("get_historical_prices")
def get_historical_prices(
    ticker: str,
    period: str | None = None,
    interval: str | None = None,
    include_indicators: bool = False,
    max_rows: int = 250,
) -> dict[str, Any]:
    """Return historical OHLCV prices. Alias for get_price_history with clearer naming."""
    return get_price_history(
        ticker,
        period=period,
        interval=interval,
        include_indicators=include_indicators,
        max_rows=max_rows,
    )


@mcp.tool("market_snapshot")
def market_snapshot(
    group: str | None = None,
    limit: int = 10,
    period: str | None = None,
    interval: str | None = None,
    include_news: bool = False,
) -> dict[str, Any]:
    """Return a compact watchlist snapshot with top names, signal counts, and provider warnings."""
    rows = _rank_watchlist(
        group,
        limit=limit,
        period=period,
        interval=interval,
        include_news=include_news,
    )
    signal_counts: dict[str, int] = {}
    warnings: dict[str, list[str]] = {}
    for row in rows:
        signal = str(row.get("signal", "Unknown"))
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
        row_warnings = row.get("metadata", {}).get("warnings", [])
        if row_warnings:
            warnings[str(row.get("ticker"))] = row_warnings
    return {
        "group": group or "all",
        "count": len(rows),
        "signal_counts": signal_counts,
        "top": rows[:limit],
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool("generate_daily_report")
def generate_daily_report(group: str | None = None) -> str:
    """Generate a Markdown daily report and return its local path."""
    return _generate_daily_report(group)


def _compact_dhan_holding(row: dict[str, Any]) -> dict[str, Any]:
    symbol = str(row.get("tradingSymbol") or row.get("symbol") or "").strip().upper()
    return {
        "symbol": symbol,
        "exchange": row.get("exchange"),
        "isin": row.get("isin"),
        "security_id": row.get("securityId"),
        "analysis_ticker": row.get("analysis_ticker") or _infer_yfinance_ticker(symbol, row.get("exchange")),
        "quantity": row.get("totalQty"),
        "avg_cost_price": row.get("avgCostPrice"),
        "last_price": row.get("last_price"),
        "cost_value": row.get("cost_value"),
        "current_value": row.get("current_value"),
        "unrealized_pnl": row.get("unrealized_pnl"),
        "unrealized_pnl_percent": row.get("unrealized_pnl_percent"),
        "market_data_provider": row.get("market_data_provider"),
        "market_data_warning": row.get("market_data_warning"),
    }


def _infer_yfinance_ticker(symbol: str, exchange: Any) -> str | None:
    if not symbol:
        return None
    normalized_exchange = str(exchange or "").strip().upper()
    if normalized_exchange == "BSE":
        return f"{symbol}.BO"
    return f"{symbol}.NS"


if __name__ == "__main__":
    mcp.run()

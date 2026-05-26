import asyncio

import pandas as pd

from stock_advisor.mcp import server


def test_mcp_registers_expected_tool_names():
    tools = asyncio.run(server.mcp.list_tools())
    names = {tool.name for tool in tools}

    assert {
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
        "list_sector_definitions",
        "refresh_bse_stock_universe",
        "refresh_full_stock_universe",
        "refresh_india_stock_universe",
        "refresh_stock_universe",
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
    }.issubset(names)


def test_market_snapshot_summarizes_ranked_rows(monkeypatch):
    monkeypatch.setattr(
        server,
        "_rank_watchlist",
        lambda group, limit, period, interval, include_news: [
            {
                "ticker": "AAPL",
                "signal": "Buy Watch",
                "final_score": 70,
                "metadata": {"warnings": []},
            },
            {
                "ticker": "TSLA",
                "signal": "Hold / Monitor",
                "final_score": 55,
                "metadata": {"warnings": ["No news sentiment data returned or news provider is not configured."]},
            },
        ],
    )

    snapshot = server.market_snapshot(group="us", limit=2)

    assert snapshot["count"] == 2
    assert snapshot["signal_counts"] == {"Buy Watch": 1, "Hold / Monitor": 1}
    assert snapshot["warnings"] == {
        "TSLA": ["No news sentiment data returned or news provider is not configured."]
    }


def test_get_price_history_caps_rows_and_serializes_dates(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="D"),
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 300],
        }
    )
    monkeypatch.setattr(server, "_get_price_history", lambda ticker, period, interval: frame)

    result = server.get_price_history("aapl", period="3mo", max_rows=2, include_indicators=False)

    assert result["ticker"] == "AAPL"
    assert result["row_count"] == 3
    assert result["returned_rows"] == 2
    assert result["rows"][0]["date"] == "2026-01-02T00:00:00"


def test_get_historical_prices_uses_price_history(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2, freq="D"),
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        }
    )
    monkeypatch.setattr(server, "_get_price_history", lambda ticker, period, interval: frame)

    result = server.get_historical_prices("aapl", period="1mo", max_rows=1)

    assert result["ticker"] == "AAPL"
    assert result["returned_rows"] == 1
    assert "sma_20" not in result["rows"][0]


def test_get_stock_data_tools(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get_basic_fundamentals",
        lambda ticker: {
            "shortName": "Example Ltd",
            "sector": "Industrials",
            "industry": "Engineering",
            "marketCap": 1000,
            "trailingPE": 20,
            "_sources": ["unit-test"],
        },
    )
    monkeypatch.setattr(
        server,
        "_get_news",
        lambda ticker, limit: [{"title": "Example wins order", "provider": "unit-test"}],
    )

    profile = server.get_stock_profile("example.ns")
    fundamentals = server.get_stock_fundamentals("example.ns")
    news = server.get_stock_news("example.ns", limit=5)

    assert profile["name"] == "Example Ltd"
    assert profile["providers"] == ["unit-test"]
    assert fundamentals["fundamentals"]["trailingPE"] == 20
    assert news["count"] == 1
    assert news["providers"] == ["unit-test"]


def test_get_latest_technical_indicators(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=220, freq="D"),
            "open": [float(i) for i in range(1, 221)],
            "high": [float(i) + 1 for i in range(1, 221)],
            "low": [float(i) - 1 for i in range(1, 221)],
            "close": [float(i) for i in range(1, 221)],
            "volume": [1000 + i for i in range(220)],
        }
    )
    frame.attrs["provider"] = "unit-test"
    monkeypatch.setattr(server, "_get_price_history", lambda ticker, period, interval: frame)

    result = server.get_latest_technical_indicators("trend.ns")

    assert result["ticker"] == "TREND.NS"
    assert result["price_provider"] == "unit-test"
    assert result["data_points"] == 220
    assert result["indicators"]["close"] == 220.0
    assert "sma_200" in result["indicators"]
    assert "chart_patterns" in result


def test_get_chart_patterns_delegates_to_detector(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60, freq="D"),
            "open": [float(i) for i in range(1, 61)],
            "high": [float(i) + 1 for i in range(1, 61)],
            "low": [float(i) - 1 for i in range(1, 61)],
            "close": [float(i) for i in range(1, 61)],
            "volume": [1000 + i for i in range(60)],
        }
    )
    frame.attrs["provider"] = "unit-test"
    monkeypatch.setattr(server, "_get_price_history", lambda ticker, period, interval: frame)
    monkeypatch.setattr(
        server,
        "_detect_chart_patterns",
        lambda prices, lookback=120: {
            "pattern_score": 70,
            "chart_pattern_direction": "bullish",
            "patterns": [{"pattern": "ascending_channel"}],
            "warnings": [],
        },
    )

    result = server.get_chart_patterns("trend.ns", lookback=80)

    assert result["ticker"] == "TREND.NS"
    assert result["price_provider"] == "unit-test"
    assert result["lookback"] == 80
    assert result["chart_patterns"]["patterns"][0]["pattern"] == "ascending_channel"


def test_analyst_and_event_tools_delegate(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get_analyst_insights",
        lambda ticker, max_rows=20: {"ticker": ticker.upper(), "max_rows": max_rows, "consensus": {}},
    )
    monkeypatch.setattr(
        server,
        "_get_stock_events",
        lambda ticker, max_rows=20: {"ticker": ticker.upper(), "max_rows": max_rows, "calendar_events": []},
    )

    analyst = server.get_analyst_insights("aapl", max_rows=5)
    events = server.get_stock_events("aapl", max_rows=7)

    assert analyst == {"ticker": "AAPL", "max_rows": 5, "consensus": {}}
    assert events == {"ticker": "AAPL", "max_rows": 7, "calendar_events": []}


def test_sector_rotation_tools_delegate(monkeypatch):
    monkeypatch.setattr(server, "_list_sector_definitions", lambda: {"auto": {"name": "Nifty Auto"}})
    monkeypatch.setattr(
        server,
        "_get_sector_rotation",
        lambda **kwargs: {"sectors": [{"sector_id": "auto"}], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_rank_sector_stocks",
        lambda sector, **kwargs: {"sector": sector, "stocks": [{"ticker": "AAA.NS"}], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_discover_sector_opportunities",
        lambda **kwargs: {"opportunities": [{"sector": {"sector_id": "auto"}}], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_run_sector_rotation_workflow",
        lambda **kwargs: {"workflow": "sector_rotation_workflow", "cache_used": False, "kwargs": kwargs},
    )

    definitions = server.list_sector_definitions()
    rotation = server.get_sector_rotation(period="6mo", max_sectors=3)
    stocks = server.rank_sector_stocks("auto", max_stocks=4)
    opportunities = server.discover_sector_opportunities(top_sectors=1)
    workflow = server.run_sector_rotation_workflow(period="auto", auto_period=True, selected_sector="auto")

    assert definitions["auto"]["name"] == "Nifty Auto"
    assert rotation["sectors"][0]["sector_id"] == "auto"
    assert rotation["kwargs"]["max_sectors"] == 3
    assert stocks["sector"] == "auto"
    assert stocks["kwargs"]["max_stocks"] == 4
    assert opportunities["opportunities"][0]["sector"]["sector_id"] == "auto"
    assert workflow["workflow"] == "sector_rotation_workflow"
    assert workflow["cache_used"] is False
    assert workflow["kwargs"]["period"] == "auto"
    assert workflow["kwargs"]["auto_period"] is True
    assert workflow["kwargs"]["selected_sector"] == "auto"


def test_market_analytics_tools_delegate(monkeypatch):
    monkeypatch.setattr(server, "_list_industry_definitions", lambda **kwargs: {"infra": {"name": "Infrastructure", "kwargs": kwargs}})
    monkeypatch.setattr(
        server,
        "_get_sector_analytics",
        lambda **kwargs: {"sectors": [{"sector_id": "infra"}], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_get_industry_analytics",
        lambda **kwargs: {"industries": [{"industry_id": "infra"}], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_rank_industry_stocks",
        lambda industry, **kwargs: {"industry": industry, "stocks": [{"ticker": "NCC.NS"}], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_get_market_indices",
        lambda **kwargs: {"indices": [{"id": "nifty_50"}], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_get_market_breadth",
        lambda **kwargs: {"summary": {"stock_count": 25}, "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_get_top_gainers",
        lambda **kwargs: {"stocks": [{"ticker": "AAA.NS"}], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        server,
        "_get_relative_rotation_graph",
        lambda **kwargs: {"points": [{"sector_id": "auto"}], "kwargs": kwargs},
    )

    definitions = server.list_industry_definitions()
    sector_analytics = server.get_sector_analytics(mode="relative_strength", selected_sector="infra", rs_cutoff=75, ma_type="21 EMA")
    industries = server.get_industry_analytics(period="6mo", min_stocks=2, weighting="market_cap")
    stocks = server.rank_industry_stocks("infra", max_stocks=4)
    indices = server.get_market_indices(period="1y", max_indices=3)
    breadth = server.get_market_breadth(max_stocks=25)
    gainers = server.get_top_gainers(return_window="20d", min_return_pct=2.5, max_rows=5)
    rrg = server.get_relative_rotation_graph(period="6mo", trail_length=8, zone="Leading,Improving")

    assert definitions["infra"]["name"] == "Infrastructure"
    assert sector_analytics["sectors"][0]["sector_id"] == "infra"
    assert sector_analytics["kwargs"]["selected_sector"] == "infra"
    assert sector_analytics["kwargs"]["rs_cutoff"] == 75
    assert sector_analytics["kwargs"]["ma_type"] == "21 EMA"
    assert industries["industries"][0]["industry_id"] == "infra"
    assert industries["kwargs"]["weighting"] == "market_cap"
    assert stocks["industry"] == "infra"
    assert stocks["kwargs"]["max_stocks"] == 4
    assert indices["indices"][0]["id"] == "nifty_50"
    assert indices["kwargs"]["max_indices"] == 3
    assert breadth["summary"]["stock_count"] == 25
    assert breadth["kwargs"]["max_stocks"] == 25
    assert gainers["stocks"][0]["ticker"] == "AAA.NS"
    assert gainers["kwargs"]["return_window"] == "20d"
    assert rrg["points"][0]["sector_id"] == "auto"
    assert rrg["kwargs"]["trail_length"] == 8
    assert rrg["kwargs"]["zone"] == "Leading,Improving"


def test_analyze_dhan_portfolio_uses_dhan_holdings(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get_dhan_portfolio_summary",
        lambda include_market_values: {
            "holding_count": 2,
            "position_count": 0,
            "holding_cost_value": 1000,
            "holding_current_value": 1250,
            "holding_unrealized_pnl": 250,
            "exchange_exposure": {"NSE": 1250},
            "warnings": ["broker warning"],
            "positions": [],
            "holdings": [
                {
                    "tradingSymbol": "GOOD",
                    "exchange": "NSE",
                    "analysis_ticker": "GOOD.NS",
                    "totalQty": 2,
                    "cost_value": 1000,
                    "current_value": 1250,
                    "unrealized_pnl": 250,
                    "unrealized_pnl_percent": 25,
                },
                {
                    "tradingSymbol": "LIQUIDCASE",
                    "exchange": "NSE",
                    "analysis_ticker": "LIQUIDCASE.NS",
                    "totalQty": 1,
                    "cost_value": 100,
                    "current_value": 100,
                },
            ],
        },
    )
    monkeypatch.setattr(server, "_get_dhan_fund_limits", lambda: {"availabelBalance": 100})

    def fake_analyze_portfolio_holdings(holdings, **kwargs):
        assert len(holdings) == 2
        assert kwargs["include_news"] is False
        return {"holding_count": 2, "holdings": [{"symbol": "GOOD"}]}

    monkeypatch.setattr(server, "_analyze_portfolio_holdings", fake_analyze_portfolio_holdings)

    result = server.analyze_dhan_portfolio(include_news=False)

    assert result["provider"] == "dhan"
    assert result["fund_limits"] == {"availabelBalance": 100}
    assert result["portfolio_analysis"]["holding_count"] == 2
    assert "holdings" not in result["dhan_summary"]
    assert result["warnings"][1] == "broker warning"


def test_get_dhan_portfolio_stocks_returns_compact_rows(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get_dhan_portfolio_summary",
        lambda include_market_values: {
            "holding_cost_value": 1000,
            "holding_current_value": 1100,
            "holding_unrealized_pnl": 100,
            "warnings": ["warning"],
            "holdings": [
                {
                    "tradingSymbol": "GOOD",
                    "exchange": "NSE",
                    "isin": "INE000A01000",
                    "securityId": "123",
                    "analysis_ticker": "GOOD.NS",
                    "totalQty": 2,
                    "avgCostPrice": 500,
                    "last_price": 550,
                    "cost_value": 1000,
                    "current_value": 1100,
                    "unrealized_pnl": 100,
                    "unrealized_pnl_percent": 10,
                }
            ],
        },
    )

    result = server.get_dhan_portfolio_stocks()

    assert result["count"] == 1
    assert result["stocks"][0]["symbol"] == "GOOD"
    assert result["stocks"][0]["analysis_ticker"] == "GOOD.NS"
    assert result["total_current_value"] == 1100


def test_get_dhan_portfolio_stocks_infers_ticker_without_market_values(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get_dhan_portfolio_summary",
        lambda include_market_values: {
            "holding_cost_value": 1000,
            "holding_current_value": None,
            "holding_unrealized_pnl": None,
            "warnings": [],
            "holdings": [
                {
                    "tradingSymbol": "MGL",
                    "exchange": "ALL",
                    "isin": "INE002S01010",
                    "securityId": "17534",
                    "totalQty": 6,
                    "avgCostPrice": 1288,
                }
            ],
        },
    )

    result = server.get_dhan_portfolio_stocks(include_market_values=False)

    assert result["stocks"][0]["symbol"] == "MGL"
    assert result["stocks"][0]["analysis_ticker"] == "MGL.NS"

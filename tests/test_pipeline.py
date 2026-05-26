import json

import pandas as pd

from stock_advisor.analysis import pipeline


def _price_frame(rows=60):
    dates = pd.date_range("2026-01-01", periods=rows, freq="D")
    close = pd.Series(range(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 1,
            "high": close + 1,
            "low": close - 2,
            "close": close,
            "volume": [1_000_000 + i * 1000 for i in range(rows)],
        }
    )


def test_analyze_stock_returns_json_safe_metadata(monkeypatch):
    monkeypatch.setattr(pipeline, "get_price_history", lambda ticker, period, interval: _price_frame())
    monkeypatch.setattr(
        pipeline,
        "get_basic_fundamentals",
        lambda ticker: {
            "shortName": "Example",
            "trailingPE": 20,
            "debtToEquity": 40,
            "profitMargins": 0.12,
            "revenueGrowth": 0.08,
            "returnOnEquity": 0.16,
            "beta": 1.0,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "get_news",
        lambda ticker: [{"title": "Positive update", "overall_sentiment_score": "0.2"}],
    )

    result = pipeline.analyze_stock(" aapl ", period="3mo", interval="1d")

    assert result["ticker"] == "AAPL"
    assert result["metadata"]["data_points"] == 60
    assert result["metadata"]["has_price_history"] is True
    assert result["metadata"]["has_news"] is True
    assert "chart_patterns" in result
    json.dumps(result, allow_nan=False)


def test_analyze_stock_includes_chart_pattern_signal_fields(monkeypatch):
    monkeypatch.setattr(pipeline, "get_price_history", lambda ticker, period, interval: _price_frame(80))
    monkeypatch.setattr(pipeline, "get_basic_fundamentals", lambda ticker: {})
    monkeypatch.setattr(pipeline, "get_news", lambda ticker: [])
    monkeypatch.setattr(
        pipeline,
        "detect_chart_patterns",
        lambda prices: {
            "patterns": [{"pattern": "double_bottom", "confidence": 72, "direction": "bullish"}],
            "dominant_pattern": {"pattern": "double_bottom", "confidence": 72},
            "pattern_score": 70,
            "chart_pattern_direction": "bullish",
            "warnings": [],
        },
    )

    result = pipeline.analyze_stock("example.ns", include_news=False)

    assert result["chart_patterns"]["pattern_score"] == 70
    assert result["latest_indicators"]["chart_pattern_score"] == 70
    assert result["latest_indicators"]["dominant_chart_pattern"] == "double_bottom"
    assert result["metadata"]["has_chart_patterns"] is True


def test_research_stock_includes_analyst_and_event_sections(monkeypatch):
    monkeypatch.setattr(pipeline, "get_price_history", lambda ticker, period, interval: _price_frame(220))
    monkeypatch.setattr(
        pipeline,
        "get_basic_fundamentals",
        lambda ticker: {
            "shortName": "Example",
            "trailingPE": 20,
            "debtToEquity": 20,
            "profitMargins": 0.12,
            "revenueGrowth": 0.12,
        },
    )
    monkeypatch.setattr(pipeline, "get_news", lambda ticker: [])
    monkeypatch.setattr(
        pipeline,
        "get_company_intelligence",
        lambda *args, **kwargs: {"evidence": [], "providers": [], "warnings": []},
    )
    monkeypatch.setattr(
        pipeline,
        "get_analyst_insights",
        lambda ticker: {"providers": ["unit-test"], "consensus": {"target_upside_percent": 12}},
    )
    monkeypatch.setattr(
        pipeline,
        "get_stock_events",
        lambda ticker: {"providers": ["unit-test"], "calendar_events": [{"event": "earnings_date"}]},
    )

    result = pipeline.research_stock("example.ns")

    assert result["analyst_insights"]["consensus"]["target_upside_percent"] == 12
    assert result["stock_events"]["calendar_events"][0]["event"] == "earnings_date"
    assert result["metadata"]["has_analyst_insights"] is True
    assert result["metadata"]["has_stock_events"] is True


def test_rank_watchlist_sorts_and_applies_limit(monkeypatch):
    monkeypatch.setattr(pipeline, "load_watchlists", lambda: {"us": ["LOW", "HIGH"], "india": []})

    def fake_analyze(ticker, period=None, interval=None, include_news=True, include_intelligence=False, **kwargs):
        return {"ticker": ticker, "final_score": 90 if ticker == "HIGH" else 40, "signal": "Hold"}

    monkeypatch.setattr(pipeline, "analyze_stock", fake_analyze)

    rows = pipeline.rank_watchlist("us", limit=1)

    assert rows == [{"ticker": "HIGH", "final_score": 90, "signal": "Hold"}]


def test_compare_stocks_deduplicates_tickers(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "analyze_stock",
        lambda ticker, period=None, interval=None, include_news=True, include_intelligence=False, **kwargs: {
            "ticker": ticker,
            "final_score": 10 if ticker == "MSFT" else 20,
        },
    )

    result = pipeline.compare_stocks(["msft", "MSFT", "aapl"], include_news=False)

    assert result["count"] == 2
    assert [row["ticker"] for row in result["rows"]] == ["AAPL", "MSFT"]

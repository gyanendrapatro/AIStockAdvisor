from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stock_advisor.data import market_data


def test_price_history_uses_stooq_when_yahoo_empty(monkeypatch):
    fallback = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2, freq="D"),
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000, 1100],
        }
    )
    fallback.attrs["provider"] = "stooq"
    monkeypatch.setattr(market_data, "_get_yahoo_price_history", lambda ticker, period, interval: pd.DataFrame())
    monkeypatch.setattr(market_data, "get_yahoo_chart_price_history", lambda ticker, period, interval: pd.DataFrame())
    monkeypatch.setattr(market_data, "get_stooq_price_history", lambda ticker, period, interval: fallback)

    result = market_data.get_price_history("AAPL", "3mo", "1d")

    assert len(result) == 2
    assert result.attrs["provider"] == "stooq"


def test_price_history_uses_yahoo_chart_before_stooq(monkeypatch):
    chart = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=1, freq="D"),
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
        }
    )
    chart.attrs["provider"] = "yahoo_chart"
    monkeypatch.setattr(market_data, "_get_yahoo_price_history", lambda ticker, period, interval: pd.DataFrame())
    monkeypatch.setattr(market_data, "get_yahoo_chart_price_history", lambda ticker, period, interval: chart)
    monkeypatch.setattr(market_data, "get_stooq_price_history", lambda ticker, period, interval: pd.DataFrame())

    result = market_data.get_price_history("AAPL", "3mo", "1d")

    assert result.attrs["provider"] == "yahoo_chart"


def test_price_history_drops_incomplete_yahoo_rows(monkeypatch):
    yahoo = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2, freq="D"),
            "open": [10.0, None],
            "high": [11.0, None],
            "low": [9.0, None],
            "close": [10.5, None],
            "volume": [1000, 1200],
        }
    )
    yahoo.attrs["provider"] = "yfinance"
    monkeypatch.setattr(market_data, "_get_yahoo_price_history", lambda ticker, period, interval: yahoo)
    monkeypatch.setattr(market_data, "get_yahoo_chart_price_history", lambda ticker, period, interval: pd.DataFrame())
    monkeypatch.setattr(market_data, "get_stooq_price_history", lambda ticker, period, interval: pd.DataFrame())

    result = market_data.get_price_history("ANANTRAJ.NS", "1y", "1d")

    assert len(result) == 1
    assert result.iloc[-1]["close"] == 10.5
    assert result.attrs["provider"] == "yfinance"


def test_price_history_drops_current_daily_candle_before_market_close(monkeypatch):
    yahoo = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-05-22"), pd.Timestamp("2026-05-25")],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000, 1200],
        }
    )
    yahoo.attrs["provider"] = "yfinance"
    monkeypatch.setattr(market_data, "_current_market_datetime", lambda: datetime(2026, 5, 25, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")))
    monkeypatch.setattr(market_data, "_get_yahoo_price_history", lambda ticker, period, interval: yahoo)
    monkeypatch.setattr(market_data, "get_yahoo_chart_price_history", lambda ticker, period, interval: pd.DataFrame())
    monkeypatch.setattr(market_data, "get_stooq_price_history", lambda ticker, period, interval: pd.DataFrame())

    result = market_data.get_price_history("NIFTY.NS", "1y", "1d")

    assert len(result) == 1
    assert str(result.iloc[-1]["date"].date()) == "2026-05-22"


def test_fundamentals_merge_sec_facts(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def info(self):
            return {"shortName": "Example Inc.", "trailingPE": 20}

    monkeypatch.setattr(market_data.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(
        market_data,
        "get_sec_fundamentals",
        lambda ticker: {"_sources": ["sec_edgar"], "sec_cik": "0000000001", "sec_revenue": 1000},
    )
    monkeypatch.setattr(market_data, "get_ownership_fundamentals", lambda ticker: {})

    result = market_data.get_basic_fundamentals("AAPL")

    assert result["shortName"] == "Example Inc."
    assert result["sec_revenue"] == 1000
    assert result["_sources"] == ["yfinance", "sec_edgar"]


def test_fundamentals_merge_local_ownership(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def info(self):
            return {"shortName": "Example Ltd.", "trailingPE": 24}

    monkeypatch.setattr(market_data.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(market_data, "get_sec_fundamentals", lambda ticker: {})
    monkeypatch.setattr(
        market_data,
        "get_ownership_fundamentals",
        lambda ticker: {"_sources": ["local_ownership"], "promoter_holding": 55, "promoter_pledge": 0},
    )

    result = market_data.get_basic_fundamentals("RELIANCE.NS")

    assert result["promoter_holding"] == 55
    assert result["promoter_pledge"] == 0
    assert result["_sources"] == ["yfinance", "local_ownership"]

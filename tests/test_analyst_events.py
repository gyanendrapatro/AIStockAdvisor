import pandas as pd

from stock_advisor.data import analyst_events


def test_get_analyst_insights_normalizes_targets_and_recommendations(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.info = {
                "currentPrice": 100,
                "recommendationKey": "buy",
                "recommendationMean": 2.1,
                "numberOfAnalystOpinions": 12,
                "targetMeanPrice": 120,
                "targetHighPrice": 140,
                "targetLowPrice": 90,
            }

        def get_analyst_price_targets(self):
            return {"mean": 120, "high": 140, "low": 90}

        def get_recommendations_summary(self):
            return pd.DataFrame(
                [{"period": "0m", "strongBuy": 3, "buy": 4, "hold": 2, "sell": 1}]
            )

        def get_upgrades_downgrades(self):
            return pd.DataFrame(
                [{"firm": "Broker", "toGrade": "Buy", "fromGrade": "Hold", "action": "up"}],
                index=pd.to_datetime(["2026-05-01"]),
            )

    monkeypatch.setattr(analyst_events.yf, "Ticker", FakeTicker)

    result = analyst_events.get_analyst_insights("example.ns")

    assert result["ticker"] == "EXAMPLE.NS"
    assert result["consensus"]["recommendation_key"] == "buy"
    assert result["consensus"]["target_upside_percent"] == 20.0
    assert result["recommendation_summary"][0]["strong_buy"] == 3
    assert result["upgrades_downgrades"][0]["index"] == "2026-05-01T00:00:00"
    assert result["providers"] == ["yfinance"]


def test_get_stock_events_normalizes_calendar_actions_and_earnings(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.calendar = {"Earnings Date": pd.Timestamp("2026-06-01")}

        def get_calendar(self):
            return self.calendar

        def get_actions(self):
            return pd.DataFrame(
                [{"Dividends": 2.0, "Stock Splits": 0.0}],
                index=pd.to_datetime(["2026-05-10"]),
            )

        def get_dividends(self):
            return pd.Series([2.0], index=pd.to_datetime(["2026-05-10"]), name="Dividends")

        def get_splits(self):
            return pd.Series([2.0], index=pd.to_datetime(["2026-05-11"]), name="Stock Splits")

        def get_earnings_dates(self, limit=20):
            return pd.DataFrame(
                [{"EPS Estimate": 10.0, "Reported EPS": 11.0, "Surprise(%)": 10.0}],
                index=pd.to_datetime(["2026-06-01"]),
            )

        def get_earnings_history(self):
            return pd.DataFrame([{"quarter": "Q4", "epsActual": 11.0}])

    monkeypatch.setattr(analyst_events.yf, "Ticker", FakeTicker)

    result = analyst_events.get_stock_events("example.ns")

    assert result["calendar_events"][0]["event"] == "earnings_date"
    assert result["calendar_events"][0]["date"] == "2026-06-01T00:00:00"
    assert result["recent_actions"][0]["dividends"] == 2.0
    assert result["recent_dividends"][0]["dividends"] == 2.0
    assert result["recent_splits"][0]["stock_splits"] == 2.0
    assert result["earnings_dates"][0]["eps_estimate"] == 10.0
    assert result["providers"] == ["yfinance"]

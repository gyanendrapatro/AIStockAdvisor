import pandas as pd
import pytest

from stock_advisor.data import dhan


def test_dhan_config_status_requires_token(monkeypatch):
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)

    status = dhan.dhan_config_status()

    assert status["configured"] is False
    assert status["mode"] == "read_only"


def test_dhan_request_requires_token(monkeypatch):
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="DHAN_ACCESS_TOKEN"):
        dhan.get_dhan_holdings()


def test_dhan_portfolio_summary_enriches_holdings(monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token")
    monkeypatch.setattr(
        dhan,
        "get_dhan_holdings",
        lambda: [
            {
                "exchange": "NSE",
                "tradingSymbol": "TCS",
                "securityId": "11536",
                "isin": "INE467B01029",
                "totalQty": 10,
                "avgCostPrice": 3000,
            }
        ],
    )
    monkeypatch.setattr(
        dhan,
        "get_dhan_positions",
        lambda: [{"tradingSymbol": "NIFTY", "realizedProfit": 100, "unrealizedProfit": -25}],
    )
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=1),
            "open": [3200],
            "high": [3200],
            "low": [3200],
            "close": [3200],
            "volume": [1000],
        }
    )
    prices.attrs["provider"] = "test"
    monkeypatch.setattr(dhan, "get_price_history", lambda ticker, period, interval: prices)

    result = dhan.get_dhan_portfolio_summary()

    assert result["holding_count"] == 1
    assert result["position_count"] == 1
    assert result["holding_cost_value"] == 30000
    assert result["holding_current_value"] == 32000
    assert result["holding_unrealized_pnl"] == 2000
    assert result["position_total_pnl"] == 75
    assert result["holdings"][0]["analysis_ticker"] == "TCS.NS"

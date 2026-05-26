from stock_advisor.analysis.portfolio import analyze_portfolio_holdings


def test_analyze_portfolio_holdings_recommends_add_and_cash_buffer():
    holdings = [
        {
            "tradingSymbol": "GOOD",
            "analysis_ticker": "GOOD.NS",
            "totalQty": 2,
            "cost_value": 1000,
            "current_value": 1100,
            "unrealized_pnl": 100,
            "unrealized_pnl_percent": 10,
        },
        {
            "tradingSymbol": "LIQUIDCASE",
            "totalQty": 1,
            "cost_value": 10000,
            "current_value": 10000,
        },
    ]

    def analyzer(ticker, **kwargs):
        assert ticker == "GOOD.NS"
        assert kwargs["include_intelligence"] is True
        return {
            "final_score": 80,
            "signal": "Buy Watch",
            "confidence": 84,
            "technical_score": 90,
            "fundamental_score": 78,
            "news_score": 50,
            "risk_score": 72,
            "momentum_liquidity_score": 76,
            "event_intelligence_score": 65,
            "fundamentals": {"sector": "Industrials", "industry": "Engineering"},
            "latest_indicators": {"close": 110, "sma_50": 100, "sma_200": 95, "rsi_14": 55},
            "reasons": ["Price is above 200-day average"],
            "risks": [],
            "metadata": {"warnings": []},
        }

    result = analyze_portfolio_holdings(holdings, include_intelligence=True, analyzer=analyzer)

    assert result["holding_count"] == 2
    assert result["holdings"][0]["recommendation"]["bucket"] == "ADD_ON_DIPS"
    assert result["holdings"][1]["recommendation"]["bucket"] == "CASH_BUFFER"
    assert result["action_counts"]["ADD_ON_DIPS"] == 1
    assert result["action_counts"]["CASH_BUFFER"] == 1


def test_analyze_portfolio_holdings_reduces_weak_deep_loss():
    holdings = [
        {
            "tradingSymbol": "WEAK",
            "analysis_ticker": "WEAK.NS",
            "totalQty": 1,
            "cost_value": 1000,
            "current_value": 500,
            "unrealized_pnl": -500,
            "unrealized_pnl_percent": -50,
        }
    ]

    def analyzer(ticker, **kwargs):
        return {
            "final_score": 45,
            "signal": "Avoid / Weak",
            "confidence": 70,
            "technical_score": 35,
            "fundamental_score": 45,
            "news_score": 50,
            "risk_score": 50,
            "momentum_liquidity_score": 25,
            "fundamentals": {},
            "latest_indicators": {"close": 50, "sma_200": 80, "rsi_14": 35},
            "reasons": [],
            "risks": ["Large historical drawdown"],
            "metadata": {"warnings": []},
        }

    result = analyze_portfolio_holdings(holdings, analyzer=analyzer)

    recommendation = result["holdings"][0]["recommendation"]
    assert recommendation["bucket"] == "REDUCE_ON_BOUNCE"
    assert recommendation["stance"] == "reduce"

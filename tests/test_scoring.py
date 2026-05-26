from stock_advisor.analysis.scoring import score_stock


def test_score_stock_returns_signal():
    indicators = {"close": 110, "sma_20": 100, "sma_50": 90, "rsi_14": 55, "macd": 2, "macd_signal": 1, "return_20d": 0.05, "volume_ratio": 1.3, "volatility_20d": 0.2, "atr_14": 2}
    fundamentals = {"trailingPE": 25, "debtToEquity": 50, "profitMargins": 0.15, "revenueGrowth": 0.1, "returnOnEquity": 0.18, "beta": 1.1}
    result = score_stock("TEST", indicators, fundamentals, [])
    assert result.final_score > 50
    assert result.signal


def test_score_stock_uses_ownership_governance():
    indicators = {"volatility_20d": 0.2, "atr_14": 2, "close": 100}
    strong = {
        "trailingPE": 25,
        "debtToEquity": 50,
        "profitMargins": 0.15,
        "promoter_holding": 60,
        "promoter_holding_qoq_change": 2,
        "promoter_pledge": 0,
        "fii_holding": 12,
        "dii_holding": 10,
        "beta": 1.0,
    }
    weak = {
        **strong,
        "promoter_holding": 25,
        "promoter_holding_qoq_change": -3,
        "promoter_pledge": 30,
    }

    strong_score = score_stock("TEST", indicators, strong, [])
    weak_score = score_stock("TEST", indicators, weak, [])

    assert strong_score.final_score > weak_score.final_score
    assert "Promoter pledge is high" in weak_score.risks


def test_score_stock_ignores_stale_news_sentiment():
    stale_positive = [
        {
            "title": "Company reports strong expansion",
            "overall_sentiment_score": 0.35,
            "days_old": 120,
        }
    ]

    result = score_stock("TEST", {}, {}, stale_positive)

    assert result.news_score == 50
    assert any("No recent news" in reason for reason in result.reasons)


def test_score_stock_uses_chart_pattern_bias():
    base = {"volatility_20d": 0.2, "atr_14": 2, "close": 100}
    bullish = {
        **base,
        "chart_pattern_score": 76,
        "chart_pattern_direction": "bullish",
        "dominant_chart_pattern": "double_bottom",
    }
    bearish = {
        **base,
        "chart_pattern_score": 24,
        "chart_pattern_direction": "bearish",
        "dominant_chart_pattern": "head_and_shoulders",
    }

    bullish_score = score_stock("TEST", bullish, {}, [])
    bearish_score = score_stock("TEST", bearish, {}, [])

    assert bullish_score.technical_score > bearish_score.technical_score
    assert any("Chart pattern bias is bullish" in reason for reason in bullish_score.reasons)

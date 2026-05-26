from stock_advisor.data import company_intelligence


def test_company_intelligence_classifies_events_and_sector_fit(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def info(self):
            return {
                "shortName": "ANANT RAJ LIMITED",
                "sector": "Real Estate",
                "industry": "Real Estate - Development",
                "longBusinessSummary": "Develops real estate projects and data center infrastructure.",
            }

    evidence = [
        {
            "title": "Anant Raj announces data center expansion project",
            "summary": "The company plans investment in digital infrastructure capacity.",
            "url": "https://example.com/expansion",
            "time_published": "2026-05-13T00:00:00Z",
            "days_old": 1,
            "overall_sentiment_score": 0.2,
            "provider": "gdelt",
        },
        {
            "title": "Anant Raj wins large data center order and enters AI infrastructure services",
            "summary": "The company signed an MoU for a new business vertical and technology platform.",
            "url": "https://example.com/order-innovation",
            "time_published": "2026-05-14T00:00:00Z",
            "days_old": 0,
            "overall_sentiment_score": 0.3,
            "provider": "google_news",
        },
        {
            "title": "Anant Raj court case update",
            "summary": "A regulatory case update mentions Anant Raj.",
            "url": "https://example.com/case",
            "days_old": 5,
            "overall_sentiment_score": -0.2,
            "provider": "gdelt",
        },
    ]

    monkeypatch.setattr(company_intelligence.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(company_intelligence, "get_news", lambda ticker, limit=8: [])
    monkeypatch.setattr(company_intelligence, "search_google_news", lambda query, limit=12: [])
    monkeypatch.setattr(company_intelligence, "search_gdelt_articles", lambda query, limit=12, timespan="30d": evidence)

    result = company_intelligence.get_company_intelligence(
        "ANANTRAJ.NS",
        fundamentals={"revenueGrowth": 0.2, "earningsGrowth": 0.15, "returnOnEquity": 0.12},
        indicators={"close": 520, "sma_50": 500, "sma_200": 550},
    )

    assert result["profile"]["sector"] == "Real Estate"
    assert "data_centers_digital_infrastructure" in result["business_areas"]
    assert result["material_event_counts"]["expansion_capex"] >= 1
    assert result["material_event_counts"]["legal_regulatory"] >= 1
    assert result["material_event_counts"]["order_book"] >= 1
    assert result["material_event_counts"]["new_business_area"] >= 1
    assert result["material_event_counts"]["innovation_product"] >= 1
    assert result["sector_fit"]["score"] > 50
    assert result["positive_catalysts"]
    assert result["order_book_updates"]
    assert result["innovation_updates"]
    assert result["risk_flags"]

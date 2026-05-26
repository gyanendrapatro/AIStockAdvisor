from stock_advisor.data import theme_discovery


def test_discover_market_themes_maps_news_to_portfolio_and_stocks(monkeypatch):
    article = {
        "title": "India power demand drives smart meter and transmission capex",
        "summary": "Grid investment and transformer orders are expected to rise.",
        "url": "https://example.com/power",
        "provider": "google_news",
        "source": "Example",
        "days_old": 1,
        "overall_sentiment_score": 0.2,
        "overall_sentiment_label": "positive",
    }

    monkeypatch.setattr(theme_discovery, "search_google_news", lambda query, limit=6: [article])
    monkeypatch.setattr(theme_discovery, "search_gdelt_articles", lambda query, limit=6, timespan="7d": [])
    monkeypatch.setattr(
        theme_discovery,
        "get_dhan_portfolio_summary",
        lambda include_market_values=False: {
            "holdings": [
                {"tradingSymbol": "TDPOWERSYS"},
                {"tradingSymbol": "GENUSPOWER"},
            ]
        },
    )
    monkeypatch.setattr(theme_discovery, "load_watchlists", lambda: {"india": ["LT.NS", "SBIN.NS"]})

    result = theme_discovery.discover_market_themes(max_themes=3)

    assert result["theme_count"] == 3
    top = result["themes"][0]
    assert top["theme_id"] == "power_grid_electrification"
    assert "Electrical equipment" in top["beneficiary_sectors"]
    assert "TDPOWERSYS.NS" in top["portfolio_matches"]
    assert "GENUSPOWER.NS" in top["stocks_to_check"]
    assert "POWERGRID.NS" in top["new_stocks_to_check"]
    assert all(item["ticker"] not in {"TDPOWERSYS.NS", "GENUSPOWER.NS", "LT.NS"} for item in result["top_new_stocks_to_check"])
    assert result["top_new_stocks_to_check"]


def test_discover_market_themes_includes_geopolitical_defence_news(monkeypatch):
    article = {
        "title": "Border conflict lifts focus on defence drones and missile systems",
        "summary": "Security spending and shipbuilding orders may rise after geopolitical tension.",
        "url": "https://example.com/defence",
        "provider": "gdelt",
        "source": "Example",
        "days_old": 1,
        "overall_sentiment_score": 0.1,
        "overall_sentiment_label": "neutral",
    }

    def fake_google(query, limit=6):
        return [article] if "geopolitics" in query else []

    monkeypatch.setattr(theme_discovery, "search_google_news", fake_google)
    monkeypatch.setattr(theme_discovery, "search_gdelt_articles", lambda query, limit=6, timespan="7d": [])
    monkeypatch.setattr(theme_discovery, "get_dhan_portfolio_summary", lambda include_market_values=False: {"holdings": []})
    monkeypatch.setattr(theme_discovery, "load_watchlists", lambda: {"india": []})

    result = theme_discovery.discover_market_themes(max_themes=2)

    top = result["themes"][0]
    assert top["theme_id"] == "geopolitics_defence_security"
    assert "Defence" in top["beneficiary_sectors"]
    assert "HAL.NS" in top["new_stocks_to_check"]
    assert result["top_new_stocks_to_check"][0]["ticker"] in {"HAL.NS", "BEL.NS", "BDL.NS", "MAZDOCK.NS"}


def test_discover_market_themes_includes_order_book_and_innovation_themes(monkeypatch):
    order_article = {
        "title": "Listed EPC company wins order and reports strong order book backlog",
        "summary": "A new letter of award supports revenue visibility.",
        "url": "https://example.com/order-book",
        "provider": "google_news",
        "source": "Example",
        "days_old": 1,
        "overall_sentiment_score": 0.2,
        "overall_sentiment_label": "positive",
    }
    innovation_article = {
        "title": "Technology company launches new product platform and enters new business segment",
        "summary": "The innovation expands addressable market.",
        "url": "https://example.com/innovation",
        "provider": "google_news",
        "source": "Example",
        "days_old": 1,
        "overall_sentiment_score": 0.2,
        "overall_sentiment_label": "positive",
    }

    def fake_google(query, limit=6):
        if "order book" in query and "letter of award" in query:
            return [order_article]
        if "innovation" in query and "new product" in query:
            return [innovation_article]
        return []

    monkeypatch.setattr(theme_discovery, "search_google_news", fake_google)
    monkeypatch.setattr(theme_discovery, "search_gdelt_articles", lambda query, limit=6, timespan="7d": [])
    monkeypatch.setattr(theme_discovery, "get_dhan_portfolio_summary", lambda include_market_values=False: {"holdings": []})
    monkeypatch.setattr(theme_discovery, "load_watchlists", lambda: {"india": []})

    result = theme_discovery.discover_market_themes(max_themes=4)
    theme_ids = {theme["theme_id"] for theme in result["themes"]}

    assert "order_book_momentum" in theme_ids
    assert "innovation_new_business_models" in theme_ids

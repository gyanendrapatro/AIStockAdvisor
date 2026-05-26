from stock_advisor.data import news


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.payload


def test_get_news_uses_free_yahoo_finance(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_news(self, count=10, tab="news"):
            return [
                {
                    "content": {
                        "title": "AAPL shares surge after strong profit beat",
                        "summary": "Analysts raise outlook after growth update.",
                        "canonicalUrl": {"url": "https://example.com/aapl"},
                        "pubDate": "2026-05-15T09:00:00Z",
                        "provider": {"displayName": "Example News"},
                    }
                }
            ]

    monkeypatch.setattr(news.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(news, "_get_gdelt_news", lambda ticker, limit: [])

    rows = news.get_news("AAPL")

    assert rows[0]["provider"] == "yahoo_finance"
    assert rows[0]["overall_sentiment_label"] == "positive"
    assert rows[0]["url"] == "https://example.com/aapl"


def test_get_news_returns_empty_when_yahoo_unavailable(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_news(self, count=10, tab="news"):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(news.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(news, "_get_gdelt_news", lambda ticker, limit: [])

    assert news.get_news("AAPL") == []


def test_get_news_mixes_yahoo_and_gdelt(monkeypatch):
    monkeypatch.setattr(
        news,
        "_get_yahoo_news",
        lambda ticker, limit: [
            {"title": "Yahoo headline", "url": "https://example.com/yahoo", "provider": "yahoo_finance"}
        ],
    )
    monkeypatch.setattr(
        news,
        "_get_gdelt_news",
        lambda ticker, limit: [
            {"title": "GDELT headline", "url": "https://example.com/gdelt", "provider": "gdelt"}
        ],
    )

    rows = news.get_news("AAPL", limit=2)

    assert [row["provider"] for row in rows] == ["yahoo_finance", "gdelt"]


def test_search_google_news_parses_rss(monkeypatch):
    payload = b"""
<rss><channel>
  <item>
    <title>Anant Raj announces data center expansion</title>
    <link>https://example.com/anant-raj</link>
    <description>Company reports strong growth.</description>
    <pubDate>Tue, 12 May 2026 08:45:00 GMT</pubDate>
    <source>Example News</source>
  </item>
</channel></rss>
"""
    monkeypatch.setattr(news, "urlopen", lambda request, timeout=8: _FakeResponse(payload))

    rows = news.search_google_news("Anant Raj", limit=1)

    assert rows[0]["provider"] == "google_news"
    assert rows[0]["source"] == "Example News"
    assert rows[0]["overall_sentiment_label"] == "positive"
    assert rows[0]["published_at"].startswith("2026-05-12T08:45:00")

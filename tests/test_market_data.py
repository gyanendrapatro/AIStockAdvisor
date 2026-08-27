from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from stock_advisor.data import market_data


def test_get_price_history_is_a_pure_cache_read(monkeypatch):
    """get_price_history must never touch a provider — only fill_price_cache_for_universe does."""
    cached = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2, freq="D"),
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000, 1100],
        }
    )
    cached.attrs["provider"] = "sqlite_cache"
    monkeypatch.setattr(market_data, "_load_cached_price_history", lambda ticker, period, interval: cached)

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("get_price_history must not call yf.download")

    monkeypatch.setattr(market_data.yf, "download", _unexpected_download)

    result = market_data.get_price_history("AAPL", "3mo", "1d")

    assert len(result) == 2
    assert result.attrs["provider"] == "sqlite_cache"


def test_get_price_history_returns_empty_when_nothing_cached(monkeypatch):
    monkeypatch.setattr(market_data, "_load_cached_price_history", lambda ticker, period, interval: pd.DataFrame())
    result = market_data.get_price_history("NEWCO.NS", "3mo", "1d")
    assert result.empty


def test_get_price_histories_reads_only_cached_tickers(monkeypatch):
    cached = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=1, freq="D"),
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
        }
    )

    def _fake_load(ticker, period, interval):
        return cached if ticker == "AAA.NS" else pd.DataFrame()

    monkeypatch.setattr(market_data, "_load_cached_price_history", _fake_load)
    result = market_data.get_price_histories(["AAA.NS", "BBB.NS"], "3mo", "1d")

    assert list(result.keys()) == ["AAA.NS"]


def test_drop_incomplete_price_rows_drops_nan_rows():
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

    result = market_data._drop_incomplete_price_rows(yahoo, "ANANTRAJ.NS", interval="1d")

    assert len(result) == 1
    assert result.iloc[-1]["close"] == 10.5


def test_drop_current_daily_row_before_market_close(monkeypatch):
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
    monkeypatch.setattr(market_data, "_current_market_datetime", lambda: datetime(2026, 5, 25, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")))

    result = market_data._drop_current_daily_row_during_market_hours(yahoo, interval="1d")

    assert len(result) == 1
    assert str(result.iloc[-1]["date"].date()) == "2026-05-22"


def _price_frame(dates: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [10.0] * len(dates),
            "high": [11.0] * len(dates),
            "low": [9.0] * len(dates),
            "close": [10.5] * len(dates),
            "volume": [1000] * len(dates),
        }
    )
    df.attrs["provider"] = "yfinance_batch"
    return df


def test_store_price_history_backward_fill_does_not_regress_latest_date(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    market_data._store_price_history("TESTX.NS", "1d", _price_frame([f"2026-08-{d:02d}" for d in range(10, 21)]))
    market_data._store_price_history("TESTX.NS", "1d", _price_frame([f"2026-07-{d:02d}" for d in range(1, 6)]))

    with market_data._price_cache_connection() as conn:
        meta = conn.execute("SELECT latest_date, earliest_date FROM price_cache_meta").fetchone()
        price_data_date = conn.execute("SELECT date FROM price_data").fetchone()["date"]

    assert meta["latest_date"] == "2026-08-20"
    assert meta["earliest_date"] == "2026-07-01"
    assert price_data_date == "2026-08-20"


def _no_bhavcopy_data(tickers, trade_date):
    """Stand-in for get_exchange_eod_rows_for_date simulating bhavcopy having nothing for any
    requested ticker on any date -- keeps pre-existing Yahoo-only tests isolated from real
    network calls, without changing their original intent (Yahoo does all the work)."""
    return {}


def _mock_bhavcopy_source(monkeypatch, *, row_fn=_no_bhavcopy_data, connectivity_ok=True):
    """Isolate every path that reaches the real NSE/BSE bhavcopy fetchers (the priority pass
    and/or the fallback pass inside fill_price_cache_for_universe) from the network. row_fn
    simulates get_exchange_eod_rows_for_date's ticker-filtered result; connectivity_ok simulates
    bhavcopy_fetch_succeeded_for_date's day-level "was the exchange reachable" signal."""
    monkeypatch.setattr(market_data, "get_exchange_eod_rows_for_date", row_fn)
    monkeypatch.setattr(market_data, "bhavcopy_fetch_succeeded_for_date", lambda trade_date: connectivity_ok)


def test_fill_price_cache_for_universe_classifies_new_ticker_as_full_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    _mock_bhavcopy_source(monkeypatch)
    calls = []
    monkeypatch.setattr(
        market_data,
        "_fetch_and_store_range",
        lambda chunk, start, end, interval: calls.append((tuple(chunk), start, end)) or {},
    )

    result = market_data.fill_price_cache_for_universe(["NEWCO.NS"], retry_attempts=0)

    assert len(calls) == 1
    assert calls[0] == (("NEWCO.NS",), "2024-01-01", (market_data._current_market_datetime().date() + timedelta(days=1)).isoformat())
    assert result["requested_ticker_count"] == 1
    assert result["skipped_up_to_date_count"] == 0


def test_fill_price_cache_for_universe_skips_up_to_date_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    _mock_bhavcopy_source(monkeypatch)
    today = market_data._current_market_datetime().date().isoformat()
    market_data._store_price_history("UPTODATE.NS", "1d", _price_frame(["2024-01-01", today]))

    calls = []
    monkeypatch.setattr(
        market_data,
        "_fetch_and_store_range",
        lambda chunk, start, end, interval: calls.append(chunk) or {},
    )

    result = market_data.fill_price_cache_for_universe(["UPTODATE.NS"], retry_attempts=0)

    assert not calls
    assert result["skipped_up_to_date_count"] == 1


def test_fill_price_cache_for_universe_bhavcopy_fallback_recovers_yahoo_failure(tmp_path, monkeypatch):
    """A ticker Yahoo can't fetch (e.g. an SME/illiquid listing) gets recovered by the bhavcopy
    fallback pass instead of ending up in failed_tickers. The priority pass is disabled here
    (window=0) to isolate the fallback pass specifically -- otherwise the priority pass would
    recover this ticker itself before classification ever runs, since both use the same mocked
    bhavcopy source."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "bhavcopy_priority_window_days", 0)
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: {})

    def _bhavcopy_has_data(tickers, trade_date):
        frame = _price_frame([trade_date.isoformat()])
        frame.attrs["provider"] = "bse_bhavcopy"
        return {ticker: frame for ticker in tickers}

    _mock_bhavcopy_source(monkeypatch, row_fn=_bhavcopy_has_data)

    result = market_data.fill_price_cache_for_universe(["544412.BO"], retry_attempts=0)

    assert result["bhavcopy_fallback_recovered_count"] == 1
    assert result["dormant_marked_count"] == 0
    assert "544412.BO" not in result["failed_tickers"]
    with market_data._price_cache_connection() as conn:
        active = conn.execute(
            "SELECT active FROM instrument_master WHERE ticker = '544412.BO'"
        ).fetchone()["active"]
    assert active in (1, None)


def test_fill_price_cache_for_universe_marks_dormant_when_bhavcopy_also_empty(tmp_path, monkeypatch):
    """A ticker with zero trades on Yahoo AND across the full bhavcopy day-walk gets
    instrument_master.active set to 0 and is removed from failed_tickers."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: {})
    _mock_bhavcopy_source(monkeypatch, connectivity_ok=True)

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('DEAD.BO', 'IN', 1)")
        conn.commit()

    result = market_data.fill_price_cache_for_universe(["DEAD.BO"], retry_attempts=0)

    assert result["dormant_marked_count"] == 1
    assert "DEAD.BO" in result["dormant_tickers"]
    assert "DEAD.BO" not in result["failed_tickers"]
    with market_data._price_cache_connection() as conn:
        active = conn.execute("SELECT active FROM instrument_master WHERE ticker = 'DEAD.BO'").fetchone()["active"]
    assert active == 0


def test_fill_price_cache_for_universe_outage_guard_skips_dormancy_marking(tmp_path, monkeypatch):
    """If the bhavcopy day-walk itself came back mostly empty (simulating an exchange-side
    outage rather than genuine zero-trade tickers), nothing gets marked dormant."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "min_bhavcopy_coverage_pct", 80.0)
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: {})

    def _fake_backfill(tickers, *, start_date, end_date=None, interval="1d", progress_callback=None):
        # Simulates an outage: almost no scanned days actually returned bhavcopy data.
        return {
            "requested_ticker_count": len(tickers),
            "trading_days_scanned": 100,
            "trading_days_with_bhavcopy_data": 5,
            "tickers_with_data_count": 0,
        }

    monkeypatch.setattr(market_data, "backfill_bhavcopy_history", _fake_backfill)

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('OUTAGE.BO', 'IN', 1)")
        conn.commit()

    result = market_data.fill_price_cache_for_universe(["OUTAGE.BO"], retry_attempts=0)

    assert result["dormant_marked_count"] == 0
    assert "OUTAGE.BO" in result["failed_tickers"]
    with market_data._price_cache_connection() as conn:
        active = conn.execute("SELECT active FROM instrument_master WHERE ticker = 'OUTAGE.BO'").fetchone()["active"]
    assert active == 1


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

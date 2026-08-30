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


def _price_frame_with_volume(rows: list[tuple[str, float]], *, provider: str) -> pd.DataFrame:
    """Like _price_frame, but volume is explicit per-row -- for testing the zero-volume filter."""
    dates, volumes = zip(*rows)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [10.0] * len(dates),
            "high": [11.0] * len(dates),
            "low": [9.0] * len(dates),
            "close": [10.5] * len(dates),
            "volume": list(volumes),
        }
    )
    df.attrs["provider"] = provider
    return df


def test_store_price_history_drops_zero_volume_yfinance_rows(tmp_path, monkeypatch):
    """Yahoo regularly carries a thinly-traded security's last known close forward as a synthetic
    zero-volume row instead of returning nothing for a day it didn't trade -- that must not be
    trusted as real data, or a dead ticker's latest_date advances forever."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    frame = _price_frame_with_volume(
        [("2026-08-20", 1000), ("2026-08-21", 0)], provider="yfinance_batch"
    )
    market_data._store_price_history("ZOMBIE.NS", "1d", frame)

    with market_data._price_cache_connection() as conn:
        meta = conn.execute("SELECT latest_date, row_count FROM price_cache_meta").fetchone()
        rows = conn.execute("SELECT date, volume FROM price_history_cache ORDER BY date").fetchall()

    assert meta["latest_date"] == "2026-08-20"
    assert meta["row_count"] == 1
    assert [dict(r) for r in rows] == [{"date": "2026-08-20", "volume": 1000.0}]


def test_store_price_history_keeps_zero_volume_bhavcopy_rows(tmp_path, monkeypatch):
    """The zero-volume filter only applies to Yahoo -- bhavcopy's small zero-volume rate looks
    like genuine auction/corporate-action rows from the authoritative exchange record."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    frame = _price_frame_with_volume(
        [("2026-08-20", 1000), ("2026-08-21", 0)], provider="nse_bhavcopy"
    )
    market_data._store_price_history("REAL.NS", "1d", frame)

    with market_data._price_cache_connection() as conn:
        meta = conn.execute("SELECT latest_date, row_count FROM price_cache_meta").fetchone()

    assert meta["latest_date"] == "2026-08-21"
    assert meta["row_count"] == 2


def test_store_price_history_no_op_when_every_yfinance_row_is_zero_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    frame = _price_frame_with_volume([("2026-08-20", 0), ("2026-08-21", 0)], provider="yfinance_batch")
    market_data._store_price_history("ALLZERO.NS", "1d", frame)

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        meta = conn.execute("SELECT * FROM price_cache_meta").fetchall()

    assert meta == []


def test_clean_zero_volume_yfinance_rows(tmp_path, monkeypatch):
    """One-time cleanup for rows written before the zero-volume filter existed. Inserts rows
    directly via SQL (bypassing _store_price_history's own filter) to simulate pre-existing
    pollution, since new writes can no longer create it."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(
        market_data,
        "_current_market_datetime",
        lambda: datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        partial_id = market_data._ticker_id(conn, "PARTIALFAKE.NS", create=True)
        all_fake_id = market_data._ticker_id(conn, "ALLFAKE.NS", create=True)
        bhav_id = market_data._ticker_id(conn, "REALVOL.NS", create=True)

        rows = [
            # PARTIALFAKE.NS: one real row, then three fake zero-volume yfinance rows on top.
            (partial_id, "1d", "2026-08-10", 10.0, 11.0, 9.0, 10.5, 500, "yfinance_batch", "2026-08-10T10:00:00"),
            (partial_id, "1d", "2026-08-11", 10.5, 10.5, 10.5, 10.5, 0, "yfinance_batch", "2026-08-11T10:00:00"),
            (partial_id, "1d", "2026-08-12", 10.5, 10.5, 10.5, 10.5, 0, "yfinance_batch", "2026-08-12T10:00:00"),
            (partial_id, "1d", "2026-08-13", 10.5, 10.5, 10.5, 10.5, 0, "yfinance_batch", "2026-08-13T10:00:00"),
            # ALLFAKE.NS: nothing but fake rows -- a ticker that never really traded in this cache.
            (all_fake_id, "1d", "2026-08-12", 2.0, 2.0, 2.0, 2.0, 0, "yfinance_batch", "2026-08-12T10:00:00"),
            (all_fake_id, "1d", "2026-08-13", 2.0, 2.0, 2.0, 2.0, 0, "yfinance_batch", "2026-08-13T10:00:00"),
            # REALVOL.NS: bhavcopy-sourced zero-volume row -- must be left alone entirely.
            (bhav_id, "1d", "2026-08-13", 5.0, 5.0, 5.0, 5.0, 0, "nse_bhavcopy", "2026-08-13T10:00:00"),
        ]
        conn.executemany(
            """
            INSERT INTO price_history_cache
            (ticker_id, interval, date, open, high, low, close, volume, provider, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.executemany(
            """
            INSERT INTO price_cache_meta (ticker_id, interval, max_period_days, latest_date, earliest_date, provider, fetched_at, row_count)
            VALUES (?, '1d', 30, ?, ?, ?, ?, ?)
            """,
            [
                (partial_id, "2026-08-13", "2026-08-10", "yfinance_batch", "2026-08-13T10:00:00", 4),
                (all_fake_id, "2026-08-13", "2026-08-12", "yfinance_batch", "2026-08-13T10:00:00", 2),
                (bhav_id, "2026-08-13", "2026-08-13", "nse_bhavcopy", "2026-08-13T10:00:00", 1),
            ],
        )
        conn.executemany(
            "INSERT INTO price_data (ticker_id, date, open, high, low, close, volume, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (partial_id, "2026-08-13", 10.5, 10.5, 10.5, 10.5, 0, "2026-08-13T10:00:00"),
                (all_fake_id, "2026-08-13", 2.0, 2.0, 2.0, 2.0, 0, "2026-08-13T10:00:00"),
                (bhav_id, "2026-08-13", 5.0, 5.0, 5.0, 5.0, 0, "2026-08-13T10:00:00"),
            ],
        )
        conn.commit()

    dry_run_result = market_data.clean_zero_volume_yfinance_rows(dry_run=True)
    assert dry_run_result["rows_matched"] == 5  # 3 from PARTIALFAKE + 2 from ALLFAKE
    assert dry_run_result["rows_deleted"] == 0
    assert dry_run_result["tickers_affected"] == 2

    with market_data._price_cache_connection() as conn:
        # Nothing changed yet.
        still_there = conn.execute(
            "SELECT COUNT(*) FROM price_history_cache WHERE ticker_id = ? AND date = '2026-08-11'", (partial_id,)
        ).fetchone()[0]
    assert still_there == 1

    result = market_data.clean_zero_volume_yfinance_rows(dry_run=False)
    assert result["tickers_affected"] == 2
    assert result["tickers_latest_date_changed"] == 2

    with market_data._price_cache_connection() as conn:
        partial_meta = conn.execute(
            "SELECT latest_date, earliest_date, row_count FROM price_cache_meta WHERE ticker_id = ?", (partial_id,)
        ).fetchone()
        all_fake_meta = conn.execute(
            "SELECT latest_date, earliest_date, row_count FROM price_cache_meta WHERE ticker_id = ?", (all_fake_id,)
        ).fetchone()
        bhav_meta = conn.execute(
            "SELECT latest_date, row_count FROM price_cache_meta WHERE ticker_id = ?", (bhav_id,)
        ).fetchone()
        partial_price_data = conn.execute("SELECT date, close FROM price_data WHERE ticker_id = ?", (partial_id,)).fetchone()
        all_fake_price_data = conn.execute("SELECT date FROM price_data WHERE ticker_id = ?", (all_fake_id,)).fetchone()
        bhav_rows = conn.execute(
            "SELECT COUNT(*) FROM price_history_cache WHERE ticker_id = ?", (bhav_id,)
        ).fetchone()[0]

    assert partial_meta["latest_date"] == "2026-08-10"
    assert partial_meta["row_count"] == 1
    assert partial_price_data["date"] == "2026-08-10"
    assert partial_price_data["close"] == 10.5

    assert all_fake_meta["latest_date"] is None
    assert all_fake_meta["row_count"] == 0
    assert all_fake_price_data is None

    assert bhav_meta["latest_date"] == "2026-08-13"
    assert bhav_meta["row_count"] == 1
    assert bhav_rows == 1  # untouched

    # Re-running after cleanup is a no-op.
    rerun = market_data.clean_zero_volume_yfinance_rows(dry_run=False)
    assert rerun["tickers_affected"] == 0


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
        lambda chunk, start, end, interval: calls.append((tuple(chunk), start, end)) or ({}, 0),
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
        lambda chunk, start, end, interval: calls.append(chunk) or ({}, 0),
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
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))

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


def test_fill_price_cache_for_universe_marks_dormant_when_forward_fetch_stays_stale(tmp_path, monkeypatch):
    """A ticker with a healthy, substantial cached history whose latest_date has been stuck
    longer than forward_fetch_stale_days must be routed through the bhavcopy confirmation even
    though it's only ever "failed" a normal forward_fetch (never full_fetch/backward_fetch) --
    and gets marked dormant if that full day-walk finds nothing newer either."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "forward_fetch_stale_days", 30)
    monkeypatch.setattr(
        market_data,
        "_current_market_datetime",
        lambda: datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))
    _mock_bhavcopy_source(monkeypatch, connectivity_ok=True)  # bhavcopy also finds nothing new

    # Already reached start_date -- only latest_date (60+ days stale) is the problem.
    market_data._store_price_history("STOPPED.NS", "1d", _price_frame(["2024-01-01", "2026-06-01"]))

    result = market_data.fill_price_cache_for_universe(["STOPPED.NS"], retry_attempts=0)

    assert result["dormant_marked_count"] == 1
    assert "STOPPED.NS" in result["dormant_tickers"]
    with market_data._price_cache_connection() as conn:
        active = conn.execute("SELECT active FROM instrument_master WHERE ticker = 'STOPPED.NS'").fetchone()["active"]
    assert active == 0


def test_fill_price_cache_for_universe_leaves_recently_stale_forward_fetch_alone(tmp_path, monkeypatch):
    """A ticker only a few days behind (normal illiquidity, well under forward_fetch_stale_days)
    must NOT be routed through the bhavcopy confirmation or marked dormant -- that's exactly
    what forward_fetch itself is for, and a real trading gap of a few days is not evidence of
    anything having stopped."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "forward_fetch_stale_days", 30)
    monkeypatch.setattr(
        market_data,
        "_current_market_datetime",
        lambda: datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))
    bhavcopy_calls = []
    monkeypatch.setattr(
        market_data,
        "backfill_bhavcopy_history",
        lambda tickers, **kwargs: bhavcopy_calls.append(list(tickers)) or {
            "trading_days_scanned": 0,
            "trading_days_with_bhavcopy_data": 0,
            "tickers_with_data_count": 0,
        },
    )

    market_data._store_price_history("QUIETDAY.NS", "1d", _price_frame(["2024-01-01", "2026-08-20"]))

    result = market_data.fill_price_cache_for_universe(["QUIETDAY.NS"], retry_attempts=0)

    # Exactly one call: the priority pass every ticker always gets. Phase C's confirmation walk
    # (a second call) must never trigger for a ticker only a few days behind.
    assert len(bhavcopy_calls) == 1
    assert result["dormant_marked_count"] == 0


def test_fill_price_cache_for_universe_narrows_bhavcopy_walk_when_purely_forward_stale(tmp_path, monkeypatch):
    """When every Phase C candidate this run is forward-stale-origin, the confirmation day-walk
    should start at the earliest of their own latest_dates, not the full effective_start_date --
    each already has solid history up to its own latest_date, so re-walking all the way back to
    the configured floor is pure waste."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "forward_fetch_stale_days", 30)
    monkeypatch.setattr(market_data.settings, "bhavcopy_priority_window_days", 0)
    monkeypatch.setattr(
        market_data,
        "_current_market_datetime",
        lambda: datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))

    calls = []

    def _fake_backfill(tickers, *, start_date, end_date=None, interval="1d", progress_callback=None):
        calls.append((sorted(tickers), start_date))
        return {"trading_days_scanned": 0, "trading_days_with_bhavcopy_data": 0, "tickers_with_data_count": 0}

    monkeypatch.setattr(market_data, "backfill_bhavcopy_history", _fake_backfill)

    market_data._store_price_history("OLDER.NS", "1d", _price_frame(["2024-01-01", "2026-06-01"]))
    market_data._store_price_history("NEWER.NS", "1d", _price_frame(["2024-01-01", "2026-07-01"]))

    market_data.fill_price_cache_for_universe(["OLDER.NS", "NEWER.NS"], retry_attempts=0)

    assert len(calls) == 1
    tickers, start_date = calls[0]
    assert tickers == ["NEWER.NS", "OLDER.NS"]
    assert start_date == "2026-06-01"  # earliest of the two latest_dates, not "2024-01-01"


def test_fill_price_cache_for_universe_splits_backward_and_forward_stale_into_separate_walks(tmp_path, monkeypatch):
    """A mixed batch (one ticker genuinely needing full history, one forward-stale) must get two
    INDEPENDENT bhavcopy walks, not one combined call over the full range for everyone -- each
    candidate group's own confirmation must not depend on the other's range or connectivity."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "forward_fetch_stale_days", 30)
    monkeypatch.setattr(market_data.settings, "bhavcopy_priority_window_days", 0)
    monkeypatch.setattr(
        market_data,
        "_current_market_datetime",
        lambda: datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))

    calls = []

    def _fake_backfill(tickers, *, start_date, end_date=None, interval="1d", progress_callback=None):
        calls.append((sorted(tickers), start_date))
        return {"trading_days_scanned": 0, "trading_days_with_bhavcopy_data": 0, "tickers_with_data_count": 0}

    monkeypatch.setattr(market_data, "backfill_bhavcopy_history", _fake_backfill)

    # NEWLISTED.NS has no cache at all -> full_fetch -> Yahoo fails -> genuinely backward-origin.
    market_data._store_price_history("STALE.NS", "1d", _price_frame(["2024-01-01", "2026-06-01"]))

    market_data.fill_price_cache_for_universe(["NEWLISTED.NS", "STALE.NS"], retry_attempts=0)

    assert len(calls) == 2
    calls_by_start = {start_date: tickers for tickers, start_date in calls}
    assert calls_by_start["2024-01-01"] == ["NEWLISTED.NS"]  # backward-origin: full range
    assert calls_by_start["2026-06-01"] == ["STALE.NS"]  # forward-stale: narrowed to its own latest_date


def test_fill_price_cache_for_universe_one_pass_low_coverage_does_not_block_the_other(tmp_path, monkeypatch):
    """The core regression test for the bug found live: bundling both candidate groups into one
    walk let one group's connectivity noise silently skip marking for the other group too. With
    independent passes, a low-coverage backward-origin walk must not block a clean forward-stale
    walk's dormancy marking, and vice versa."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "forward_fetch_stale_days", 30)
    monkeypatch.setattr(market_data.settings, "bhavcopy_priority_window_days", 0)
    monkeypatch.setattr(market_data.settings, "min_bhavcopy_coverage_pct", 80.0)
    monkeypatch.setattr(
        market_data,
        "_current_market_datetime",
        lambda: datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))

    def _fake_backfill(tickers, *, start_date, end_date=None, interval="1d", progress_callback=None):
        if start_date == "2024-01-01":
            # Backward-origin walk: simulate a connectivity outage -- low coverage.
            return {"trading_days_scanned": 100, "trading_days_with_bhavcopy_data": 10, "tickers_with_data_count": 0}
        # Forward-stale walk: clean, high coverage.
        return {"trading_days_scanned": 20, "trading_days_with_bhavcopy_data": 20, "tickers_with_data_count": 0}

    monkeypatch.setattr(market_data, "backfill_bhavcopy_history", _fake_backfill)

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('NEWLISTED.NS', 'IN', 1)")
        conn.commit()
    market_data._store_price_history("STOPPED.NS", "1d", _price_frame(["2024-01-01", "2026-06-01"]))

    result = market_data.fill_price_cache_for_universe(["NEWLISTED.NS", "STOPPED.NS"], retry_attempts=0)

    # Forward-stale (STOPPED.NS) got confirmed dormant despite the backward-origin walk's outage.
    assert result["dormant_marked_count"] == 1
    assert "STOPPED.NS" in result["dormant_tickers"]
    # Backward-origin (NEWLISTED.NS) was NOT marked dormant/exhausted -- its own walk had low coverage.
    assert "NEWLISTED.NS" not in result["dormant_tickers"]
    assert result["backfill_exhausted_marked_count"] == 0
    with market_data._price_cache_connection() as conn:
        newlisted_active = conn.execute("SELECT active FROM instrument_master WHERE ticker = 'NEWLISTED.NS'").fetchone()["active"]
        stopped_active = conn.execute("SELECT active FROM instrument_master WHERE ticker = 'STOPPED.NS'").fetchone()["active"]
    assert newlisted_active == 1
    assert stopped_active == 0


def test_fill_price_cache_for_universe_reports_pending_dormancy_check_count(tmp_path, monkeypatch):
    """A ticker only a few days behind (well under forward_fetch_stale_days) should be counted in
    forward_fetch_pending_dormancy_check_count, not marked dormant -- so a recurring
    forward-updated/failed count is visibly a converging population, not stuck forever."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "forward_fetch_stale_days", 30)
    monkeypatch.setattr(
        market_data,
        "_current_market_datetime",
        lambda: datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))
    _mock_bhavcopy_source(monkeypatch)

    market_data._store_price_history("ILLIQUID.NS", "1d", _price_frame(["2024-01-01", "2026-08-20"]))

    result = market_data.fill_price_cache_for_universe(["ILLIQUID.NS"], retry_attempts=0)

    assert result["dormant_marked_count"] == 0
    assert result["forward_fetch_pending_dormancy_check_count"] == 1
    assert result["forward_fetch_stale_days"] == 30


def test_fill_price_cache_for_universe_marks_dormant_when_bhavcopy_also_empty(tmp_path, monkeypatch):
    """A ticker with zero trades on Yahoo AND across the full bhavcopy day-walk gets
    instrument_master.active set to 0 and is removed from failed_tickers."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))
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
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))

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


def test_fill_price_cache_for_universe_marks_backfill_exhausted_when_bhavcopy_confirms_no_earlier_data(tmp_path, monkeypatch):
    """A ticker with existing (shallow) cache whose earliest_date can't be pushed back to
    start_date by either Yahoo or a complete bhavcopy day-walk gets backfill_exhausted_start_date
    recorded, so a future run stops retrying its backward-fill (it already has data, so it's not
    a dormancy case -- just a confirmed historical floor, e.g. a recent IPO)."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data, "_fetch_and_store_range", lambda chunk, start, end, interval: ({}, 0))
    _mock_bhavcopy_source(monkeypatch, connectivity_ok=True)  # bhavcopy finds nothing earlier either

    market_data._store_price_history("RECENTIPO.NS", "1d", _price_frame(["2026-08-25", "2026-08-26"]))

    result = market_data.fill_price_cache_for_universe(["RECENTIPO.NS"], retry_attempts=0)

    assert result["backfill_exhausted_marked_count"] == 1
    with market_data._price_cache_connection() as conn:
        row = conn.execute(
            "SELECT backfill_exhausted_start_date FROM price_cache_meta m "
            "JOIN instrument_master im ON im.ticker_id = m.ticker_id WHERE im.ticker = 'RECENTIPO.NS'"
        ).fetchone()
    assert row["backfill_exhausted_start_date"] == "2024-01-01"


def test_fill_price_cache_for_universe_skips_backward_fetch_when_already_exhausted(tmp_path, monkeypatch):
    """Once backfill_exhausted_start_date matches the current start_date, no Yahoo (or bhavcopy
    fallback) attempt is made for that ticker at all -- the whole point of recording it."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")

    today = market_data._current_market_datetime().date().isoformat()
    market_data._store_price_history("EXHAUSTED.NS", "1d", _price_frame(["2026-08-25", today]))
    with market_data._price_cache_connection() as conn:
        conn.execute(
            "UPDATE price_cache_meta SET backfill_exhausted_start_date = ? "
            "WHERE ticker_id = (SELECT ticker_id FROM instrument_master WHERE ticker = 'EXHAUSTED.NS')",
            ("2024-01-01",),
        )
        conn.commit()

    calls = []
    monkeypatch.setattr(
        market_data,
        "_fetch_and_store_range",
        lambda chunk, start, end, interval: calls.append((tuple(chunk), start, end)) or ({}, 0),
    )
    _mock_bhavcopy_source(monkeypatch)

    result = market_data.fill_price_cache_for_universe(["EXHAUSTED.NS"], retry_attempts=0)

    assert not calls  # no Yahoo fetch attempted -- backward need suppressed, already current on forward
    assert result["skipped_up_to_date_count"] == 1


def test_fill_price_cache_for_universe_marks_exhausted_when_yahoo_succeeds_but_stays_short(tmp_path, monkeypatch):
    """A ticker where Yahoo 'succeeds' every single run but only ever returns one stray trading
    day (common for extremely illiquid names) must still be routed through the bhavcopy
    confirmation and marked exhausted -- relying on Yahoo's own success/failure signal alone
    would let it cycle through full_fetch forever, since it never technically fails."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    _mock_bhavcopy_source(monkeypatch, connectivity_ok=True)  # bhavcopy also finds nothing more

    def _sparse_success(chunk, start, end, interval):
        frame = _price_frame(["2026-07-17"])  # one stray day, far short of start_date
        fetched = {}
        live_metrics_count = 0
        for ticker in chunk:
            if market_data._store_price_history(ticker, interval, frame):
                live_metrics_count += 1
            fetched[ticker] = frame
        return fetched, live_metrics_count

    monkeypatch.setattr(market_data, "_fetch_and_store_range", _sparse_success)

    result = market_data.fill_price_cache_for_universe(["SPARSE.NS"], retry_attempts=0)

    assert result["full_fetch_count"] == 1  # Yahoo "succeeded" every time, per this mock
    assert result["backfill_exhausted_marked_count"] == 1
    with market_data._price_cache_connection() as conn:
        row = conn.execute(
            "SELECT backfill_exhausted_start_date FROM price_cache_meta m "
            "JOIN instrument_master im ON im.ticker_id = m.ticker_id WHERE im.ticker = 'SPARSE.NS'"
        ).fetchone()
    assert row["backfill_exhausted_start_date"] == "2024-01-01"


def test_fill_price_cache_for_universe_rechecks_when_start_date_moves_earlier(tmp_path, monkeypatch):
    """A stale exhausted-marker (recorded for an older, later start_date) must not suppress a
    fresh backward-fill attempt once the configured floor moves earlier than what was checked."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2020-01-01")

    today = market_data._current_market_datetime().date().isoformat()
    market_data._store_price_history("REPONED.NS", "1d", _price_frame(["2024-01-01", today]))
    with market_data._price_cache_connection() as conn:
        # Exhausted for the *old* 2024-01-01 floor, not the new 2020-01-01 one.
        conn.execute(
            "UPDATE price_cache_meta SET backfill_exhausted_start_date = ? "
            "WHERE ticker_id = (SELECT ticker_id FROM instrument_master WHERE ticker = 'REPONED.NS')",
            ("2024-01-01",),
        )
        conn.commit()

    calls = []
    monkeypatch.setattr(
        market_data,
        "_fetch_and_store_range",
        lambda chunk, start, end, interval: calls.append((tuple(chunk), start, end)) or ({}, 0),
    )
    _mock_bhavcopy_source(monkeypatch)

    market_data.fill_price_cache_for_universe(["REPONED.NS"], retry_attempts=0)

    assert calls  # the earlier floor must trigger a real backward-fill attempt, marker or not


def test_store_shares_outstanding_and_recompute_market_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    # Seed a cached close price the way _store_price_history normally would.
    market_data._store_price_history("TESTX.NS", "1d", _price_frame(["2026-08-26"]))

    assert market_data.store_shares_outstanding("TESTX.NS", 1_000_000.0, source="nse_xbrl") is True
    result = market_data.recompute_market_cap_from_shares_outstanding()
    assert result["recomputed_ticker_count"] == 1

    with market_data._price_cache_connection() as conn:
        row = conn.execute(
            "SELECT market_cap, date FROM market_cap mc JOIN instrument_master im ON im.ticker_id = mc.ticker_id "
            "WHERE im.ticker = 'TESTX.NS'"
        ).fetchone()
    assert row["market_cap"] == 1_000_000.0 * 10.5  # _price_frame's close is 10.5
    assert row["date"] == "2026-08-26"


def test_store_shares_outstanding_rejects_non_positive_values():
    assert market_data.store_shares_outstanding("TESTY.NS", 0, source="nse_xbrl") is False
    assert market_data.store_shares_outstanding("TESTY.NS", -5, source="nse_xbrl") is False


def test_store_security_classification_upserts(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    market_data.store_security_classification(
        "TESTIICS.NS",
        macro_sector_code="IN03", macro_sector="Energy",
        sector_code="IN0301", sector="Oil, Gas & Consumable Fuels",
        industry_code="IN030103", industry="Petroleum Products",
        basic_industry_code="IN030103001", basic_industry="Refineries & Marketing",
        source=market_data.IICS_CLASSIFICATION_SOURCE,
    )
    with market_data._price_cache_connection() as conn:
        row = conn.execute(
            "SELECT sc.* FROM security_classification sc JOIN instrument_master im ON im.ticker_id = sc.ticker_id "
            "WHERE im.ticker = 'TESTIICS.NS'"
        ).fetchone()
    assert row["macro_sector"] == "Energy"
    assert row["basic_industry_code"] == "IN030103001"
    assert row["classification_source"] == market_data.IICS_CLASSIFICATION_SOURCE

    # Upsert: a second call with a changed value updates in place, not a duplicate row.
    market_data.store_security_classification(
        "TESTIICS.NS",
        macro_sector_code="IN03", macro_sector="Energy",
        sector_code="IN0301", sector="Oil, Gas & Consumable Fuels",
        industry_code="IN030103", industry="Petroleum Products",
        basic_industry_code="IN030103002", basic_industry="Other Refining",
        source=market_data.IICS_CLASSIFICATION_SOURCE,
    )
    with market_data._price_cache_connection() as conn:
        rows = conn.execute(
            "SELECT sc.basic_industry_code FROM security_classification sc JOIN instrument_master im ON im.ticker_id = sc.ticker_id "
            "WHERE im.ticker = 'TESTIICS.NS'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["basic_industry_code"] == "IN030103002"


def test_load_iics_classification_seed_from_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    seed_path = tmp_path / "iics_classification.csv"
    seed_path.write_text(
        "ticker,macro_sector_code,macro_sector,sector_code,sector,industry_code,industry,"
        "basic_industry_code,basic_industry,synced_at\n"
        "SEEDCO.NS,IN03,Energy,IN0301,\"Oil, Gas & Consumable Fuels\",IN030103,Petroleum Products,"
        "IN030103001,Refineries & Marketing,2026-01-01T00:00:00+00:00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(market_data, "_IICS_SEED_CSV_PATH", seed_path)

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)

    with market_data._price_cache_connection() as conn:
        row = conn.execute(
            "SELECT sc.* FROM security_classification sc JOIN instrument_master im ON im.ticker_id = sc.ticker_id "
            "WHERE im.ticker = 'SEEDCO.NS'"
        ).fetchone()
    assert row["macro_sector"] == "Energy"
    assert row["sector"] == "Oil, Gas & Consumable Fuels"
    assert row["classification_source"] == market_data.IICS_CLASSIFICATION_SOURCE

    # Already seeded -- opening another connection (re-running _ensure_price_cache_schema) must
    # not re-read the file or error even if the file were to disappear.
    seed_path.unlink()
    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)  # no error, no-op


def test_sync_instrument_master_protects_iics_classification_from_overwrite(tmp_path, monkeypatch):
    """The regression test for the exact clobbering risk found while reviewing the existing
    table: a routine CSV-sourced universe sync must never downgrade a ticker the IICS backfill
    already resolved back to Unclassified."""
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data, "load_stock_universe", lambda universe: pd.DataFrame(columns=market_data.UNIVERSE_COLUMNS))

    csv_row = {
        "ticker": "TESTZ.NS", "symbol": "TESTZ", "name": "Test Z", "isin": "INETESTZ01",
        "source": "test", "active": True, "series": "EQ",
        "exchange": "NSE", "security_id": "1", "nse_ticker": "TESTZ.NS", "bse_ticker": None,
        "nse_security_id": "1", "bse_security_id": None,
        "sector": "Unclassified", "industry": "Unclassified", "basic_industry": "Unclassified",
        "index_name": "NSE FULL EQUITY", "classification_source": None, "data_quality": None,
    }
    monkeypatch.setattr(market_data, "build_stock_master_frame", lambda universes=None: pd.DataFrame([csv_row]))

    # First sync: no IICS data yet -> the CSV-sourced (Unclassified) values apply normally.
    market_data.sync_instrument_master()
    with market_data._price_cache_connection() as conn:
        row = conn.execute(
            "SELECT sc.sector FROM security_classification sc JOIN instrument_master im ON im.ticker_id = sc.ticker_id "
            "WHERE im.ticker = 'TESTZ.NS'"
        ).fetchone()
    assert row["sector"] == "Unclassified"

    # The IICS backfill resolves it for real.
    market_data.store_security_classification(
        "TESTZ.NS",
        macro_sector_code="IN03", macro_sector="Energy",
        sector_code="IN0301", sector="Oil, Gas & Consumable Fuels",
        industry_code="IN030103", industry="Petroleum Products",
        basic_industry_code="IN030103001", basic_industry="Refineries & Marketing",
        source=market_data.IICS_CLASSIFICATION_SOURCE,
    )

    # A routine CSV-sourced sync must NOT clobber it back to Unclassified.
    market_data.sync_instrument_master()
    with market_data._price_cache_connection() as conn:
        row = conn.execute(
            "SELECT sc.sector, sc.classification_source FROM security_classification sc "
            "JOIN instrument_master im ON im.ticker_id = sc.ticker_id WHERE im.ticker = 'TESTZ.NS'"
        ).fetchone()
    assert row["sector"] == "Oil, Gas & Consumable Fuels"
    assert row["classification_source"] == market_data.IICS_CLASSIFICATION_SOURCE


def test_list_tickers_missing_shares_outstanding(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('HASSHARES.NS', 'IN', 1)")
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('NOSHARES.NS', 'IN', 1)")
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('DORMANT.NS', 'IN', 0)")
        conn.commit()
    market_data.store_shares_outstanding("HASSHARES.NS", 500_000.0, source="nse_xbrl")

    missing = market_data.list_tickers_missing_shares_outstanding()

    assert "NOSHARES.NS" in missing
    assert "HASSHARES.NS" not in missing
    assert "DORMANT.NS" not in missing  # active_only=True default excludes it


def test_list_tickers_missing_shares_outstanding_upgrade_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('SCREENERSRC.NS', 'IN', 1)")
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('NSESRC.NS', 'IN', 1)")
        conn.commit()
    market_data.store_shares_outstanding("SCREENERSRC.NS", 100_000.0, source="screener_derived")
    market_data.store_shares_outstanding("NSESRC.NS", 200_000.0, source="nse_xbrl")

    default_missing = market_data.list_tickers_missing_shares_outstanding()
    upgrade_candidates = market_data.list_tickers_missing_shares_outstanding(include_screener_sourced=True)

    # Default mode: both already have a value, so neither is "missing".
    assert "SCREENERSRC.NS" not in default_missing
    assert "NSESRC.NS" not in default_missing
    # Upgrade mode: the screener-sourced one is a candidate for an NSE upgrade; the
    # already-NSE-sourced one must never be re-flagged (that would let a later screener re-run
    # downgrade it).
    assert "SCREENERSRC.NS" in upgrade_candidates
    assert "NSESRC.NS" not in upgrade_candidates


def test_list_instrument_master_tickers_excludes_dormant_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, exchange, active) VALUES ('LIVE.NS', 'IN', 'NSE', 1)")
        conn.execute("INSERT INTO instrument_master (ticker, market, exchange, active) VALUES ('DEAD.NS', 'IN', 'NSE', 0)")
        conn.commit()

    active_only = market_data.list_instrument_master_tickers("full_nse")
    everyone = market_data.list_instrument_master_tickers("full_nse", active_only=False)

    assert "LIVE.NS" in active_only
    assert "DEAD.NS" not in active_only
    assert "DEAD.NS" in everyone


def test_load_stock_universe_from_db_excludes_dormant_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, exchange, active) VALUES ('LIVE.NS', 'IN', 'NSE', 1)")
        conn.execute("INSERT INTO instrument_master (ticker, market, exchange, active) VALUES ('DEAD.NS', 'IN', 'NSE', 0)")
        conn.commit()

    active_df = market_data.load_stock_universe_from_db("full_nse")
    everyone_df = market_data.load_stock_universe_from_db("full_nse", active_only=False)

    assert "LIVE.NS" in set(active_df["ticker"])
    assert "DEAD.NS" not in set(active_df["ticker"])
    assert "DEAD.NS" in set(everyone_df["ticker"])


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


def test_refresh_live_metrics_computes_all_return_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    latest = "2026-08-30"

    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        ticker_id = market_data._ticker_id(conn, "TESTLM.NS", create=True)
        rows = [
            (ticker_id, "1d", "2026-08-30", 200.0, 200.0, 200.0, 200.0, 1000, "yfinance_batch", latest + "T00:00:00"),
            (ticker_id, "1d", "2026-08-23", 180.0, 180.0, 180.0, 180.0, 1000, "yfinance_batch", latest + "T00:00:00"),  # -7d
            (ticker_id, "1d", "2026-07-31", 160.0, 160.0, 160.0, 160.0, 1000, "yfinance_batch", latest + "T00:00:00"),  # -30d
            (ticker_id, "1d", "2026-06-01", 140.0, 140.0, 140.0, 140.0, 1000, "yfinance_batch", latest + "T00:00:00"),  # -90d
            (ticker_id, "1d", "2026-03-03", 120.0, 120.0, 120.0, 120.0, 1000, "yfinance_batch", latest + "T00:00:00"),  # -180d
            (ticker_id, "1d", "2025-08-30", 100.0, 100.0, 100.0, 100.0, 1000, "yfinance_batch", latest + "T00:00:00"),  # -365d
        ]
        conn.executemany(
            """
            INSERT INTO price_history_cache (ticker_id, interval, date, open, high, low, close, volume, provider, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        market_data._refresh_live_metrics(conn, ticker_id, "1d", latest, 200.0)
        conn.commit()

    with market_data._price_cache_connection() as conn:
        row = conn.execute("SELECT * FROM live_metrics WHERE ticker_id = ?", (ticker_id,)).fetchone()

    assert row["return_7d_pct"] == round(100 * (200.0 - 180.0) / 180.0, 4)
    assert row["return_30d_pct"] == round(100 * (200.0 - 160.0) / 160.0, 4)
    assert row["return_90d_pct"] == round(100 * (200.0 - 140.0) / 140.0, 4)
    assert row["return_180d_pct"] == round(100 * (200.0 - 120.0) / 120.0, 4)
    assert row["return_365d_pct"] == round(100 * (200.0 - 100.0) / 100.0, 4)


def test_load_stock_universe_from_db_includes_new_return_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('TESTDB.NS', 'IN', 1)")
        ticker_id = market_data._ticker_id(conn, "TESTDB.NS")
        conn.execute(
            """
            INSERT INTO live_metrics (ticker_id, last_price, return_7d_pct, return_90d_pct, return_180d_pct)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticker_id, 200.0, 11.11, 42.86, 66.67),
        )
        conn.commit()

    df = market_data.load_stock_universe_from_db("all_india")
    row = df[df["ticker"] == "TESTDB.NS"].iloc[0]
    assert row["return_7d_pct"] == 11.11
    assert row["return_90d_pct"] == 42.86
    assert row["return_180d_pct"] == 66.67


def test_get_live_metrics_for_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    with market_data._price_cache_connection() as conn:
        market_data._ensure_price_cache_schema(conn)
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('HASMETRICS.NS', 'IN', 1)")
        conn.execute("INSERT INTO instrument_master (ticker, market, active) VALUES ('NOMETRICS.NS', 'IN', 1)")
        ticker_id = market_data._ticker_id(conn, "HASMETRICS.NS")
        conn.execute(
            "INSERT INTO live_metrics (ticker_id, last_price, return_7d_pct) VALUES (?, ?, ?)",
            (ticker_id, 55.5, 3.2),
        )
        conn.commit()

    result = market_data.get_live_metrics_for_tickers(["HASMETRICS.NS", "NOMETRICS.NS"])

    assert "HASMETRICS.NS" in result
    assert result["HASMETRICS.NS"]["last_price"] == 55.5
    assert result["HASMETRICS.NS"]["return_7d_pct"] == 3.2
    assert "NOMETRICS.NS" not in result  # no live_metrics row -> omitted, not a blank entry


def test_store_price_history_returns_true_when_live_metrics_refreshed(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    result = market_data._store_price_history("FRESH.NS", "1d", _price_frame(["2026-08-30"]))

    assert result is True


def test_store_price_history_returns_false_for_backward_fill_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    market_data._store_price_history("OLDCACHE.NS", "1d", _price_frame(["2026-08-30"]))

    # An older backward-fill batch never reaches the ticker's current latest_date -> no
    # live_metrics refresh, even though the write itself succeeds.
    result = market_data._store_price_history("OLDCACHE.NS", "1d", _price_frame(["2024-01-01"]))

    assert result is False


def test_store_price_history_returns_false_on_swallowed_exception(monkeypatch):
    # No db_path override -- _price_cache_connection will fail against whatever default path
    # resolves in a way that raises inside the try block; simulate directly for determinism.
    monkeypatch.setattr(
        market_data,
        "_price_cache_connection",
        lambda: (_ for _ in ()).throw(RuntimeError("db unavailable")),
    )

    result = market_data._store_price_history("BROKEN.NS", "1d", _price_frame(["2026-08-30"]))

    assert result is False


def test_fetch_and_store_range_returns_live_metrics_count(monkeypatch):
    frame_a = _price_frame(["2026-08-01"])
    frame_b = _price_frame(["2026-08-01"])
    monkeypatch.setattr(market_data.yf, "download", lambda **kwargs: "raw")
    monkeypatch.setattr(
        market_data, "_split_yahoo_batch_history", lambda raw, chunk, interval: {"AAA.NS": frame_a, "BBB.NS": frame_b}
    )
    monkeypatch.setattr(market_data, "_overlay_exchange_latest_rows", lambda frames, interval: frames)
    store_results = {"AAA.NS": True, "BBB.NS": False}
    monkeypatch.setattr(market_data, "_store_price_history", lambda ticker, interval, frame: store_results[ticker])

    fetched, live_metrics_count = market_data._fetch_and_store_range(["AAA.NS", "BBB.NS"], "2026-01-01", "2026-08-02", "1d")

    assert set(fetched) == {"AAA.NS", "BBB.NS"}
    assert live_metrics_count == 1


def test_backfill_bhavcopy_history_returns_live_metrics_refreshed_count(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")

    def _rows_for_date(tickers, trade_date):
        frame = _price_frame([trade_date.isoformat()])
        return {ticker: frame for ticker in tickers}

    _mock_bhavcopy_source(monkeypatch, row_fn=_rows_for_date)
    store_results = iter([True, False])
    monkeypatch.setattr(market_data, "_store_price_history", lambda ticker, interval, frame: next(store_results))

    result = market_data.backfill_bhavcopy_history(["AAA.NS", "BBB.NS"], start_date="2026-08-03", end_date="2026-08-03", interval="1d")

    assert result["live_metrics_refreshed_count"] == 1


def test_fill_price_cache_for_universe_reports_live_metrics_updated_total(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data.settings, "db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(market_data.settings, "price_history_start_date", "2024-01-01")
    monkeypatch.setattr(market_data.settings, "bhavcopy_priority_window_days", 0)
    _mock_bhavcopy_source(monkeypatch)

    def _fake_fetch_and_store(chunk, start, end, interval):
        return {t: _price_frame(["2024-01-01"]) for t in chunk}, len(chunk)

    monkeypatch.setattr(market_data, "_fetch_and_store_range", _fake_fetch_and_store)

    progress_updates = []
    result = market_data.fill_price_cache_for_universe(
        ["AAA.NS", "BBB.NS"], retry_attempts=0, chunk_size=1, progress_callback=lambda u: progress_updates.append(dict(u))
    )

    live_metrics_totals = [u["live_metrics_updated_total"] for u in progress_updates if u.get("phase") == "full_fetch"]
    assert live_metrics_totals == [1, 2]  # cumulative across the two chunk_size=1 chunks
    assert result["live_metrics_updated_count"] == 2

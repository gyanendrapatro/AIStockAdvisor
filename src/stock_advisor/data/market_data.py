from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timedelta
import csv
import logging
import os
import sqlite3
import time as time_module
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from stock_advisor.config.settings import PROJECT_ROOT, settings
from stock_advisor.data.exchange_eod import (
    bhavcopy_fetch_succeeded_for_date,
    get_exchange_eod_rows_for_date,
    get_latest_exchange_eod_rows,
)
from stock_advisor.data.nse_shareholding import (
    build_shareholding_index,
    default_sme_lookback_from_date,
    extract_total_paid_up_shares,
    fetch_shareholding_master,
    nse_session,
)
from stock_advisor.data.screener import (
    DEFAULT_REQUEST_DELAY_SECONDS as SCREENER_DEFAULT_REQUEST_DELAY_SECONDS,
    fetch_screener_shares_outstanding,
)
from stock_advisor.data.sec_edgar import get_sec_fundamentals
from stock_advisor.data.ownership import get_ownership_fundamentals
from stock_advisor.data.universe import (
    UNIVERSE_COLUMNS,
    build_stock_master_frame,
    load_stock_universe,
    refresh_bse_stock_universe,
    refresh_full_stock_universe,
    refresh_india_stock_universe,
    refresh_stock_universe,
)

logger = logging.getLogger(__name__)
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
DAILY_MARKET_CLOSE_BUFFER = time(15, 45)
PRICE_CACHE_ENABLED = os.getenv("PRICE_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
PRICE_CACHE_MAX_AGE_HOURS = float(os.getenv("PRICE_CACHE_MAX_AGE_HOURS", "8"))
CACHEABLE_INTERVALS = {"1d", "1wk", "1mo"}
CACHE_REFRESH_HINT = "Refresh the price cache from the sidebar to populate this."

FUNDAMENTAL_KEYS = [
    "shortName",
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "debtToEquity",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "returnOnEquity",
    "dividendYield",
    "beta",
]


def get_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Read cached OHLCV price history for a ticker.

    Pure read from price_history_cache — this never calls a market data provider and
    never writes anything. The only place price history is ever fetched and stored is
    fill_price_cache_for_universe(), triggered from the sidebar's "Refresh price cache"
    button or the daily_refresh CLI/cron — no MCP tool ever fetches live. If nothing
    is cached yet for this ticker (or the cached period is shorter than requested), this
    returns an empty DataFrame — callers should surface CACHE_REFRESH_HINT to the user.
    """
    return _load_cached_price_history(ticker, period, interval)


def get_price_histories(
    tickers: list[str] | tuple[str, ...],
    period: str = "6mo",
    interval: str = "1d",
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read cached OHLCV histories for many tickers.

    Pure read from price_history_cache — never fetches from a provider or writes.
    Tickers with nothing cached are simply absent from the returned dict.
    """
    unique_tickers = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))
    if not unique_tickers:
        return {}

    results: dict[str, pd.DataFrame] = {}
    for index, ticker in enumerate(unique_tickers):
        cached = _load_cached_price_history(ticker, period, interval)
        if not cached.empty:
            results[ticker] = cached
        if progress_callback:
            progress_callback(index + 1, len(unique_tickers))
    return results


def _load_cached_price_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not _price_cache_allowed(interval):
        return pd.DataFrame()
    ticker_key = _cache_ticker(ticker)
    requested_days = _period_days(period)
    try:
        with _price_cache_connection() as conn:
            _ensure_price_cache_schema(conn)
            ticker_id = _ticker_id(conn, ticker_key)
            if ticker_id is None:
                return pd.DataFrame()
            meta = conn.execute(
                """
                SELECT max_period_days, fetched_at
                FROM price_cache_meta
                WHERE ticker_id = ? AND interval = ?
                """,
                (ticker_id, interval),
            ).fetchone()
            if not meta:
                return pd.DataFrame()
            if requested_days is not None and int(meta["max_period_days"] or 0) < requested_days:
                return pd.DataFrame()
            params: list[Any] = [ticker_id, interval]
            date_filter = ""
            if requested_days is not None:
                start_date = (_current_market_datetime().date() - timedelta(days=requested_days + 7)).isoformat()
                date_filter = " AND date >= ?"
                params.append(start_date)
            rows = conn.execute(
                f"""
                SELECT date, open, high, low, close, volume, provider, fetched_at
                FROM price_history_cache
                WHERE ticker_id = ? AND interval = ?{date_filter}
                ORDER BY date
                """,
                params,
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Price cache load failed for %s: %s", ticker_key, exc)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(row) for row in rows])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    if df.empty:
        return pd.DataFrame()
    df.attrs["provider"] = "sqlite_cache"
    df.attrs["cache_fetched_at"] = str(meta["fetched_at"])
    df.attrs["selected_ticker"] = ticker_key
    return df.drop(columns=["provider", "fetched_at"], errors="ignore")


def _store_price_history(ticker: str, interval: str, df: pd.DataFrame) -> bool:
    """Persist a fetched price-history batch. Returns True iff this call refreshed live_metrics
    for the ticker (i.e. the batch reached a new latest_date) -- used by callers up the stack to
    surface a live "live metrics updated for N tickers" count in the sidebar during a refresh."""
    if df.empty or not _price_cache_allowed(interval):
        return False
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return False

    ticker_key = _cache_ticker(ticker)
    provider = str(df.attrs.get("provider") or "unknown")
    fetched_at = _current_market_datetime().isoformat()
    cache_df = df.copy()
    cache_df["date"] = pd.to_datetime(cache_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in cache_df.columns:
            cache_df[column] = None
        cache_df[column] = pd.to_numeric(cache_df[column], errors="coerce")
    has_turnover = "turnover" in cache_df.columns
    if has_turnover:
        cache_df["turnover"] = pd.to_numeric(cache_df["turnover"], errors="coerce")
    cache_df = cache_df.dropna(subset=["date", "open", "high", "low", "close"])
    if provider == "yfinance_batch":
        # yf.download regularly carries a thinly-traded security's last known close forward as a
        # synthetic zero-volume "close" for every calendar day it doesn't actually trade, instead
        # of returning nothing -- confirmed directly against bhavcopy: e.g. EDUCOMP.NS genuinely
        # traded (bhavcopy, real volume) on 2026-08-24, and Yahoo served the identical 0.88 close
        # with volume=0 on every surrounding day. Treating that as a real fetch silently advances
        # latest_date forever (masking genuine staleness from the dormancy check) and pollutes
        # price_history_cache with fabricated flat rows that feed straight into
        # return_30d_pct/return_365d_pct. Bhavcopy is exempt -- its small zero-volume rate looks
        # like genuine auction/corporate-action rows from the authoritative exchange record, not
        # fabrication.
        cache_df = cache_df[cache_df["volume"].fillna(0) > 0]
    if cache_df.empty:
        return False

    rows = [
        (
            interval,
            row.date,
            None if pd.isna(row.open) else float(row.open),
            None if pd.isna(row.high) else float(row.high),
            None if pd.isna(row.low) else float(row.low),
            None if pd.isna(row.close) else float(row.close),
            None if pd.isna(row.volume) else float(row.volume),
            provider,
            fetched_at,
        )
        for row in cache_df.itertuples(index=False)
    ]
    # This batch's own span — may be an older backward-fill (start date preponed) whose dates
    # are all before what's already cached, so it must never be treated as "the latest data".
    batch_min_date = min(row[1] for row in rows)
    batch_max_date = max(row[1] for row in rows)

    try:
        with _price_cache_connection() as conn:
            _ensure_price_cache_schema(conn)
            ticker_id = _ticker_id(conn, ticker_key, create=True)
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_history_cache
                (ticker_id, interval, date, open, high, low, close, volume, provider, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(ticker_id,) + row for row in rows],
            )
            existing_meta = conn.execute(
                """
                SELECT latest_date, earliest_date
                FROM price_cache_meta
                WHERE ticker_id = ? AND interval = ?
                """,
                (ticker_id, interval),
            ).fetchone()
            existing_latest = str(existing_meta["latest_date"]) if existing_meta and existing_meta["latest_date"] else None
            existing_earliest = str(existing_meta["earliest_date"]) if existing_meta and existing_meta["earliest_date"] else None
            # Combine explicitly with max()/min() rather than overwriting outright — a backward-fill
            # batch's own max/min can be older than what's already cached, and price_cache_meta must
            # never regress latest_date/earliest_date just because this particular write was old data.
            latest_date = max(batch_max_date, existing_latest) if existing_latest else batch_max_date
            earliest_date = min(batch_min_date, existing_earliest) if existing_earliest else batch_min_date
            today = _current_market_datetime().date()
            max_period_days = max(1, (today - datetime.strptime(earliest_date, "%Y-%m-%d").date()).days)
            row_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM price_history_cache
                WHERE ticker_id = ? AND interval = ?
                """,
                (ticker_id, interval),
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO price_cache_meta
                (ticker_id, interval, max_period_days, latest_date, earliest_date, provider, fetched_at, row_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker_id, interval) DO UPDATE SET
                    max_period_days=excluded.max_period_days,
                    latest_date=excluded.latest_date,
                    earliest_date=excluded.earliest_date,
                    provider=excluded.provider,
                    fetched_at=excluded.fetched_at,
                    row_count=excluded.row_count
                """,
                (ticker_id, interval, max_period_days, latest_date, earliest_date, provider, fetched_at, int(row_count or 0)),
            )
            # Deliberately NOT touching backfill_exhausted_start_date here (unlike an
            # INSERT OR REPLACE, which would silently wipe it back to NULL on every write) --
            # only _mark_backfill_exhausted sets it, and only fill_price_cache_for_universe's
            # classification step reads it. A plain price write (this function) must never
            # invalidate a confirmed "nothing exists before this date" finding.

            # price_data/live_metrics reflect this batch's OWN latest row, not the all-time latest_date
            # above — their own ON CONFLICT ... WHERE guards below already refuse to regress if this
            # batch happens to be an older backward-fill, so it's safe to always attempt the upsert.
            latest_row = next(row for row in rows if row[1] == batch_max_date)
            latest_turnover = None
            if has_turnover:
                match = cache_df.loc[cache_df["date"] == batch_max_date, "turnover"]
                if not match.empty and pd.notna(match.iloc[-1]):
                    latest_turnover = float(match.iloc[-1])
            conn.execute(
                """
                INSERT INTO price_data (ticker_id, date, open, high, low, close, volume, turnover, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker_id) DO UPDATE SET
                    date=excluded.date, open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume,
                    turnover=COALESCE(excluded.turnover, price_data.turnover),
                    fetched_at=excluded.fetched_at
                WHERE excluded.date >= COALESCE(price_data.date, '')
                """,
                (
                    ticker_id, latest_row[1], latest_row[2], latest_row[3], latest_row[4], latest_row[5],
                    latest_row[6], latest_turnover, fetched_at,
                ),
            )

            if interval == "1d" and batch_max_date == latest_date:
                _refresh_live_metrics(conn, ticker_id, interval, batch_max_date, float(latest_row[5]))
                return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Price cache store failed for %s: %s", ticker_key, exc)
        return False
    return False


def _refresh_live_metrics(conn: sqlite3.Connection, ticker_id: int, interval: str, latest_date: str, last_price: float) -> None:
    """Recompute year_high/year_low/near_52w_high_pct/return_{7,30,90,180,365}d_pct for a
    ticker from price_history_cache (which already has whatever was just written, plus any
    older history) and upsert into live_metrics. Cheap: a handful of indexed range queries."""
    window = conn.execute(
        """
        SELECT MAX(high) AS year_high, MIN(low) AS year_low
        FROM price_history_cache
        WHERE ticker_id = ? AND interval = ? AND date >= date(?, '-370 days')
        """,
        (ticker_id, interval, latest_date),
    ).fetchone()
    year_high = float(window["year_high"]) if window and window["year_high"] is not None else None
    year_low = float(window["year_low"]) if window and window["year_low"] is not None else None
    near_52w_high_pct = (
        round(100 * (year_high - last_price) / year_high, 4) if year_high else None
    )

    def _return_pct(days: int) -> float | None:
        prior = conn.execute(
            """
            SELECT close FROM price_history_cache
            WHERE ticker_id = ? AND interval = ? AND date <= date(?, ?)
            ORDER BY date DESC LIMIT 1
            """,
            (ticker_id, interval, latest_date, f"-{days} days"),
        ).fetchone()
        if not prior or not prior["close"]:
            return None
        prior_close = float(prior["close"])
        if prior_close == 0:
            return None
        return round(100 * (last_price - prior_close) / prior_close, 4)

    return_7d_pct = _return_pct(7)
    return_30d_pct = _return_pct(30)
    return_90d_pct = _return_pct(90)
    return_180d_pct = _return_pct(180)
    return_365d_pct = _return_pct(365)
    computed_at = _current_market_datetime().isoformat()
    conn.execute(
        """
        INSERT INTO live_metrics
        (ticker_id, last_price, year_high, year_low, near_52w_high_pct,
         return_7d_pct, return_30d_pct, return_90d_pct, return_180d_pct, return_365d_pct,
         as_of_date, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker_id) DO UPDATE SET
            last_price=excluded.last_price, year_high=excluded.year_high, year_low=excluded.year_low,
            near_52w_high_pct=excluded.near_52w_high_pct,
            return_7d_pct=excluded.return_7d_pct, return_30d_pct=excluded.return_30d_pct,
            return_90d_pct=excluded.return_90d_pct, return_180d_pct=excluded.return_180d_pct,
            return_365d_pct=excluded.return_365d_pct,
            as_of_date=excluded.as_of_date, computed_at=excluded.computed_at
        WHERE excluded.as_of_date >= COALESCE(live_metrics.as_of_date, '')
        """,
        (
            ticker_id, last_price, year_high, year_low, near_52w_high_pct,
            return_7d_pct, return_30d_pct, return_90d_pct, return_180d_pct, return_365d_pct,
            latest_date, computed_at,
        ),
    )


def get_live_metrics_for_tickers(tickers: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Bulk-read live_metrics (price_history_cache-derived last_price/year_high/year_low/
    near_52w_high_pct/return_{7,30,90,180,365}d_pct) for the given tickers, keyed by ticker.

    This is THE read path for these metrics -- live_metrics is the only place they're ever
    written (see _refresh_live_metrics), computed purely from price_history_cache. Used by
    universe._apply_live_metrics_overlay so the CSV-based universe views display the same
    cache-derived numbers as everything else, instead of their own separately-sourced (or
    entirely absent) copies. A ticker with no live_metrics row yet is simply omitted from the
    result -- callers should leave whatever they already had for it untouched.
    """
    unique = list(dict.fromkeys(_cache_ticker(t) for t in tickers if _cache_ticker(t)))
    if not unique:
        return {}
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique)
        rows = conn.execute(
            f"""
            SELECT im.ticker AS ticker, lm.last_price, lm.year_high, lm.year_low, lm.near_52w_high_pct,
                   lm.return_7d_pct, lm.return_30d_pct, lm.return_90d_pct, lm.return_180d_pct, lm.return_365d_pct
            FROM instrument_master im
            JOIN live_metrics lm ON lm.ticker_id = im.ticker_id
            WHERE im.ticker IN ({placeholders})
            """,
            unique,
        ).fetchall()
    return {row["ticker"]: {key: row[key] for key in row.keys() if key != "ticker"} for row in rows}


def _overlay_exchange_latest_rows(frames: dict[str, pd.DataFrame], *, interval: str) -> dict[str, pd.DataFrame]:
    """Merge latest official NSE/BSE EOD rows into existing daily histories and persist them.

    Only called from fill_price_cache_for_universe's fetch path — this both merges the
    overlay into the in-memory frame it returns and stores it, so it counts as a write.
    """
    if str(interval).strip().lower() != "1d" or not frames:
        return frames
    candidate_frames = { _cache_ticker(ticker): frame for ticker, frame in frames.items() if frame is not None and not frame.empty }
    if not candidate_frames:
        return frames
    try:
        exchange_rows = get_latest_exchange_eod_rows(tuple(candidate_frames))
    except Exception as exc:  # noqa: BLE001
        logger.info("Exchange EOD overlay unavailable: %s", exc)
        return frames
    if not exchange_rows:
        return frames

    out = dict(frames)
    for ticker, frame in candidate_frames.items():
        exchange_frame = exchange_rows.get(ticker)
        if exchange_frame is None or exchange_frame.empty:
            continue
        merged = _merge_price_history(frame, exchange_frame)
        if merged.empty:
            continue
        out[ticker] = merged
        _store_price_history(ticker, interval, exchange_frame)
    return out


def _merge_price_history(base: pd.DataFrame, patch: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    if base.empty or patch.empty or not required.issubset(base.columns) or not required.issubset(patch.columns):
        return base
    merged = pd.concat([base, patch], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    if merged.empty:
        return base
    merged["date_key"] = merged["date"].dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates("date_key", keep="last").drop(columns=["date_key"]).sort_values("date").reset_index(drop=True)
    merged.attrs.update(base.attrs)
    merged.attrs["provider"] = str(patch.attrs.get("provider") or base.attrs.get("provider") or "exchange_eod_overlay")
    merged.attrs["exchange_trade_date"] = patch.attrs.get("exchange_trade_date")
    return merged


def _cached_history_is_fresh(df: pd.DataFrame, interval: str) -> bool:
    if df.empty or not _price_cache_allowed(interval):
        return False
    return _cache_fetched_at_is_fresh(df.attrs.get("cache_fetched_at"), interval)


def _cache_fetched_at_is_fresh(value: Any, interval: str) -> bool:
    fetched_at = _parse_cache_datetime(value)
    if fetched_at is None:
        return False
    now = _current_market_datetime()
    if interval == "1d" and now.time() >= DAILY_MARKET_CLOSE_BUFFER:
        fetched_local = fetched_at.astimezone(MARKET_TIMEZONE)
        if fetched_local.date() == now.date() and fetched_local.time() < DAILY_MARKET_CLOSE_BUFFER:
            return False
    age = now - fetched_at.astimezone(MARKET_TIMEZONE)
    return age <= timedelta(hours=max(0.1, PRICE_CACHE_MAX_AGE_HOURS))


def _price_cache_allowed(interval: str) -> bool:
    return PRICE_CACHE_ENABLED and str(interval) in CACHEABLE_INTERVALS


def _price_cache_connection() -> sqlite3.Connection:
    # This app now has several concurrent writers against the same file at once (a price-cache
    # refresh, a shares-outstanding backfill, market-cap recompute, an ad-hoc query -- all opening
    # their own short-lived connection via this same factory). The default rollback-journal mode
    # takes an exclusive lock for the whole duration of any write, and sqlite3.connect()'s default
    # 5s busy-timeout is too short once real work (not just a quick UPDATE) is what's holding that
    # lock -- that combination is exactly what "database is locked" means. WAL mode lets readers
    # and a single writer proceed without blocking each other (a one-time, persisted-in-the-file
    # setting -- cheap to re-issue on every connect once already WAL), and a much longer
    # busy_timeout makes a connection wait out a genuinely busy writer instead of failing outright.
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_price_cache_schema(conn: sqlite3.Connection) -> None:
    _migrate_stock_list_table_rename(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instrument_master (
            ticker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            market TEXT NOT NULL,
            symbol TEXT,
            name TEXT,
            isin TEXT,
            source TEXT,
            active INTEGER,
            series TEXT,
            exchange TEXT,
            security_id TEXT,
            nse_ticker TEXT,
            bse_ticker TEXT,
            nse_security_id TEXT,
            bse_security_id TEXT,
            in_nifty_total_market INTEGER NOT NULL DEFAULT 0,
            synced_at TEXT
        )
        """
    )
    _migrate_instrument_master_columns(conn)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_instrument_master_isin_unique ON instrument_master(isin)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS security_classification (
            ticker_id INTEGER PRIMARY KEY REFERENCES instrument_master(ticker_id),
            sector TEXT,
            industry TEXT,
            basic_industry TEXT,
            index_name TEXT,
            classification_source TEXT,
            data_quality TEXT,
            synced_at TEXT
        )
        """
    )
    _migrate_security_classification_split(conn)
    _migrate_security_classification_iics_columns(conn)
    _load_iics_classification_seed_if_needed(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_data (
            ticker_id INTEGER PRIMARY KEY REFERENCES instrument_master(ticker_id),
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            turnover REAL,
            fetched_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_cap (
            ticker_id INTEGER PRIMARY KEY REFERENCES instrument_master(ticker_id),
            market_cap REAL,
            free_float_market_cap REAL,
            shares_outstanding REAL,
            shares_outstanding_source TEXT,
            shares_outstanding_fetched_at TEXT,
            date TEXT,
            fetched_at TEXT
        )
        """
    )
    _migrate_market_cap_columns(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_metrics (
            ticker_id INTEGER PRIMARY KEY REFERENCES instrument_master(ticker_id),
            last_price REAL,
            year_high REAL,
            year_low REAL,
            near_52w_high_pct REAL,
            return_7d_pct REAL,
            return_30d_pct REAL,
            return_90d_pct REAL,
            return_180d_pct REAL,
            return_365d_pct REAL,
            as_of_date TEXT,
            computed_at TEXT
        )
        """
    )
    _migrate_live_metrics_columns(conn)
    _migrate_legacy_ticker_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history_cache (
            ticker_id INTEGER NOT NULL REFERENCES instrument_master(ticker_id),
            interval TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            provider TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (ticker_id, interval, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_cache_meta (
            ticker_id INTEGER NOT NULL REFERENCES instrument_master(ticker_id),
            interval TEXT NOT NULL,
            max_period_days INTEGER NOT NULL,
            latest_date TEXT,
            earliest_date TEXT,
            backfill_exhausted_start_date TEXT,
            provider TEXT,
            fetched_at TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            PRIMARY KEY (ticker_id, interval)
        )
        """
    )
    _migrate_price_cache_meta_columns(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_history_cache_lookup ON price_history_cache (ticker_id, interval, date)")


def _migrate_market_cap_columns(conn: sqlite3.Connection) -> None:
    """Add shares_outstanding/shares_outstanding_source/shares_outstanding_fetched_at to
    market_cap (needed by recompute_market_cap_from_shares_outstanding). No-ops once already
    migrated. shares_outstanding_source ("nse_xbrl" or "screener_derived") lets the NSE backfill
    pass safely upgrade a screener-derived figure without a later screener re-run downgrading a
    more authoritative NSE one."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(market_cap)")}
    if not cols:
        return  # table doesn't exist yet (handled by the CREATE TABLE IF NOT EXISTS above)
    if "shares_outstanding" not in cols:
        conn.execute("ALTER TABLE market_cap ADD COLUMN shares_outstanding REAL")
    if "shares_outstanding_source" not in cols:
        conn.execute("ALTER TABLE market_cap ADD COLUMN shares_outstanding_source TEXT")
    if "shares_outstanding_fetched_at" not in cols:
        conn.execute("ALTER TABLE market_cap ADD COLUMN shares_outstanding_fetched_at TEXT")
    conn.commit()


def _migrate_live_metrics_columns(conn: sqlite3.Connection) -> None:
    """Add return_7d_pct/return_90d_pct/return_180d_pct to live_metrics (needed alongside the
    existing return_30d_pct/return_365d_pct -- see _refresh_live_metrics). No-ops once already
    migrated."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(live_metrics)")}
    if not cols:
        return  # table doesn't exist yet (handled by the CREATE TABLE IF NOT EXISTS above)
    for column in ["return_7d_pct", "return_90d_pct", "return_180d_pct"]:
        if column not in cols:
            conn.execute(f"ALTER TABLE live_metrics ADD COLUMN {column} REAL")
    conn.commit()


def _migrate_price_cache_meta_columns(conn: sqlite3.Connection) -> None:
    """Add earliest_date to price_cache_meta (needed to tell whether a configured
    PRICE_HISTORY_START_DATE has moved earlier than what's already cached) and backfill it
    from price_history_cache for any pre-existing rows. Also adds backfill_exhausted_start_date
    (needed by fill_price_cache_for_universe to stop retrying a backward-fill that a complete
    bhavcopy day-walk has already confirmed can't go back any further — see that function's
    docstring). No-ops once already migrated."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(price_cache_meta)")}
    if not cols:
        return  # table doesn't exist yet (handled by the CREATE TABLE IF NOT EXISTS above)
    if "earliest_date" not in cols:
        conn.execute("ALTER TABLE price_cache_meta ADD COLUMN earliest_date TEXT")
    if "backfill_exhausted_start_date" not in cols:
        conn.execute("ALTER TABLE price_cache_meta ADD COLUMN backfill_exhausted_start_date TEXT")
    conn.execute(
        """
        UPDATE price_cache_meta
        SET earliest_date = (
            SELECT MIN(date) FROM price_history_cache h
            WHERE h.ticker_id = price_cache_meta.ticker_id AND h.interval = price_cache_meta.interval
        )
        WHERE earliest_date IS NULL
        """
    )
    conn.commit()


def _migrate_stock_list_table_rename(conn: sqlite3.Connection) -> None:
    """One-time rename: stock_list -> instrument_master (clearer name for the security-identity
    table post schema-split). No-ops if already renamed or on a fresh install."""
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "stock_list" in tables and "instrument_master" not in tables:
        conn.execute("ALTER TABLE stock_list RENAME TO instrument_master")
        conn.commit()


def _migrate_instrument_master_columns(conn: sqlite3.Connection) -> None:
    """Drop instrument_master's live-metric columns (they're fact/time-varying data that belongs
    in price_data/market_cap/live_metrics, not the security-master/dimension table) and add
    synced_at. No-ops once already migrated. Cheap: none of these columns are indexed/constrained,
    so SQLite's ALTER TABLE ... DROP COLUMN is a metadata-only change here, not a full rewrite.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(instrument_master)")}
    if not cols:
        return  # table doesn't exist yet (handled by the CREATE TABLE IF NOT EXISTS above)
    metric_cols = [
        "free_float_market_cap", "last_price", "year_high", "year_low",
        "near_52w_high_pct", "return_30d_pct", "return_365d_pct", "refreshed_at",
    ]
    for column in metric_cols:
        if column in cols:
            conn.execute(f"ALTER TABLE instrument_master DROP COLUMN {column}")
    if "synced_at" not in cols:
        conn.execute("ALTER TABLE instrument_master ADD COLUMN synced_at TEXT")
    if "in_nifty_total_market" not in cols:
        conn.execute("ALTER TABLE instrument_master ADD COLUMN in_nifty_total_market INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def _migrate_security_classification_split(conn: sqlite3.Connection) -> None:
    """Move sector/industry/basic_industry/index_name/classification_source/data_quality out of
    instrument_master into their own security_classification table (keyed by ticker_id). One-time,
    idempotent — backfills security_classification from any existing instrument_master data before
    dropping the columns, so nothing is lost.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(instrument_master)")}
    classification_cols = ["sector", "industry", "basic_industry", "index_name", "classification_source", "data_quality"]
    if not any(c in cols for c in classification_cols):
        return  # already migrated, or fresh install
    conn.execute(
        """
        INSERT INTO security_classification (ticker_id, sector, industry, basic_industry, index_name, classification_source, data_quality, synced_at)
        SELECT ticker_id, sector, industry, basic_industry, index_name, classification_source, data_quality, synced_at
        FROM instrument_master
        WHERE ticker_id NOT IN (SELECT ticker_id FROM security_classification)
        """
    )
    for column in classification_cols:
        if column in cols:
            conn.execute(f"ALTER TABLE instrument_master DROP COLUMN {column}")
    conn.commit()


IICS_CLASSIFICATION_SOURCE = "nse_bse_iics_screener"
_IICS_SEED_CSV_PATH = PROJECT_ROOT / "data" / "iics_classification.csv"


def _migrate_security_classification_iics_columns(conn: sqlite3.Connection) -> None:
    """Add the Macro-Economic Sector level and every level's numeric IICS code to
    security_classification -- the table previously only had 3 of the official 4 levels
    (sector/industry/basic_industry names, no macro_sector, no codes at all). No-ops once
    already migrated. The existing sector/industry/basic_industry name columns are reused as-is
    for IICS's Sector/Industry/Basic-Industry names -- see build_iics_classification_mapping."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(security_classification)")}
    if not cols:
        return  # table doesn't exist yet (handled by the CREATE TABLE IF NOT EXISTS above)
    for column in ["macro_sector_code", "macro_sector", "sector_code", "industry_code", "basic_industry_code"]:
        if column not in cols:
            conn.execute(f"ALTER TABLE security_classification ADD COLUMN {column} TEXT")
    conn.commit()


def store_security_classification(
    ticker: str,
    *,
    macro_sector_code: str | None,
    macro_sector: str | None,
    sector_code: str | None,
    sector: str | None,
    industry_code: str | None,
    industry: str | None,
    basic_industry_code: str | None,
    basic_industry: str | None,
    source: str,
) -> bool:
    """Persist one ticker's full 4-level IICS classification into security_classification.
    Used by both the one-time screener.in/BSE-PDF backfill (source=IICS_CLASSIFICATION_SOURCE)
    and load_iics_classification_seed (replaying the committed data/iics_classification.csv into
    a fresh DB). sync_instrument_master's routine CSV-sourced sync is guarded to never overwrite
    a row this wrote -- see its classification upsert."""
    ticker_key = _cache_ticker(ticker)
    if not ticker_key:
        return False
    synced_at = _current_market_datetime().isoformat()
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        ticker_id = _ticker_id(conn, ticker_key, create=True)
        conn.execute(
            """
            INSERT INTO security_classification
            (ticker_id, macro_sector_code, macro_sector, sector_code, sector, industry_code, industry,
             basic_industry_code, basic_industry, classification_source, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker_id) DO UPDATE SET
                macro_sector_code=excluded.macro_sector_code,
                macro_sector=excluded.macro_sector,
                sector_code=excluded.sector_code,
                sector=excluded.sector,
                industry_code=excluded.industry_code,
                industry=excluded.industry,
                basic_industry_code=excluded.basic_industry_code,
                basic_industry=excluded.basic_industry,
                classification_source=excluded.classification_source,
                synced_at=excluded.synced_at
            """,
            (
                ticker_id, macro_sector_code, macro_sector, sector_code, sector, industry_code, industry,
                basic_industry_code, basic_industry, source, synced_at,
            ),
        )
        conn.commit()
    return True


def _load_iics_classification_seed_if_needed(conn: sqlite3.Connection) -> None:
    """Seed security_classification from the committed data/iics_classification.csv on a fresh
    (or not-yet-seeded) database -- pure local file + DB write, no network. This is what makes a
    fresh clone of the repo get this data without ever re-scraping screener.in: the scrape itself
    only ever happens via the separate, manual scripts/build_iics_classification.py, which writes
    that CSV; this function just replays it. Called from _ensure_price_cache_schema, which runs on
    nearly every connection this app opens, so it must stay cheap once already seeded -- a single
    EXISTS check, no file I/O, before touching the CSV at all."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(security_classification)")}
    if "classification_source" not in cols:
        return  # table/column doesn't exist yet
    already_seeded = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM security_classification WHERE classification_source = ?)",
        (IICS_CLASSIFICATION_SOURCE,),
    ).fetchone()[0]
    if already_seeded or not _IICS_SEED_CSV_PATH.exists():
        return
    with _IICS_SEED_CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker_key = _cache_ticker(row.get("ticker"))
            if not ticker_key:
                continue
            ticker_id = _ticker_id(conn, ticker_key, create=True)
            conn.execute(
                """
                INSERT INTO security_classification
                (ticker_id, macro_sector_code, macro_sector, sector_code, sector, industry_code, industry,
                 basic_industry_code, basic_industry, classification_source, synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker_id) DO UPDATE SET
                    macro_sector_code=excluded.macro_sector_code,
                    macro_sector=excluded.macro_sector,
                    sector_code=excluded.sector_code,
                    sector=excluded.sector,
                    industry_code=excluded.industry_code,
                    industry=excluded.industry,
                    basic_industry_code=excluded.basic_industry_code,
                    basic_industry=excluded.basic_industry,
                    classification_source=excluded.classification_source,
                    synced_at=excluded.synced_at
                WHERE security_classification.classification_source IS NOT ?
                """,
                (
                    ticker_id,
                    row.get("macro_sector_code"), row.get("macro_sector"),
                    row.get("sector_code"), row.get("sector"),
                    row.get("industry_code"), row.get("industry"),
                    row.get("basic_industry_code"), row.get("basic_industry"),
                    IICS_CLASSIFICATION_SOURCE, row.get("synced_at") or row.get("fetched_at") or _current_market_datetime().isoformat(),
                    IICS_CLASSIFICATION_SOURCE,
                ),
            )
    conn.commit()


def _migrate_legacy_ticker_schema(conn: sqlite3.Connection) -> None:
    """One-time, resumable migration from the old ticker-TEXT-keyed schema to ticker_id.

    Safe to call on every connection open: no-ops once migrated, and resumable if a prior
    attempt was interrupted between the rename and the drop (it checks for the `_legacy`
    tables first, so real cached data is never stranded under a renamed table).
    """
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    has_legacy_backup = "price_cache_meta_legacy" in tables or "price_history_cache_legacy" in tables

    if not has_legacy_backup:
        if "price_cache_meta" not in tables:
            return  # brand new install, nothing to migrate
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(price_cache_meta)")}
        if "ticker_id" in cols or "ticker" not in cols:
            return  # already migrated, or an unrecognized shape — don't touch
        logger.info("Migrating legacy price_cache_meta/price_history_cache to ticker_id schema...")
        conn.execute("ALTER TABLE price_cache_meta RENAME TO price_cache_meta_legacy")
        conn.execute("ALTER TABLE price_history_cache RENAME TO price_history_cache_legacy")

    conn.execute(
        """
        INSERT OR IGNORE INTO instrument_master (ticker, market)
        SELECT DISTINCT ticker, CASE WHEN ticker LIKE '%.NS' OR ticker LIKE '%.BO' THEN 'IN' ELSE 'US' END
        FROM price_cache_meta_legacy
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO instrument_master (ticker, market)
        SELECT DISTINCT ticker, CASE WHEN ticker LIKE '%.NS' OR ticker LIKE '%.BO' THEN 'IN' ELSE 'US' END
        FROM price_history_cache_legacy
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_cache_meta (
            ticker_id INTEGER NOT NULL REFERENCES instrument_master(ticker_id),
            interval TEXT NOT NULL, max_period_days INTEGER NOT NULL, latest_date TEXT,
            provider TEXT, fetched_at TEXT NOT NULL, row_count INTEGER NOT NULL,
            PRIMARY KEY (ticker_id, interval)
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO price_cache_meta (ticker_id, interval, max_period_days, latest_date, provider, fetched_at, row_count)
        SELECT sl.ticker_id, m.interval, m.max_period_days, m.latest_date, m.provider, m.fetched_at, m.row_count
        FROM price_cache_meta_legacy m JOIN instrument_master sl ON sl.ticker = m.ticker
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history_cache (
            ticker_id INTEGER NOT NULL REFERENCES instrument_master(ticker_id),
            interval TEXT NOT NULL, date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            provider TEXT, fetched_at TEXT NOT NULL,
            PRIMARY KEY (ticker_id, interval, date)
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO price_history_cache (ticker_id, interval, date, open, high, low, close, volume, provider, fetched_at)
        SELECT sl.ticker_id, h.interval, h.date, h.open, h.high, h.low, h.close, h.volume, h.provider, h.fetched_at
        FROM price_history_cache_legacy h JOIN instrument_master sl ON sl.ticker = h.ticker
        """
    )

    conn.execute("DROP TABLE price_cache_meta_legacy")
    conn.execute("DROP TABLE price_history_cache_legacy")
    conn.commit()
    logger.info("Legacy price cache migration complete.")

    try:
        sync_instrument_master()
    except Exception as exc:  # noqa: BLE001
        logger.warning("instrument_master enrichment sync after migration failed: %s", exc)


def _ticker_id(conn: sqlite3.Connection, ticker_key: str, *, create: bool = False) -> int | None:
    row = conn.execute("SELECT ticker_id FROM instrument_master WHERE ticker = ?", (ticker_key,)).fetchone()
    if row:
        return int(row["ticker_id"])
    if not create:
        return None
    conn.execute(
        "INSERT OR IGNORE INTO instrument_master (ticker, market) VALUES (?, ?)",
        (ticker_key, _infer_market(ticker_key)),
    )
    row = conn.execute("SELECT ticker_id FROM instrument_master WHERE ticker = ?", (ticker_key,)).fetchone()
    return int(row["ticker_id"]) if row else None


def _bulk_ticker_ids(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, int]:
    """Resolve many ticker strings to ticker_id in one pass (chunked under SQLite's ~999-variable
    per-statement limit). Only returns tickers that already have an instrument_master row —
    callers that need lazy-create semantics should use _ticker_id(..., create=True) instead."""
    unique = list(dict.fromkeys(t for t in tickers if t))
    if not unique:
        return {}
    result: dict[str, int] = {}
    chunk_size = 500
    for start in range(0, len(unique), chunk_size):
        chunk = unique[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT ticker, ticker_id FROM instrument_master WHERE ticker IN ({placeholders})", chunk
        ).fetchall()
        result.update({row["ticker"]: row["ticker_id"] for row in rows})
    return result


_INSTRUMENT_MASTER_IDENTITY_COLUMNS = [
    "symbol", "name", "isin", "source", "active", "series",
    "exchange", "security_id", "nse_ticker", "bse_ticker", "nse_security_id", "bse_security_id",
]
_SECURITY_CLASSIFICATION_COLUMNS = [
    "sector", "industry", "basic_industry", "index_name", "classification_source", "data_quality",
]


def sync_instrument_master(universes: list[str] | None = None) -> dict[str, Any]:
    """Consolidate data/*.csv NSE/BSE universes into instrument_master + security_classification
    + market_cap (insert new, refresh existing).

    instrument_master only gets identity columns (plus market/synced_at) — it's a pure
    security-master/dimension table. Classification lives in security_classification;
    free_float_market_cap (the only metric the broad universe CSV provides) lives in market_cap.
    NIFTY Total Market membership is tagged as its own flag, independent of the ISIN-precedence
    identity backfill, so 'broad' universe membership stays reconstructable from the DB even
    after ISIN dedup.
    """
    frame = build_stock_master_frame(universes=universes)
    universes_used = list(universes) if universes else None
    if frame.empty:
        return {"synced_ticker_count": 0, "universes": universes_used}

    synced_at = _current_market_datetime().isoformat()
    identity_columns = ["market", "synced_at"] + _INSTRUMENT_MASTER_IDENTITY_COLUMNS
    placeholders = ", ".join(["?"] * (len(identity_columns) + 1))
    # isin is deliberately excluded from the UPDATE branch: it's only ever set on a fresh INSERT
    # (guarded by ON CONFLICT(isin) DO NOTHING below). Letting an UPDATE on an already-known
    # ticker touch isin risks a *second*, independent UNIQUE-constraint violation against some
    # other existing row — a real failure mode, not just a theoretical one (hit on the very first
    # full sync run against live data).
    update_clause = ", ".join(f"{c}=excluded.{c}" for c in identity_columns if c != "isin")
    records = frame.to_dict("records")
    identity_rows = [
        (
            record.get("ticker"),
            _infer_market(record.get("ticker")),
            synced_at,
            *(record.get(c) for c in _INSTRUMENT_MASTER_IDENTITY_COLUMNS),
        )
        for record in records
    ]

    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        conn.executemany(
            f"""
            INSERT INTO instrument_master (ticker, {", ".join(identity_columns)})
            VALUES ({placeholders})
            ON CONFLICT(ticker) DO UPDATE SET {update_clause}
            ON CONFLICT(isin) DO NOTHING
            """,
            identity_rows,
        )
        conn.commit()

        ticker_id_map = _bulk_ticker_ids(conn, [record.get("ticker") for record in records])

        classification_rows = [
            (ticker_id_map[record["ticker"]], synced_at, *(record.get(c) for c in _SECURITY_CLASSIFICATION_COLUMNS))
            for record in records
            if record.get("ticker") in ticker_id_map
        ]
        if classification_rows:
            classification_update = ", ".join(f"{c}=excluded.{c}" for c in [*_SECURITY_CLASSIFICATION_COLUMNS, "synced_at"])
            conn.executemany(
                f"""
                INSERT INTO security_classification (ticker_id, synced_at, {", ".join(_SECURITY_CLASSIFICATION_COLUMNS)})
                VALUES (?, ?, {", ".join("?" for _ in _SECURITY_CLASSIFICATION_COLUMNS)})
                ON CONFLICT(ticker_id) DO UPDATE SET {classification_update}
                WHERE security_classification.classification_source IS NOT '{IICS_CLASSIFICATION_SOURCE}'
                """,
                classification_rows,
            )
            conn.commit()

        # NIFTY Total Market membership + free_float_market_cap: loaded fresh here (not reused
        # from `frame`), so this stays correct even for a partial single-universe sync that
        # didn't include "broad" in `universes`.
        broad_frame = load_stock_universe(universe="broad")
        if not broad_frame.empty:
            broad_records = broad_frame.to_dict("records")
            broad_ticker_id_map = _bulk_ticker_ids(conn, [r.get("ticker") for r in broad_records])
            broad_ids = list(broad_ticker_id_map.values())
            if broad_ids:
                conn.executemany(
                    "UPDATE instrument_master SET in_nifty_total_market = 1 WHERE ticker_id = ?",
                    [(tid,) for tid in broad_ids],
                )
                market_cap_rows = [
                    (broad_ticker_id_map[r["ticker"]], r.get("free_float_market_cap"), r.get("refreshed_at"), synced_at)
                    for r in broad_records
                    if r.get("ticker") in broad_ticker_id_map
                ]
                conn.executemany(
                    """
                    INSERT INTO market_cap (ticker_id, free_float_market_cap, date, fetched_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(ticker_id) DO UPDATE SET
                        free_float_market_cap=excluded.free_float_market_cap,
                        date=excluded.date,
                        fetched_at=excluded.fetched_at
                    """,
                    market_cap_rows,
                )
                conn.commit()

    return {"synced_ticker_count": len(identity_rows), "universes": universes_used}


def list_instrument_master_tickers(universe: str, *, active_only: bool = True) -> list[str]:
    """Return tickers from instrument_master for an exchange-derivable universe ('full_nse',
    'full_bse', 'all_india') or 'broad' (via the in_nifty_total_market flag).

    active_only=True (default) excludes tickers confirmed dormant by
    fill_price_cache_for_universe (zero trades found across a full bhavcopy day-walk) — the same
    "NULL = active" convention as list_all_instrument_master_tickers. This is the universe source
    for every sector/industry/breadth/RRG/crossover/top-gainers analytics function, so a dormant
    ticker's last (stale, possibly months old) cached price no longer gets compared against
    everything else as if it were current."""
    normalized = str(universe or "").strip().lower()
    if normalized in {"broad", "nse_total_market", "nifty_total_market", "total_market"}:
        where = "in_nifty_total_market = 1"
    elif normalized in {"full_nse", "nse_full", "nse_equity"}:
        where = "market = 'IN' AND exchange IN ('NSE', 'NSE+BSE')"
    elif normalized in {"full_bse", "bse_full", "bse_equity"}:
        where = "market = 'IN' AND exchange IN ('BSE', 'NSE+BSE')"
    else:  # all_india and aliases
        where = "market = 'IN'"
    if active_only:
        where += " AND active IS NOT 0"
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        rows = conn.execute(f"SELECT ticker FROM instrument_master WHERE {where}").fetchall()
    return [row["ticker"] for row in rows]


def list_all_instrument_master_tickers(*, active_only: bool = True) -> list[str]:
    """Return every ticker in instrument_master, no exchange/market predicate — guarantees full
    coverage regardless of market/exchange data gaps. instrument_master currently has legitimate
    rows with market='US' (e.g. AAPL, used as a benchmark) or exchange IS NULL (e.g. ^NSEI) that
    every exchange-derived universe in list_instrument_master_tickers() above would miss.
    active_only=True excludes only rows explicitly marked active=0 — it keeps rows where active
    IS NULL, since some legitimate rows (the US benchmark tickers, index proxies) are NULL rather
    than 1 today."""
    where = "active IS NOT 0" if active_only else "1=1"
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        rows = conn.execute(f"SELECT ticker FROM instrument_master WHERE {where}").fetchall()
    return [row["ticker"] for row in rows]


def lookup_instrument_identity(ticker: str) -> dict[str, Any] | None:
    """Single-ticker identity lookup against instrument_master — matches on ticker, nse_ticker,
    bse_ticker, or symbol (whichever the caller happens to have). Returns isin/nse_ticker/
    bse_ticker/security ids/name, or None if the ticker isn't in instrument_master at all."""
    ticker_key = _cache_ticker(ticker)
    symbol = ticker_key.split(".", 1)[0]
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        row = conn.execute(
            """
            SELECT isin, nse_ticker, bse_ticker, nse_security_id, bse_security_id, security_id, name
            FROM instrument_master
            WHERE ticker = ? OR nse_ticker = ? OR bse_ticker = ? OR symbol = ?
            LIMIT 1
            """,
            (ticker_key, ticker_key, ticker_key, symbol),
        ).fetchone()
    if not row:
        return None
    return {
        "isin": row["isin"],
        "nse_ticker": row["nse_ticker"],
        "bse_ticker": row["bse_ticker"],
        "nse_security_id": row["nse_security_id"],
        "bse_security_id": row["bse_security_id"] or row["security_id"],
        "company_name": row["name"],
    }


def list_tickers_missing_shares_outstanding(
    *, active_only: bool = True, include_screener_sourced: bool = False
) -> list[str]:
    """instrument_master tickers with no market_cap.shares_outstanding yet — the candidate set
    for scripts/backfill_shares_outstanding.py. Shares outstanding barely changes (only on
    buybacks/issuance/splits/bonus issues), so once set for a ticker it stays out of this list
    until manually cleared, unlike price/market cap which need refreshing constantly.

    include_screener_sourced=True also returns tickers whose current value came from the
    lower-quality screener.in-derived fallback (source='screener_derived') -- upgrade candidates
    for the NSE XBRL pass, which is more authoritative. Used only by that pass; the default keeps
    this a pure "still has nothing" query."""
    where_active = "im.active IS NOT 0" if active_only else "1=1"
    missing_clause = "mc.shares_outstanding IS NULL"
    if include_screener_sourced:
        missing_clause = f"({missing_clause} OR mc.shares_outstanding_source = 'screener_derived')"
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        rows = conn.execute(
            f"""
            SELECT im.ticker AS ticker
            FROM instrument_master im
            LEFT JOIN market_cap mc ON mc.ticker_id = im.ticker_id
            WHERE {where_active} AND {missing_clause}
            """
        ).fetchall()
    return [row["ticker"] for row in rows]


def store_shares_outstanding(ticker: str, shares_outstanding: float, *, source: str) -> bool:
    """Persist a shares-outstanding figure for one ticker into market_cap. source is
    "nse_xbrl" (scripts/backfill_shares_outstanding.py's primary, official pass) or
    "screener_derived" (its opt-in fallback for BSE-only tickers NSE doesn't cover). Everywhere
    else, market cap itself is derived cheaply and locally from this by
    recompute_market_cap_from_shares_outstanding()."""
    ticker_key = _cache_ticker(ticker)
    if not ticker_key or not shares_outstanding or shares_outstanding <= 0:
        return False
    fetched_at = _current_market_datetime().isoformat()
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        ticker_id = _ticker_id(conn, ticker_key, create=True)
        conn.execute(
            """
            INSERT INTO market_cap (ticker_id, shares_outstanding, shares_outstanding_source, shares_outstanding_fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker_id) DO UPDATE SET
                shares_outstanding=excluded.shares_outstanding,
                shares_outstanding_source=excluded.shares_outstanding_source,
                shares_outstanding_fetched_at=excluded.shares_outstanding_fetched_at
            """,
            (ticker_id, shares_outstanding, source, fetched_at),
        )
        conn.commit()
    return True


def recompute_market_cap_from_shares_outstanding() -> dict[str, Any]:
    """Recompute market_cap.market_cap = shares_outstanding * latest close (price_data), for
    every ticker that has a shares_outstanding on file. Pure local SQL, no network call — safe
    to run on every price-cache refresh, since shares_outstanding itself is only ever updated by
    the separate, occasional screener.in backfill script. This is what makes market cap "fresh
    every day" without fetching it every day."""
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        fetched_at = _current_market_datetime().isoformat()
        cursor = conn.execute(
            """
            UPDATE market_cap
            SET market_cap = shares_outstanding * (
                    SELECT pd.close FROM price_data pd WHERE pd.ticker_id = market_cap.ticker_id
                ),
                date = (
                    SELECT pd.date FROM price_data pd WHERE pd.ticker_id = market_cap.ticker_id
                ),
                fetched_at = ?
            WHERE shares_outstanding IS NOT NULL
              AND EXISTS (SELECT 1 FROM price_data pd WHERE pd.ticker_id = market_cap.ticker_id AND pd.close IS NOT NULL)
            """,
            (fetched_at,),
        )
        conn.commit()
        updated = cursor.rowcount if cursor.rowcount is not None else 0
    return {"recomputed_ticker_count": max(0, updated)}


def clean_zero_volume_yfinance_rows(*, dry_run: bool = True) -> dict[str, Any]:
    """One-time cleanup for price_history_cache rows written before _store_price_history started
    discarding Yahoo's zero-volume placeholder rows (see its docstring): yf.download regularly
    carries a thinly-traded security's last known close forward as a synthetic zero-volume row for
    days it didn't actually trade, and every one of those written before the fix is still sitting
    in the cache today, corrupting latest_date/earliest_date and every return calculation that
    reads price_history_cache.

    Deletes every price_history_cache row with provider='yfinance_batch' and volume 0/NULL, then
    for each affected ticker recomputes price_cache_meta (latest_date/earliest_date/row_count) and
    live_metrics from whatever real rows remain -- so a ticker whose latest cached row was fake
    correctly rolls back to its last genuine trade date, making it eligible for normal
    re-classification (and eventually Phase C / dormancy confirmation) on the next refresh.

    dry_run=True (default): reports what would change without deleting or updating anything.
    Safe to re-run any number of times -- a no-op once the cache is clean.
    """
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        fake_row_filter = "interval = '1d' AND provider = 'yfinance_batch' AND (volume = 0 OR volume IS NULL)"
        affected_ids = [
            row["ticker_id"]
            for row in conn.execute(f"SELECT DISTINCT ticker_id FROM price_history_cache WHERE {fake_row_filter}").fetchall()
        ]
        rows_matched = int(
            conn.execute(f"SELECT COUNT(*) AS cnt FROM price_history_cache WHERE {fake_row_filter}").fetchone()["cnt"] or 0
        )

        if dry_run or not affected_ids:
            return {
                "dry_run": dry_run,
                "rows_deleted": 0,
                "rows_matched": rows_matched,
                "tickers_affected": len(affected_ids),
                "tickers_latest_date_changed": 0,
            }

        conn.execute(f"DELETE FROM price_history_cache WHERE {fake_row_filter}")

        today = _current_market_datetime().date()
        latest_date_changed_count = 0
        for ticker_id in affected_ids:
            existing_meta = conn.execute(
                "SELECT latest_date FROM price_cache_meta WHERE ticker_id = ? AND interval = '1d'", (ticker_id,)
            ).fetchone()
            old_latest = str(existing_meta["latest_date"]) if existing_meta and existing_meta["latest_date"] else None

            remaining = conn.execute(
                """
                SELECT MIN(date) AS earliest_date, MAX(date) AS latest_date, COUNT(*) AS row_count
                FROM price_history_cache WHERE ticker_id = ? AND interval = '1d'
                """,
                (ticker_id,),
            ).fetchone()
            new_latest = str(remaining["latest_date"]) if remaining and remaining["row_count"] else None

            if new_latest:
                new_earliest = str(remaining["earliest_date"])
                max_period_days = max(1, (today - datetime.strptime(new_earliest, "%Y-%m-%d").date()).days)
                conn.execute(
                    """
                    UPDATE price_cache_meta
                    SET latest_date = ?, earliest_date = ?, row_count = ?, max_period_days = ?
                    WHERE ticker_id = ? AND interval = '1d'
                    """,
                    (new_latest, new_earliest, int(remaining["row_count"]), max_period_days, ticker_id),
                )
                latest_row = conn.execute(
                    "SELECT close FROM price_history_cache WHERE ticker_id = ? AND interval = '1d' AND date = ?",
                    (ticker_id, new_latest),
                ).fetchone()
                if latest_row and latest_row["close"] is not None:
                    _refresh_live_metrics(conn, ticker_id, "1d", new_latest, float(latest_row["close"]))
            else:
                # every cached row for this ticker was fake -- nothing real left at all.
                conn.execute(
                    "UPDATE price_cache_meta SET latest_date = NULL, earliest_date = NULL, row_count = 0 WHERE ticker_id = ? AND interval = '1d'",
                    (ticker_id,),
                )

            # price_data holds a single "current" snapshot per ticker; if it was pointing at a
            # date that just got deleted as fake, roll it back to the last real row (or leave it
            # empty if nothing real remains) -- deliberately bypassing the normal
            # WHERE excluded.date >= price_data.date guard, since this is a corrective rollback,
            # not a routine fetch that should never regress.
            price_data_row = conn.execute("SELECT date FROM price_data WHERE ticker_id = ?", (ticker_id,)).fetchone()
            if price_data_row and price_data_row["date"] and str(price_data_row["date"]) != new_latest:
                if new_latest:
                    real_row = conn.execute(
                        """
                        SELECT date, open, high, low, close, volume FROM price_history_cache
                        WHERE ticker_id = ? AND interval = '1d' AND date = ?
                        """,
                        (ticker_id, new_latest),
                    ).fetchone()
                    if real_row:
                        conn.execute(
                            """
                            UPDATE price_data
                            SET date = ?, open = ?, high = ?, low = ?, close = ?, volume = ?
                            WHERE ticker_id = ?
                            """,
                            (
                                real_row["date"], real_row["open"], real_row["high"], real_row["low"],
                                real_row["close"], real_row["volume"], ticker_id,
                            ),
                        )
                else:
                    conn.execute("DELETE FROM price_data WHERE ticker_id = ?", (ticker_id,))

            if new_latest != old_latest:
                latest_date_changed_count += 1

        conn.commit()

    return {
        "dry_run": dry_run,
        "rows_deleted": rows_matched,
        "rows_matched": rows_matched,
        "tickers_affected": len(affected_ids),
        "tickers_latest_date_changed": latest_date_changed_count,
    }


def backfill_shares_outstanding(
    tickers: list[str] | None = None,
    *,
    force_refresh_all: bool = False,
    include_screener_fallback: bool = False,
    delay_seconds: float = SCREENER_DEFAULT_REQUEST_DELAY_SECONDS,
    limit: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fill market_cap.shares_outstanding, NSE shareholding-pattern XBRL first (official, free,
    bulk), screener.in as an opt-in fallback for tickers with no NSE listing at all (the BSE-only
    tail). Shared by scripts/backfill_shares_outstanding.py (thin CLI wrapper) and the sidebar's
    "Backfill shares outstanding" button — the one place this ever gets fetched from a provider,
    same role fill_price_cache_for_universe plays for price history.

    tickers=None (default): candidates are every active ticker missing a value, or still on the
    lower-quality screener-derived source (upgrade candidates) — unless force_refresh_all=True, in
    which case every active ticker is re-checked regardless of current value/source.
    include_screener_fallback=False (default): tickers NSE can't resolve are just left missing —
    screener.in's terms restrict bulk copying, so that path only runs when explicitly asked for.
    limit: cap the candidate list after derivation — for quick manual test runs, not used by the
    sidebar button.
    """
    if tickers is not None:
        candidates = list(tickers)
    elif force_refresh_all:
        candidates = list_all_instrument_master_tickers(active_only=True)
    else:
        candidates = list_tickers_missing_shares_outstanding(include_screener_sourced=True)
    if limit is not None:
        candidates = candidates[: max(0, limit)]

    def _report(phase: str, completed: int, total: int, **extra: Any) -> None:
        if progress_callback:
            progress_callback({"phase": phase, "completed": completed, "total": total, **extra})

    nse_result = _backfill_shares_outstanding_via_nse(candidates, delay_seconds=delay_seconds, report=_report)
    still_missing = nse_result.pop("still_missing")

    screener_result: dict[str, Any] = {
        "screener_candidate_count": 0,
        "screener_updated_count": 0,
        "screener_skipped_count": 0,
    }
    if still_missing and include_screener_fallback:
        screener_result = _backfill_shares_outstanding_via_screener(
            still_missing, delay_seconds=delay_seconds, report=_report
        )
        final_missing_count = screener_result["screener_skipped_count"]
    else:
        final_missing_count = len(still_missing)

    return {**nse_result, **screener_result, "still_missing_count": final_missing_count}


def _backfill_shares_outstanding_via_nse(
    candidates: list[str],
    *,
    delay_seconds: float,
    report: Callable[..., None],
) -> dict[str, Any]:
    """Phase A: resolve candidates from NSE's shareholding-pattern XBRL. .NS tickers match by
    bare symbol; .BO tickers match by ISIN cross-reference (shareholding pattern is company-wide,
    so an NSE filing correctly answers a BSE-format ticker's share count too, when the company is
    also NSE-listed)."""
    total = len(candidates)
    if total == 0:
        return {"nse_candidate_count": 0, "nse_resolved_count": 0, "nse_isin_cross_referenced_count": 0, "still_missing": []}

    session = nse_session()
    equities = fetch_shareholding_master(session, universe="equities")
    sme = fetch_shareholding_master(
        session,
        universe="sme",
        from_date=default_sme_lookback_from_date(),
        to_date=_current_market_datetime().date().strftime("%d-%m-%Y"),
    )
    by_symbol, by_isin = build_shareholding_index(equities + sme)

    resolved_via_symbol = 0
    resolved_via_isin = 0
    still_missing: list[str] = []

    for index, ticker in enumerate(candidates, start=1):
        record = None
        via_isin = False
        if ticker.endswith(".NS"):
            record = by_symbol.get(ticker[: -len(".NS")])
        elif ticker.endswith(".BO"):
            identity = lookup_instrument_identity(ticker) or {}
            isin = str(identity.get("isin") or "").strip().upper()
            if isin:
                record = by_isin.get(isin)
                via_isin = record is not None

        resolved = False
        if record is not None:
            xbrl_url = record.get("xbrl")
            shares = None
            if xbrl_url:
                try:
                    xr = requests.get(xbrl_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                    if xr.status_code == 200:
                        shares = extract_total_paid_up_shares(xr.text)
                except requests.RequestException:
                    shares = None
                if index < total:
                    time_module.sleep(delay_seconds)
            if shares and store_shares_outstanding(ticker, shares, source="nse_xbrl"):
                resolved = True
                if via_isin:
                    resolved_via_isin += 1
                else:
                    resolved_via_symbol += 1

        if not resolved:
            still_missing.append(ticker)
        report("shares_outstanding_nse", index, total, source="nse_xbrl")

    return {
        "nse_candidate_count": total,
        "nse_resolved_count": resolved_via_symbol + resolved_via_isin,
        "nse_isin_cross_referenced_count": resolved_via_isin,
        "still_missing": still_missing,
    }


def _backfill_shares_outstanding_via_screener(
    candidates: list[str],
    *,
    delay_seconds: float,
    report: Callable[..., None],
) -> dict[str, Any]:
    """Phase B (opt-in only): screener.in-derived fallback for tickers NSE can't resolve — in
    practice, tickers with no NSE listing at all (the BSE-only tail)."""
    total = len(candidates)
    session = requests.Session()
    updated = 0
    skipped = 0
    for index, ticker in enumerate(candidates, start=1):
        shares = fetch_screener_shares_outstanding(ticker, session=session)
        if shares and store_shares_outstanding(ticker, shares, source="screener_derived"):
            updated += 1
        else:
            skipped += 1
        report("shares_outstanding_screener", index, total, source="screener")
        if index < total:
            time_module.sleep(delay_seconds)
    return {
        "screener_candidate_count": total,
        "screener_updated_count": updated,
        "screener_skipped_count": skipped,
    }


def load_stock_universe_from_db(
    universe: str, *, refresh: bool = False, max_stocks: int | None = None, active_only: bool = True
) -> pd.DataFrame:
    """DB-backed replacement for load_stock_universe() — same UNIVERSE_COLUMNS-shaped output,
    sourced from instrument_master + security_classification + market_cap + live_metrics
    instead of re-parsing a CSV. Supports 'broad' via the in_nifty_total_market flag, so every
    universe type load_stock_universe() supports (except 'local', which never touched the CSVs
    anyway) is reconstructable from the DB.

    `refresh=True` mirrors load_stock_universe()'s own semantics: it triggers a live network
    refresh of the underlying universe CSV first (via the same universe.py refresh_* functions),
    which already self-syncs instrument_master/security_classification/market_cap afterward — so
    the DB read below reflects the fresh data.

    active_only=True (default) excludes tickers confirmed dormant by
    fill_price_cache_for_universe — this is the universe source for every sector/industry/
    breadth/RRG/crossover/top-gainers analytics function, so a dormant ticker's last (stale,
    possibly months old) cached price no longer gets compared against everything else as if it
    were current. The `active` column is still selected either way, so a caller that explicitly
    wants the full picture can pass active_only=False and filter the returned frame itself.
    """
    normalized = str(universe or "").strip().lower()
    if refresh:
        try:
            if normalized in {"broad", "nse_total_market", "nifty_total_market", "total_market"}:
                refresh_stock_universe()
            elif normalized in {"full_nse", "nse_full", "nse_equity"}:
                refresh_full_stock_universe(max_symbols=max_stocks)
            elif normalized in {"full_bse", "bse_full", "bse_equity"}:
                refresh_bse_stock_universe()
            elif normalized in {"all_india", "india", "nse_bse", "bse_nse"}:
                refresh_india_stock_universe()
        except Exception as exc:  # noqa: BLE001
            logger.info("Universe refresh before DB load failed for %s: %s", universe, exc)
    if normalized in {"broad", "nse_total_market", "nifty_total_market", "total_market"}:
        where = "im.in_nifty_total_market = 1"
    elif normalized in {"full_nse", "nse_full", "nse_equity"}:
        where = "im.market = 'IN' AND im.exchange IN ('NSE', 'NSE+BSE')"
    elif normalized in {"full_bse", "bse_full", "bse_equity"}:
        where = "im.market = 'IN' AND im.exchange IN ('BSE', 'NSE+BSE')"
    elif normalized in {"all_india", "india", "nse_bse", "bse_nse"}:
        where = "im.market = 'IN'"
    else:  # "local" or unrecognized — never CSV/DB-backed
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    if active_only:
        where += " AND im.active IS NOT 0"

    query = f"""
        SELECT
            im.ticker, im.symbol, im.name, im.isin,
            sc.sector, sc.industry, sc.basic_industry, sc.index_name,
            im.source, im.active, im.series,
            mc.free_float_market_cap,
            lm.last_price, lm.year_high, lm.year_low, lm.near_52w_high_pct,
            lm.return_7d_pct, lm.return_30d_pct, lm.return_90d_pct, lm.return_180d_pct, lm.return_365d_pct,
            COALESCE(mc.fetched_at, lm.computed_at, im.synced_at) AS refreshed_at,
            im.exchange, im.security_id, im.nse_ticker, im.bse_ticker, im.nse_security_id, im.bse_security_id,
            sc.data_quality, sc.classification_source
        FROM instrument_master im
        LEFT JOIN security_classification sc ON sc.ticker_id = im.ticker_id
        LEFT JOIN market_cap mc ON mc.ticker_id = im.ticker_id
        LEFT JOIN live_metrics lm ON lm.ticker_id = im.ticker_id
        WHERE {where}
    """
    if max_stocks is not None and max_stocks > 0:
        query += f" LIMIT {int(max_stocks)}"
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        rows = conn.execute(query).fetchall()
    if not rows:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    df = pd.DataFrame([dict(row) for row in rows])
    df["active"] = df["active"].map(lambda v: bool(v) if v is not None else None)
    return df[UNIVERSE_COLUMNS]


def _period_days(period: str) -> int | None:
    value = str(period or "").strip().lower()
    if value in {"max", "all"}:
        return None
    units = [
        ("months", 31),
        ("month", 31),
        ("weeks", 7),
        ("week", 7),
        ("years", 365),
        ("year", 365),
        ("days", 1),
        ("day", 1),
        ("mo", 31),
        ("wk", 7),
        ("yr", 365),
        ("w", 7),
        ("y", 365),
        ("m", 31),
        ("d", 1),
    ]
    for unit, multiplier in units:
        if value.endswith(unit):
            raw = value[: -len(unit)].strip()
            try:
                return max(1, int(float(raw) * multiplier))
            except ValueError:
                return None
    return None


def _cache_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _infer_market(ticker: str) -> str:
    """Classify a ticker as India ('IN': .NS/.BO suffix, or a '^'-prefixed India index/sector
    proxy like ^NSEI/^CNXIT used throughout this app) or US ('US', everything else)."""
    upper = str(ticker or "").strip().upper()
    if upper.endswith(".NS") or upper.endswith(".BO") or upper.startswith("^"):
        return "IN"
    return "US"


def _parse_cache_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=MARKET_TIMEZONE)
    return parsed


def get_price_cache_status(tickers: list[str] | tuple[str, ...] | None = None, interval: str = "1d") -> dict[str, Any]:
    """Return SQLite OHLCV cache coverage for optional tickers."""
    if not PRICE_CACHE_ENABLED:
        return {"enabled": False, "db_path": str(settings.db_path), "cached_ticker_count": 0, "row_count": 0}
    ticker_keys = [_cache_ticker(ticker) for ticker in tickers or [] if _cache_ticker(ticker)]
    params: list[Any] = [interval]
    ticker_filter = ""
    if ticker_keys:
        placeholders = ",".join("?" for _ in ticker_keys)
        ticker_filter = f" AND sl.ticker IN ({placeholders})"
        params.extend(ticker_keys)
    try:
        with _price_cache_connection() as conn:
            _ensure_price_cache_schema(conn)
            rows = conn.execute(
                f"""
                SELECT m.row_count, m.latest_date, m.fetched_at
                FROM price_cache_meta m
                JOIN instrument_master sl ON sl.ticker_id = m.ticker_id
                WHERE m.interval = ?{ticker_filter}
                """,
                params,
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "db_path": str(settings.db_path), "error": str(exc), "cached_ticker_count": 0, "row_count": 0}
    requested_count = len(ticker_keys) if ticker_keys else None
    cached_count = len(rows)
    fresh_count = sum(1 for row in rows if _cache_fetched_at_is_fresh(row["fetched_at"], interval))
    stale_count = max(0, cached_count - fresh_count)
    missing_count = max(0, requested_count - cached_count) if requested_count is not None else None
    row_count = sum(int(row["row_count"] or 0) for row in rows)
    latest_dates = [str(row["latest_date"]) for row in rows if row["latest_date"]]
    latest_date_counts = Counter(latest_dates)
    fetched_dates = [str(row["fetched_at"]) for row in rows if row["fetched_at"]]
    dominant_latest_date, dominant_latest_date_count = (None, 0)
    if latest_date_counts:
        dominant_latest_date, dominant_latest_date_count = latest_date_counts.most_common(1)[0]
    max_latest_date = max(latest_dates) if latest_dates else None
    return {
        "enabled": True,
        "db_path": str(settings.db_path),
        "interval": interval,
        "requested_ticker_count": requested_count,
        "cached_ticker_count": cached_count,
        "fresh_ticker_count": fresh_count,
        "stale_ticker_count": stale_count,
        "missing_ticker_count": missing_count,
        "coverage_pct": round(100 * cached_count / requested_count, 2) if requested_count else None,
        "fresh_coverage_pct": round(100 * fresh_count / requested_count, 2) if requested_count else None,
        "row_count": row_count,
        "latest_price_date": max_latest_date,
        "latest_price_date_count": latest_date_counts.get(max_latest_date, 0) if max_latest_date else 0,
        "dominant_latest_price_date": dominant_latest_date,
        "dominant_latest_price_date_count": dominant_latest_date_count,
        "latest_price_date_distribution": dict(sorted(latest_date_counts.items(), reverse=True)),
        "latest_fetched_at": max(fetched_dates) if fetched_dates else None,
        "max_age_hours": PRICE_CACHE_MAX_AGE_HOURS,
    }


def refresh_latest_exchange_eod_cache(
    tickers: list[str] | tuple[str, ...],
    *,
    interval: str = "1d",
    lookback_days: int | None = None,
) -> dict[str, Any]:
    """Fetch the latest official NSE/BSE bhavcopy rows and persist them locally.

    This is intentionally separate from historical Yahoo/Stooq warming. It gives
    the app the latest exchange EOD candle first, while longer history can still
    be filled by the broader cache warmer.
    """
    unique_tickers = list(dict.fromkeys(_cache_ticker(ticker) for ticker in tickers if _cache_ticker(ticker)))
    if not unique_tickers:
        return {
            "requested_ticker_count": 0,
            "available_ticker_count": 0,
            "missing_ticker_count": 0,
            "providers": {},
            "trade_dates": {},
        }
    if str(interval).strip().lower() != "1d":
        return {
            "requested_ticker_count": len(unique_tickers),
            "available_ticker_count": 0,
            "missing_ticker_count": len(unique_tickers),
            "providers": {},
            "trade_dates": {},
            "warnings": ["Exchange bhavcopy refresh only supports 1d interval."],
        }

    rows = get_latest_exchange_eod_rows(unique_tickers, lookback_days=lookback_days)
    providers: dict[str, int] = {}
    trade_dates: dict[str, int] = {}
    for ticker, frame in rows.items():
        if frame.empty:
            continue
        provider = str(frame.attrs.get("provider") or "exchange_eod")
        trade_date = str(frame.attrs.get("exchange_trade_date") or "")
        providers[provider] = providers.get(provider, 0) + 1
        if trade_date:
            trade_dates[trade_date] = trade_dates.get(trade_date, 0) + 1
        _store_price_history(ticker, interval, frame)

    missing = [ticker for ticker in unique_tickers if ticker not in rows]
    return {
        "requested_ticker_count": len(unique_tickers),
        "available_ticker_count": len(rows),
        "missing_ticker_count": len(missing),
        "providers": providers,
        "trade_dates": dict(sorted(trade_dates.items(), reverse=True)),
        "latest_trade_date": max(trade_dates) if trade_dates else None,
        "missing_tickers": missing[:50],
        "cache_status": get_price_cache_status(unique_tickers, interval=interval),
    }


def backfill_bhavcopy_history(
    tickers: list[str] | tuple[str, ...],
    *,
    start_date: str,
    end_date: str | None = None,
    interval: str = "1d",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Backfill price_history_cache from official NSE/BSE bhavcopy archives, day by day.

    For tickers Yahoo has no data for (typically SME/illiquid listings the exchanges'
    own bhavcopy still carries in full — see get_exchange_eod_rows_for_date's SM/ST/BE/BZ
    series handling), this walks every trading day in [start_date, end_date] instead of
    diffing against price_cache_meta, since these tickers have no meta row to diff against.

    Cost scales with the number of trading days requested, not the ticker count: each day is
    one NSE + one BSE bhavcopy fetch (the whole exchange), filtered in-memory for the requested
    tickers, not one fetch per ticker.
    """
    if str(interval).strip().lower() != "1d":
        return {"requested_ticker_count": 0, "warnings": ["Bhavcopy backfill only supports 1d interval."]}

    unique_tickers = list(dict.fromkeys(_cache_ticker(ticker) for ticker in tickers if _cache_ticker(ticker)))
    if not unique_tickers:
        return {
            "requested_ticker_count": 0,
            "trading_days_scanned": 0,
            "tickers_with_data_count": 0,
            "tickers_without_data_count": 0,
        }

    start = datetime.strptime(str(start_date), "%Y-%m-%d").date()
    end = datetime.strptime(str(end_date), "%Y-%m-%d").date() if end_date else _current_market_datetime().date()

    trade_dates: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            trade_dates.append(cursor)
        cursor += timedelta(days=1)

    frames_by_ticker: dict[str, list[pd.DataFrame]] = {ticker: [] for ticker in unique_tickers}
    provider_counts: dict[str, int] = {}
    days_with_data = 0

    for index, trade_date in enumerate(trade_dates):
        day_rows = get_exchange_eod_rows_for_date(unique_tickers, trade_date)
        # Connectivity signal is deliberately independent of whether OUR tickers matched --
        # see bhavcopy_fetch_succeeded_for_date's docstring. A day where the exchange responded
        # but none of our (possibly all-dormant) candidates traded must still count as "covered".
        if bhavcopy_fetch_succeeded_for_date(trade_date):
            days_with_data += 1
        for ticker, frame in day_rows.items():
            if frame is None or frame.empty:
                continue
            frames_by_ticker[ticker].append(frame)
            provider = str(frame.attrs.get("provider") or "exchange_eod")
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        if progress_callback:
            progress_callback(
                {
                    "phase": "scanning_days",
                    "source": "bhavcopy",
                    "completed": index + 1,
                    "total": len(trade_dates),
                    "ticker_count": len(unique_tickers),
                }
            )

    stored_count = 0
    live_metrics_refreshed_count = 0
    for ticker, day_frames in frames_by_ticker.items():
        if day_frames:
            combined = pd.concat(day_frames, ignore_index=True)
            combined.attrs["provider"] = str(day_frames[-1].attrs.get("provider") or "exchange_eod_bhavcopy")
            if _store_price_history(ticker, interval, combined):
                live_metrics_refreshed_count += 1
            stored_count += 1

    no_data_tickers = [ticker for ticker, day_frames in frames_by_ticker.items() if not day_frames]
    return {
        "requested_ticker_count": len(unique_tickers),
        "trading_days_scanned": len(trade_dates),
        "trading_days_with_bhavcopy_data": days_with_data,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "tickers_with_data_count": stored_count,
        "tickers_without_data_count": len(no_data_tickers),
        "no_data_tickers": no_data_tickers[:50],
        "providers": provider_counts,
        "live_metrics_refreshed_count": live_metrics_refreshed_count,
        "cache_status": get_price_cache_status(unique_tickers, interval=interval),
    }


def _tickers_without_price_cache(
    *, interval: str = "1d", active_only: bool = True, tickers: list[str] | None = None
) -> list[str]:
    """instrument_master tickers with no price_cache_meta row at all for this interval —
    i.e. every prior fetch attempt (Yahoo or bhavcopy) has come up completely empty for them.
    Optionally scoped to a given ticker list rather than the whole table."""
    where_active = "im.active IS NOT 0" if active_only else "1=1"
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        if tickers is not None:
            unique = list(dict.fromkeys(_cache_ticker(t) for t in tickers if _cache_ticker(t)))
            if not unique:
                return []
            placeholders = ",".join("?" for _ in unique)
            rows = conn.execute(
                f"""
                SELECT im.ticker AS ticker
                FROM instrument_master im
                LEFT JOIN price_cache_meta m ON m.ticker_id = im.ticker_id AND m.interval = ?
                WHERE m.ticker_id IS NULL AND {where_active} AND im.ticker IN ({placeholders})
                """,
                [interval, *unique],
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT im.ticker AS ticker
                FROM instrument_master im
                LEFT JOIN price_cache_meta m ON m.ticker_id = im.ticker_id AND m.interval = ?
                WHERE m.ticker_id IS NULL AND {where_active}
                """,
                [interval],
            ).fetchall()
    return [row["ticker"] for row in rows]


def _tickers_still_short_of_start_date(tickers: list[str], *, start_date: str, interval: str) -> list[str]:
    """Which of these tickers still have earliest_date > start_date right now, and aren't
    already exhausted for this exact start_date. Deliberately DB-truth-based rather than
    tracking "did the Yahoo fetch report success or failure" -- a Yahoo attempt can "succeed"
    while returning almost nothing (e.g. one stray trading day for an illiquid ticker), which
    still leaves the ticker genuinely short. Relying on the fetch's own success/failure signal
    was exactly the gap that let such tickers cycle through full_fetch forever, always
    "succeeding" against a source that never has enough to satisfy them."""
    unique = list(dict.fromkeys(_cache_ticker(t) for t in tickers if _cache_ticker(t)))
    if not unique:
        return []
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique)
        rows = conn.execute(
            f"""
            SELECT im.ticker AS ticker
            FROM instrument_master im JOIN price_cache_meta m ON m.ticker_id = im.ticker_id
            WHERE m.interval = ? AND im.ticker IN ({placeholders})
              AND m.earliest_date > ?
              AND (m.backfill_exhausted_start_date IS NULL OR m.backfill_exhausted_start_date != ?)
            """,
            [interval, *unique, start_date, start_date],
        ).fetchall()
    return [row["ticker"] for row in rows]


def _tickers_stale_beyond(tickers: list[str], *, cutoff_date: str, interval: str) -> list[str]:
    """Which of these tickers have latest_date older than cutoff_date right now. Used to find
    forward-fill tickers that are stuck, not just a day or two behind (see
    settings.forward_fetch_stale_days) -- unlike the backward-fill exhausted-marker, there's no
    permanent "confirmed" state here (a ticker could resume trading any day), so this is just a
    plain DB-truth check, re-evaluated fresh each call rather than cached."""
    unique = list(dict.fromkeys(_cache_ticker(t) for t in tickers if _cache_ticker(t)))
    if not unique:
        return []
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique)
        rows = conn.execute(
            f"""
            SELECT im.ticker AS ticker
            FROM instrument_master im JOIN price_cache_meta m ON m.ticker_id = im.ticker_id
            WHERE m.interval = ? AND im.ticker IN ({placeholders}) AND m.latest_date < ?
            """,
            [interval, *unique, cutoff_date],
        ).fetchall()
    return [row["ticker"] for row in rows]


def _min_latest_date(tickers: list[str], *, interval: str) -> str | None:
    """The earliest price_cache_meta.latest_date among these tickers, or None if none have a
    meta row. Used to narrow Phase C's bhavcopy day-walk range when every candidate this run is
    forward-stale-origin -- each already has solid history up to its own latest_date, so only
    latest_date onward is new ground; walking all the way back to effective_start_date for such a
    batch would just re-confirm history that's already known good."""
    unique = list(dict.fromkeys(_cache_ticker(t) for t in tickers if _cache_ticker(t)))
    if not unique:
        return None
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique)
        row = conn.execute(
            f"""
            SELECT MIN(m.latest_date) AS min_latest_date
            FROM instrument_master im JOIN price_cache_meta m ON m.ticker_id = im.ticker_id
            WHERE m.interval = ? AND im.ticker IN ({placeholders}) AND m.latest_date IS NOT NULL
            """,
            [interval, *unique],
        ).fetchone()
    return str(row["min_latest_date"]) if row and row["min_latest_date"] else None


def _count_tickers_pending_dormancy_check(
    tickers: list[str], *, stale_cutoff_date: str, expected_trading_day: str, interval: str
) -> int:
    """How many of these tickers are currently behind the latest expected trading day but haven't
    yet gone stale_cutoff_date days without a trade -- i.e. still within the normal "a few days
    behind is fine for a thinly-traded stock" window, not yet eligible for the Phase C dormancy
    check. Surfaced in the completion message so a recurring forward-fetch/failed count reads as
    "known, converging population" rather than "stuck forever"."""
    unique = list(dict.fromkeys(_cache_ticker(t) for t in tickers if _cache_ticker(t)))
    if not unique:
        return 0
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique)
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM instrument_master im JOIN price_cache_meta m ON m.ticker_id = im.ticker_id
            WHERE m.interval = ? AND im.ticker IN ({placeholders})
              AND m.latest_date < ? AND m.latest_date >= ?
            """,
            [interval, *unique, expected_trading_day, stale_cutoff_date],
        ).fetchone()
    return int(row["cnt"] or 0)


def _mark_tickers_dormant(tickers: list[str]) -> int:
    """Set instrument_master.active = 0 for the given tickers. Used when neither Yahoo nor a
    full bhavcopy day-walk of the configured history window found a single trade for a ticker —
    list_all_instrument_master_tickers(active_only=True) then excludes it from future refreshes."""
    unique = list(dict.fromkeys(_cache_ticker(t) for t in tickers if _cache_ticker(t)))
    if not unique:
        return 0
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique)
        conn.execute(f"UPDATE instrument_master SET active = 0 WHERE ticker IN ({placeholders})", unique)
        conn.commit()
    return len(unique)


def _mark_backfill_exhausted(tickers: list[str], *, start_date: str, interval: str) -> int:
    """Record that a complete bhavcopy day-walk for [start_date, today] already confirmed these
    tickers have nothing earlier than their current earliest_date. fill_price_cache_for_universe
    then stops re-attempting their backward-fill on future runs, for this exact start_date --
    past history can't change, so retrying is pure waste. Only call this after a bhavcopy pass
    whose day-coverage was actually trustworthy (see min_bhavcopy_coverage_pct); a scan that
    might have missed real trading days due to an outage must never produce this marker.
    Only affects rows still short of start_date -- a ticker that happened to get backfilled by
    something else in the meantime is left alone."""
    unique = list(dict.fromkeys(_cache_ticker(t) for t in tickers if _cache_ticker(t)))
    if not unique:
        return 0
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique)
        cursor = conn.execute(
            f"""
            UPDATE price_cache_meta
            SET backfill_exhausted_start_date = ?
            WHERE interval = ?
              AND earliest_date > ?
              AND ticker_id IN (SELECT ticker_id FROM instrument_master WHERE ticker IN ({placeholders}))
            """,
            [start_date, interval, start_date, *unique],
        )
        conn.commit()
        updated = cursor.rowcount if cursor.rowcount is not None else 0
    return max(0, updated)


def _most_recent_expected_trading_day() -> str:
    """The latest calendar date price_cache_meta.latest_date should have reached, ignoring
    market holidays (only weekends are excluded). Before the daily close buffer, "today" isn't
    finished yet, so the expectation is yesterday (or the prior weekday)."""
    now = _current_market_datetime()
    expected = now.date()
    if now.time() < DAILY_MARKET_CLOSE_BUFFER:
        expected -= timedelta(days=1)
    while expected.weekday() >= 5:
        expected -= timedelta(days=1)
    return expected.isoformat()


def _fetch_and_store_range(chunk: list[str], start: str, end: str, interval: str) -> tuple[dict[str, pd.DataFrame], int]:
    """Fetch OHLCV for a chunk of tickers over [start, end) from Yahoo, overlay the latest
    official NSE/BSE EOD row, and persist. Only called from fill_price_cache_for_universe —
    the one place price history is ever fetched from a provider.

    Returns (fetched, live_metrics_refreshed_count) -- the second element is how many of this
    chunk's tickers had live_metrics recomputed (i.e. reached a new latest_date), surfaced up
    through fill_price_cache_for_universe's progress_callback for sidebar visibility."""
    try:
        raw = yf.download(
            tickers=chunk,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=30,
            group_by="ticker",
        )
    except Exception as exc:
        logger.warning("Batch price history fetch failed for %s tickers (%s to %s): %s", len(chunk), start, end, exc)
        return {}, 0
    fetched = _split_yahoo_batch_history(raw, chunk, interval=interval)
    fetched = _overlay_exchange_latest_rows(fetched, interval=interval)
    live_metrics_refreshed_count = 0
    for ticker, frame in fetched.items():
        if _store_price_history(ticker, interval, frame):
            live_metrics_refreshed_count += 1
    return fetched, live_metrics_refreshed_count


def fill_price_cache_for_universe(
    tickers: list[str] | tuple[str, ...],
    *,
    interval: str = "1d",
    start_date: str | None = None,
    chunk_size: int = 80,
    retry_attempts: int = 2,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Incrementally fill price_history_cache for every given ticker.

    This is the one place price history is ever fetched from a provider and written to
    price_history_cache/price_cache_meta/price_data/live_metrics. Everywhere else in the app
    (get_price_history, get_price_histories, and everything built on them) only reads whatever
    is cached here. Triggered from the sidebar's "Refresh price cache" button and the
    daily_refresh CLI/cron — no MCP tool ever triggers a live fetch.

    Fetch order (bhavcopy is authoritative and free -- one exchange-wide file per day, cost
    independent of ticker count -- so it's preferred wherever it's cheap to check; Yahoo is
    preferred for a bulk multi-year pull when it's likely to work, with bhavcopy as its fallback):

      0. Priority pass (ALL requested tickers, every call): walk the last
         bhavcopy_priority_window_days of official NSE/BSE bhavcopy first, before anything else.
         INSERT OR REPLACE means any date bhavcopy has here overwrites whatever was cached for it
         before (including a prior Yahoo-sourced row) -- this is what makes a previous run's
         Yahoo-filled "latest day" get superseded by bhavcopy on the very next refresh, once
         bhavcopy has published it.

    Per ticker, classified against price_cache_meta as it stands *after* the priority pass:
      - no price_cache_meta row at all -> full-range Yahoo fetch from start_date to today.
      - cached earliest_date is after start_date (the configured floor moved earlier than
        what's cached) -> backward-fill Yahoo fetch from start_date to the existing earliest_date
        -- UNLESS a complete bhavcopy day-walk already confirmed, for this exact start_date, that
        nothing exists before earliest_date (backfill_exhausted_start_date matches), in which
        case backward-fill is skipped entirely: past history can't change, so re-attempting a
        confirmed-empty range is pure waste. Automatically re-eligible if start_date ever moves
        earlier than what was checked.
      - cached latest_date is behind the most recent expected trading day -> forward Yahoo fetch
        from latest_date (inclusive, to safely re-cover a possibly-incomplete last row) to today.
      - needs both backward and forward -> treated as a full-range fetch (start_date to today);
        INSERT OR REPLACE makes re-covering the already-cached middle harmless, and this
        combination is rare (only right after preponing start_date while also being behind on
        the latest day).
      - neither -> skipped, zero network calls.

      Fallback pass: two INDEPENDENT bhavcopy confirmation walks, each with its own day-coverage
      guard -- deliberately not one combined walk, since min_bhavcopy_coverage_pct is a per-walk
      connectivity signal, and bundling both groups together would let one group's noisy walk
      block the other's otherwise-clean confirmation (confirmed live: a batch of forward-stale
      tickers stuck behind a bundled run's coverage dip came back clean and were marked
      immediately once walked in isolation).
        (a) backward-origin: any full_fetch/backward_fetch ticker Yahoo still failed on, plus any
            that "succeeded" but is still short of start_date (a Yahoo attempt can succeed while
            returning almost nothing) -- walked over the full [start_date, today], since these
            genuinely need that whole range checked. A ticker with zero trades anywhere in it is
            marked instrument_master.active=0 (dormant); one that still falls short of start_date
            gets backfill_exhausted_start_date recorded (see above).
        (b) forward-stale: any forward_fetch ticker still more than forward_fetch_stale_days
            behind after its attempt (a ticker merely a day or two behind is skipped here -- the
            priority pass just tried the same recent window moments earlier, and that's what
            forward_fetch is normally for) -- always walked from the earliest of these candidates'
            own latest_date onward, never the full start_date range, since each already has solid
            history up to that point. A candidate with nothing newer than its already-cached
            latest_date is marked dormant.
      Each pass is independently guarded by its own bhavcopy day-coverage check, so a transient
      NSE/BSE outage during either walk can't be misread as mass delisting or a confirmed
      historical floor for that pass -- without holding the other pass hostage to it.

      Note: a Yahoo "success" is only trusted when the returned row has real volume -- Yahoo
      regularly carries a thinly-traded security's last known close forward as a synthetic
      zero-volume row for days it didn't actually trade, instead of returning nothing, and
      _store_price_history discards those before they can masquerade as fresh data (verified
      against bhavcopy: e.g. a real trade with volume on one day, flanked by identical
      zero-volume Yahoo "closes" on the surrounding days).
    """
    effective_start_date = str(start_date or settings.price_history_start_date).strip()
    unique_tickers = list(dict.fromkeys(_cache_ticker(ticker) for ticker in tickers if _cache_ticker(ticker)))
    total = len(unique_tickers)
    if not unique_tickers:
        return {
            "requested_ticker_count": 0,
            "skipped_up_to_date_count": 0,
            "full_fetch_count": 0,
            "backward_fetch_count": 0,
            "forward_fetch_count": 0,
            "bhavcopy_fallback_recovered_count": 0,
            "dormant_marked_count": 0,
            "dormant_tickers": [],
            "backfill_exhausted_marked_count": 0,
            "backfill_exhausted_skipped_count": 0,
            "forward_fetch_pending_dormancy_check_count": 0,
            "forward_fetch_stale_days": int(settings.forward_fetch_stale_days),
            "market_cap_recomputed_count": 0,
            "live_metrics_updated_count": 0,
            "failed_count": 0,
            "failed_tickers": [],
            "start_date": effective_start_date,
        }

    def _wrapped_progress(stage: str) -> Callable[[dict[str, Any]], None] | None:
        if not progress_callback:
            return None

        def _inner(update: dict[str, Any]) -> None:
            tagged = dict(update)
            tagged["phase"] = f"{stage}_{update.get('phase', '')}"
            progress_callback(tagged)

        return _inner

    today_date = _current_market_datetime().date()
    expected_trading_day = _most_recent_expected_trading_day()
    end_exclusive = (today_date + timedelta(days=1)).isoformat()

    # Running total of tickers whose live_metrics got recomputed this run, across every phase
    # (this priority pass, the Yahoo phases below, and Phase A/Phase B bhavcopy walks) -- surfaced
    # live via progress_callback for sidebar visibility, and returned as live_metrics_updated_count.
    live_metrics_updated_total = 0

    priority_window_days = max(0, int(settings.bhavcopy_priority_window_days))
    if priority_window_days > 0 and interval == "1d":
        priority_start = (today_date - timedelta(days=priority_window_days)).isoformat()
        priority_result = backfill_bhavcopy_history(
            unique_tickers,
            start_date=priority_start,
            end_date=today_date.isoformat(),
            interval=interval,
            progress_callback=_wrapped_progress("bhavcopy_priority"),
        )
        live_metrics_updated_total += int(priority_result.get("live_metrics_refreshed_count") or 0)

    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique_tickers)
        meta_rows = conn.execute(
            f"""
            SELECT im.ticker AS ticker, m.latest_date AS latest_date, m.earliest_date AS earliest_date,
                   m.backfill_exhausted_start_date AS backfill_exhausted_start_date
            FROM instrument_master im
            LEFT JOIN price_cache_meta m ON m.ticker_id = im.ticker_id AND m.interval = ?
            WHERE im.ticker IN ({placeholders})
            """,
            [interval, *unique_tickers],
        ).fetchall()
    meta_by_ticker = {row["ticker"]: row for row in meta_rows}

    full_fetch: list[str] = []
    backward_by_date: dict[str, list[str]] = {}
    forward_by_date: dict[str, list[str]] = {}
    skipped_count = 0
    backfill_exhausted_skipped_count = 0
    for ticker in unique_tickers:
        meta = meta_by_ticker.get(ticker)
        latest_date = str(meta["latest_date"]) if meta and meta["latest_date"] else None
        if latest_date is None:
            full_fetch.append(ticker)
            continue
        earliest_date = str(meta["earliest_date"]) if meta["earliest_date"] else latest_date
        needs_forward = latest_date < expected_trading_day
        # A complete bhavcopy day-walk already confirmed, for this exact start date, that nothing
        # exists before earliest_date -- past history doesn't change, so don't keep re-attempting
        # a backward-fill that can only ever fail again. Re-eligible automatically if
        # effective_start_date ever moves earlier than what was checked (the marker just won't
        # match any more).
        exhausted_for_current_start = (
            meta["backfill_exhausted_start_date"] == effective_start_date if meta else False
        )
        needs_backward = earliest_date > effective_start_date and not exhausted_for_current_start
        if needs_backward is False and earliest_date > effective_start_date:
            backfill_exhausted_skipped_count += 1
        if needs_forward and needs_backward:
            full_fetch.append(ticker)
        elif needs_backward:
            backward_by_date.setdefault(earliest_date, []).append(ticker)
        elif needs_forward:
            forward_by_date.setdefault(latest_date, []).append(ticker)
        else:
            skipped_count += 1

    chunk_size = max(1, int(chunk_size or 80))
    attempts = max(1, int(retry_attempts) + 1)
    completed = skipped_count
    fetched_counts = {"full_fetch": 0, "backward_fetch": 0, "forward_fetch": 0}
    failed: list[str] = []

    def _report(phase: str, **extra: Any) -> None:
        if progress_callback:
            progress_callback(
                {
                    "phase": phase,
                    "source": "yfinance",
                    "completed": completed,
                    "total": total,
                    "live_metrics_updated_total": live_metrics_updated_total,
                    **extra,
                }
            )

    _report("classifying")

    def _fetch_group(phase: str, group_tickers: list[str], start: str, end: str) -> None:
        # completed/total (above) only move when a ticker actually succeeds -- a chunk that
        # fails outright burns a whole retry cycle (real network time) with zero visible change,
        # which reads as "stuck" from the UI. attempt/chunk below move on every single network
        # call regardless of outcome, so progress stays visibly alive through a bad stretch.
        nonlocal completed, live_metrics_updated_total
        remaining = list(group_tickers)
        for attempt_index in range(attempts):
            if not remaining:
                break
            chunks_this_attempt = max(1, (len(remaining) + chunk_size - 1) // chunk_size)
            still_remaining: list[str] = []
            for chunk_index, chunk_start in enumerate(range(0, len(remaining), chunk_size), start=1):
                chunk = remaining[chunk_start : chunk_start + chunk_size]
                fetched, live_metrics_count = _fetch_and_store_range(chunk, start, end, interval)
                live_metrics_updated_total += live_metrics_count
                for ticker in chunk:
                    if ticker in fetched:
                        fetched_counts[phase] += 1
                        completed += 1
                    else:
                        still_remaining.append(ticker)
                _report(
                    phase,
                    attempt=attempt_index + 1,
                    attempts_total=attempts,
                    chunk=chunk_index,
                    chunks_total=chunks_this_attempt,
                )
            remaining = still_remaining
        failed.extend(remaining)

    forward_fetch_tickers = {ticker for group in forward_by_date.values() for ticker in group}

    if full_fetch:
        _fetch_group("full_fetch", full_fetch, effective_start_date, end_exclusive)
    for earliest_date, group_tickers in backward_by_date.items():
        _fetch_group("backward_fetch", group_tickers, effective_start_date, earliest_date)
    for latest_date, group_tickers in forward_by_date.items():
        _fetch_group("forward_fetch", group_tickers, latest_date, end_exclusive)

    # Fallback pass: full_fetch/backward_fetch failures need this -- the priority pass already
    # just tried the recent window for every ticker (including forward_fetch failures) moments
    # ago, so retrying those via bhavcopy again wouldn't find anything new *this run*. Also
    # includes:
    #   - any full_fetch/backward_fetch-origin ticker that "succeeded" against Yahoo but whose
    #     earliest date still doesn't reach effective_start_date (see
    #     _tickers_still_short_of_start_date) -- otherwise a ticker Yahoo only ever returns one
    #     stray trading day for "succeeds" forever and never gets confirmed exhausted.
    #   - any forward_fetch-origin ticker still more than forward_fetch_stale_days behind after
    #     its attempt (see _tickers_stale_beyond) -- a ticker that's merely a day or two behind
    #     is normal (that's what forward_fetch is *for*), but one stuck for weeks despite every
    #     priority pass and forward_fetch attempt in between is worth a real confirmation: if a
    #     full day-walk finds nothing newer either, it's marked dormant instead of retried
    #     forever. Unlike the backward-fill exhausted-marker, there's no permanent "confirmed"
    #     state for this -- a still-active ticker that just hasn't traded recently stays a
    #     candidate every run until it either gets fresh data or is confirmed dormant.
    bhavcopy_fallback_recovered_count = 0
    dormant_marked_count = 0
    dormant_tickers: list[str] = []
    backfill_exhausted_marked_count = 0
    backward_origin_tickers = full_fetch + [ticker for group in backward_by_date.values() for ticker in group]
    still_short_after_yahoo = _tickers_still_short_of_start_date(
        backward_origin_tickers, start_date=effective_start_date, interval=interval
    )
    forward_stale_cutoff = (today_date - timedelta(days=max(1, int(settings.forward_fetch_stale_days)))).isoformat()
    forward_stale_candidates = _tickers_stale_beyond(
        list(forward_fetch_tickers), cutoff_date=forward_stale_cutoff, interval=interval
    )
    backward_origin_candidates = [ticker for ticker in failed if ticker not in forward_fetch_tickers] + still_short_after_yahoo

    newly_dormant: set[str] = set()
    newly_recovered: set[str] = set()

    # Two independent confirmation passes, deliberately not one combined walk+coverage-check --
    # min_bhavcopy_coverage_pct is a per-walk signal (was the exchange reachable for most of the
    # days THIS walk scanned), so bundling both candidate groups into one call means one group's
    # connectivity noise decides both groups' fate. A backward-origin candidate forces the walk
    # back to the full effective_start_date range; if even a handful of those ~700 scanned days
    # have a hiccup, the shared percentage can dip below threshold and silently skip marking for
    # an otherwise-clean forward-stale batch too (confirmed live: 55 forward-stale tickers stuck
    # in a bundled run came back 94-100% coverage and marked dormant immediately once re-checked
    # in isolation). Splitting them means each group's confirmation only depends on its own walk.
    if backward_origin_candidates and interval == "1d":
        fallback_result_a = backfill_bhavcopy_history(
            backward_origin_candidates,
            start_date=effective_start_date,
            end_date=today_date.isoformat(),
            interval=interval,
            progress_callback=_wrapped_progress("bhavcopy_fallback"),
        )
        scanned_a = int(fallback_result_a.get("trading_days_scanned") or 0)
        covered_a = int(fallback_result_a.get("trading_days_with_bhavcopy_data") or 0)
        coverage_pct_a = round(100 * covered_a / scanned_a, 2) if scanned_a else 0.0
        live_metrics_updated_total += int(fallback_result_a.get("live_metrics_refreshed_count") or 0)

        still_missing_a = set(
            _tickers_without_price_cache(interval=interval, active_only=True, tickers=backward_origin_candidates)
        )
        newly_recovered |= {ticker for ticker in backward_origin_candidates if ticker not in still_missing_a}

        if coverage_pct_a >= settings.min_bhavcopy_coverage_pct:
            if still_missing_a:
                dormant_marked_count += _mark_tickers_dormant(list(still_missing_a))
                newly_dormant |= still_missing_a
            # A trustworthy, complete day-walk just ran for every backward-origin candidate here --
            # whichever still have a meta row (i.e. weren't just marked dormant above) but still
            # fall short of effective_start_date are now confirmed, not just "not done yet". The
            # UPDATE inside only touches rows that both exist and still fall short, so passing the
            # full candidate list (including any just-dormant-marked ones, which have no meta row
            # to match) is safe.
            backfill_exhausted_marked_count = _mark_backfill_exhausted(
                backward_origin_candidates, start_date=effective_start_date, interval=interval
            )

    if forward_stale_candidates and interval == "1d":
        # Each forward-stale candidate already has solid history up to its own latest_date --
        # only latest_date onward is new ground, so this walk is always narrowed to the earliest
        # of those, independent of whatever range Pass A needed this run.
        narrowed_start = _min_latest_date(forward_stale_candidates, interval=interval) or effective_start_date
        fallback_result_b = backfill_bhavcopy_history(
            forward_stale_candidates,
            start_date=narrowed_start,
            end_date=today_date.isoformat(),
            interval=interval,
            progress_callback=_wrapped_progress("bhavcopy_fallback"),
        )
        scanned_b = int(fallback_result_b.get("trading_days_scanned") or 0)
        covered_b = int(fallback_result_b.get("trading_days_with_bhavcopy_data") or 0)
        coverage_pct_b = round(100 * covered_b / scanned_b, 2) if scanned_b else 0.0
        live_metrics_updated_total += int(fallback_result_b.get("live_metrics_refreshed_count") or 0)

        still_missing_b = set(
            _tickers_without_price_cache(interval=interval, active_only=True, tickers=forward_stale_candidates)
        )
        newly_recovered |= {ticker for ticker in forward_stale_candidates if ticker not in still_missing_b}

        if coverage_pct_b >= settings.min_bhavcopy_coverage_pct:
            # The walk above already covered every forward-stale candidate's own latest_date
            # onward -- if it found nothing newer than what was already cached, that's
            # authoritative: no trades since latest_date. Confirmed dormant, not just "still behind".
            still_forward_stale = _tickers_stale_beyond(
                forward_stale_candidates, cutoff_date=forward_stale_cutoff, interval=interval
            )
            if still_forward_stale:
                dormant_marked_count += _mark_tickers_dormant(still_forward_stale)
                newly_dormant |= set(still_forward_stale)

    if newly_recovered:
        bhavcopy_fallback_recovered_count = len(newly_recovered)
        failed = [ticker for ticker in failed if ticker not in newly_recovered]
    if newly_dormant:
        dormant_tickers = sorted(newly_dormant)[:50]
        failed = [ticker for ticker in failed if ticker not in newly_dormant]

    # Free, local, no network: any ticker with a shares_outstanding on file (from the separate,
    # occasional screener.in backfill script) gets its market_cap refreshed from today's close.
    market_cap_recompute = recompute_market_cap_from_shares_outstanding()

    forward_fetch_pending_dormancy_check_count = _count_tickers_pending_dormancy_check(
        unique_tickers,
        stale_cutoff_date=forward_stale_cutoff,
        expected_trading_day=expected_trading_day,
        interval=interval,
    )

    return {
        "requested_ticker_count": total,
        "skipped_up_to_date_count": skipped_count,
        "full_fetch_count": fetched_counts["full_fetch"],
        "backward_fetch_count": fetched_counts["backward_fetch"],
        "forward_fetch_count": fetched_counts["forward_fetch"],
        "bhavcopy_fallback_recovered_count": bhavcopy_fallback_recovered_count,
        "dormant_marked_count": dormant_marked_count,
        "dormant_tickers": dormant_tickers,
        "backfill_exhausted_marked_count": backfill_exhausted_marked_count,
        "backfill_exhausted_skipped_count": backfill_exhausted_skipped_count,
        "forward_fetch_pending_dormancy_check_count": forward_fetch_pending_dormancy_check_count,
        "forward_fetch_stale_days": int(settings.forward_fetch_stale_days),
        "market_cap_recomputed_count": market_cap_recompute.get("recomputed_ticker_count", 0),
        "live_metrics_updated_count": live_metrics_updated_total,
        "failed_count": len(failed),
        "failed_tickers": failed[:50],
        "start_date": effective_start_date,
        "cache_status": get_price_cache_status(unique_tickers, interval=interval),
    }


def _split_yahoo_batch_history(raw: pd.DataFrame, tickers: list[str], *, interval: str = "1d") -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
        frame = _normalize_yahoo_history_frame(raw)
        frame.attrs["provider"] = "yfinance_batch"
        cleaned = _drop_incomplete_price_rows(frame, tickers[0], interval=interval)
        if not cleaned.empty:
            frames[tickers[0]] = cleaned
        return frames

    if not isinstance(raw.columns, pd.MultiIndex):
        return frames

    level0 = {str(value) for value in raw.columns.get_level_values(0)}
    level1 = {str(value) for value in raw.columns.get_level_values(1)}
    ticker_first = any(ticker in level0 for ticker in tickers)
    for ticker in tickers:
        try:
            item = raw[ticker] if ticker_first else raw.xs(ticker, axis=1, level=1)
        except (KeyError, ValueError):
            continue
        frame = _normalize_yahoo_history_frame(item)
        frame.attrs["provider"] = "yfinance_batch"
        cleaned = _drop_incomplete_price_rows(frame, ticker, interval=interval)
        if not cleaned.empty:
            frames[ticker] = cleaned
    return frames


def _normalize_yahoo_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized = df.copy()
    normalized = normalized.rename(columns={column: str(column).lower().replace(" ", "_") for column in normalized.columns})
    normalized.index.name = "date"
    return normalized.reset_index()


def _drop_incomplete_price_rows(df: pd.DataFrame, ticker: str, *, interval: str = "1d") -> pd.DataFrame:
    if df.empty:
        return df

    required = ["date", "open", "high", "low", "close"]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        logger.warning("Price history for %s is missing required columns: %s", ticker, missing_columns)
        return pd.DataFrame()

    before = len(df)
    cleaned = df.dropna(subset=required).copy()
    cleaned = _drop_current_daily_row_during_market_hours(cleaned, interval=interval)
    cleaned.attrs.update(df.attrs)
    dropped = before - len(cleaned)
    if dropped:
        logger.info("Dropped %s incomplete price rows for %s", dropped, ticker)
    return cleaned


def _drop_current_daily_row_during_market_hours(df: pd.DataFrame, *, interval: str) -> pd.DataFrame:
    if df.empty or str(interval).strip().lower() != "1d" or "date" not in df.columns:
        return df
    now = _current_market_datetime()
    if now.time() >= DAILY_MARKET_CLOSE_BUFFER:
        return df

    dates = pd.to_datetime(df["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        row_dates = dates.dt.tz_convert(MARKET_TIMEZONE).dt.date
    else:
        row_dates = dates.dt.date
    return df[row_dates != now.date()].copy()


def _current_market_datetime() -> datetime:
    return datetime.now(MARKET_TIMEZONE)


def get_basic_fundamentals(ticker: str) -> dict[str, Any]:
    """Fetch compact free fundamentals plus optional local ownership/governance data."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("Fundamental data fetch failed for %s: %s", ticker, exc)
        info = {}
    result = {key: _clean_scalar(info.get(key)) for key in FUNDAMENTAL_KEYS}
    sources = []
    if any(value is not None for value in result.values()):
        sources.append("yfinance")

    sec_facts = get_sec_fundamentals(ticker)
    if sec_facts:
        sources.extend(sec_facts.pop("_sources", []))
        result.update(sec_facts)

    ownership = get_ownership_fundamentals(ticker)
    if ownership:
        sources.extend(ownership.pop("_sources", []))
        result.update(ownership)

    result["_sources"] = list(dict.fromkeys(sources))
    return result


def _clean_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
import logging
import os
import sqlite3
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from stock_advisor.config.settings import settings
from stock_advisor.data.exchange_eod import (
    bhavcopy_fetch_succeeded_for_date,
    get_exchange_eod_rows_for_date,
    get_latest_exchange_eod_rows,
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


def _store_price_history(ticker: str, interval: str, df: pd.DataFrame) -> None:
    if df.empty or not _price_cache_allowed(interval):
        return
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return

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
    if cache_df.empty:
        return

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
                INSERT OR REPLACE INTO price_cache_meta
                (ticker_id, interval, max_period_days, latest_date, earliest_date, provider, fetched_at, row_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker_id, interval, max_period_days, latest_date, earliest_date, provider, fetched_at, int(row_count or 0)),
            )

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
    except Exception as exc:  # noqa: BLE001
        logger.debug("Price cache store failed for %s: %s", ticker_key, exc)


def _refresh_live_metrics(conn: sqlite3.Connection, ticker_id: int, interval: str, latest_date: str, last_price: float) -> None:
    """Recompute year_high/year_low/near_52w_high_pct/return_30d_pct/return_365d_pct for a
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

    return_30d_pct = _return_pct(30)
    return_365d_pct = _return_pct(365)
    computed_at = _current_market_datetime().isoformat()
    conn.execute(
        """
        INSERT INTO live_metrics (ticker_id, last_price, year_high, year_low, near_52w_high_pct, return_30d_pct, return_365d_pct, as_of_date, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker_id) DO UPDATE SET
            last_price=excluded.last_price, year_high=excluded.year_high, year_low=excluded.year_low,
            near_52w_high_pct=excluded.near_52w_high_pct, return_30d_pct=excluded.return_30d_pct,
            return_365d_pct=excluded.return_365d_pct, as_of_date=excluded.as_of_date, computed_at=excluded.computed_at
        WHERE excluded.as_of_date >= COALESCE(live_metrics.as_of_date, '')
        """,
        (ticker_id, last_price, year_high, year_low, near_52w_high_pct, return_30d_pct, return_365d_pct, latest_date, computed_at),
    )


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
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
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
            date TEXT,
            fetched_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_metrics (
            ticker_id INTEGER PRIMARY KEY REFERENCES instrument_master(ticker_id),
            last_price REAL,
            year_high REAL,
            year_low REAL,
            near_52w_high_pct REAL,
            return_30d_pct REAL,
            return_365d_pct REAL,
            as_of_date TEXT,
            computed_at TEXT
        )
        """
    )
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
            provider TEXT,
            fetched_at TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            PRIMARY KEY (ticker_id, interval)
        )
        """
    )
    _migrate_price_cache_meta_columns(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_history_cache_lookup ON price_history_cache (ticker_id, interval, date)")


def _migrate_price_cache_meta_columns(conn: sqlite3.Connection) -> None:
    """Add earliest_date to price_cache_meta (needed to tell whether a configured
    PRICE_HISTORY_START_DATE has moved earlier than what's already cached) and backfill it
    from price_history_cache for any pre-existing rows. No-ops once already migrated."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(price_cache_meta)")}
    if not cols:
        return  # table doesn't exist yet (handled by the CREATE TABLE IF NOT EXISTS above)
    if "earliest_date" not in cols:
        conn.execute("ALTER TABLE price_cache_meta ADD COLUMN earliest_date TEXT")
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


def list_instrument_master_tickers(universe: str) -> list[str]:
    """Return tickers from instrument_master for an exchange-derivable universe ('full_nse',
    'full_bse', 'all_india') or 'broad' (via the in_nifty_total_market flag)."""
    normalized = str(universe or "").strip().lower()
    if normalized in {"broad", "nse_total_market", "nifty_total_market", "total_market"}:
        where = "in_nifty_total_market = 1"
    elif normalized in {"full_nse", "nse_full", "nse_equity"}:
        where = "market = 'IN' AND exchange IN ('NSE', 'NSE+BSE')"
    elif normalized in {"full_bse", "bse_full", "bse_equity"}:
        where = "market = 'IN' AND exchange IN ('BSE', 'NSE+BSE')"
    else:  # all_india and aliases
        where = "market = 'IN'"
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


def backfill_market_cap_from_yfinance(tickers: list[str] | None = None, *, max_workers: int = 8) -> dict[str, Any]:
    """Fetch yfinance marketCap for the given tickers (default: every instrument_master ticker)
    and upsert into market_cap.market_cap. Real per-ticker network cost (one call per ticker) —
    call explicitly; not part of any automatic sync."""
    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        if tickers is None:
            tickers = [row["ticker"] for row in conn.execute("SELECT ticker FROM instrument_master")]
        ticker_id_map = _bulk_ticker_ids(conn, tickers)

    if not ticker_id_map:
        return {"requested_ticker_count": 0, "updated_ticker_count": 0, "missing_ticker_count": 0}

    def _fetch_one(ticker: str) -> tuple[str, float | None]:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:  # noqa: BLE001
            return ticker, None
        value = info.get("marketCap")
        return ticker, (float(value) if isinstance(value, (int, float)) else None)

    results: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(ticker_id_map)))) as executor:
        futures = [executor.submit(_fetch_one, ticker) for ticker in ticker_id_map]
        for future in as_completed(futures):
            ticker, market_cap = future.result()
            results[ticker] = market_cap

    fetched_at = _current_market_datetime().isoformat()
    rows_to_write = [
        (ticker_id_map[ticker], market_cap, fetched_at, fetched_at)
        for ticker, market_cap in results.items()
        if market_cap is not None
    ]
    with _price_cache_connection() as conn:
        conn.executemany(
            """
            INSERT INTO market_cap (ticker_id, market_cap, date, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker_id) DO UPDATE SET market_cap=excluded.market_cap, date=excluded.date, fetched_at=excluded.fetched_at
            """,
            rows_to_write,
        )
        conn.commit()
    return {
        "requested_ticker_count": len(ticker_id_map),
        "updated_ticker_count": len(rows_to_write),
        "missing_ticker_count": len(ticker_id_map) - len(rows_to_write),
    }


def load_stock_universe_from_db(
    universe: str, *, refresh: bool = False, max_stocks: int | None = None
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

    query = f"""
        SELECT
            im.ticker, im.symbol, im.name, im.isin,
            sc.sector, sc.industry, sc.basic_industry, sc.index_name,
            im.source, im.active, im.series,
            mc.free_float_market_cap,
            lm.last_price, lm.year_high, lm.year_low, lm.near_52w_high_pct, lm.return_30d_pct, lm.return_365d_pct,
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
            progress_callback({"phase": "scanning_days", "source": "bhavcopy", "completed": index + 1, "total": len(trade_dates)})

    stored_count = 0
    for ticker, day_frames in frames_by_ticker.items():
        if day_frames:
            combined = pd.concat(day_frames, ignore_index=True)
            combined.attrs["provider"] = str(day_frames[-1].attrs.get("provider") or "exchange_eod_bhavcopy")
            _store_price_history(ticker, interval, combined)
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


def _fetch_and_store_range(chunk: list[str], start: str, end: str, interval: str) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for a chunk of tickers over [start, end) from Yahoo, overlay the latest
    official NSE/BSE EOD row, and persist. Only called from fill_price_cache_for_universe —
    the one place price history is ever fetched from a provider."""
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
        return {}
    fetched = _split_yahoo_batch_history(raw, chunk, interval=interval)
    fetched = _overlay_exchange_latest_rows(fetched, interval=interval)
    for ticker, frame in fetched.items():
        _store_price_history(ticker, interval, frame)
    return fetched


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
        what's cached) -> backward-fill Yahoo fetch from start_date to the existing earliest_date.
      - cached latest_date is behind the most recent expected trading day -> forward Yahoo fetch
        from latest_date (inclusive, to safely re-cover a possibly-incomplete last row) to today.
      - needs both backward and forward -> treated as a full-range fetch (start_date to today);
        INSERT OR REPLACE makes re-covering the already-cached middle harmless, and this
        combination is rare (only right after preponing start_date while also being behind on
        the latest day).
      - neither -> skipped, zero network calls.

      Fallback pass: any full_fetch/backward_fetch ticker Yahoo still failed on (forward_fetch
      failures are skipped here -- the priority pass just tried the same recent window moments
      earlier) gets a full bhavcopy day-walk over [start_date, today]. A ticker with zero trades
      on either exchange across that entire range is marked instrument_master.active=0 (dormant)
      so future refreshes stop retrying it -- guarded by a bhavcopy day-coverage check so a
      transient NSE/BSE outage during the walk can't be misread as mass delisting.
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

    priority_window_days = max(0, int(settings.bhavcopy_priority_window_days))
    if priority_window_days > 0 and interval == "1d":
        priority_start = (today_date - timedelta(days=priority_window_days)).isoformat()
        backfill_bhavcopy_history(
            unique_tickers,
            start_date=priority_start,
            end_date=today_date.isoformat(),
            interval=interval,
            progress_callback=_wrapped_progress("bhavcopy_priority"),
        )

    with _price_cache_connection() as conn:
        _ensure_price_cache_schema(conn)
        placeholders = ",".join("?" for _ in unique_tickers)
        meta_rows = conn.execute(
            f"""
            SELECT im.ticker AS ticker, m.latest_date AS latest_date, m.earliest_date AS earliest_date
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
    for ticker in unique_tickers:
        meta = meta_by_ticker.get(ticker)
        latest_date = str(meta["latest_date"]) if meta and meta["latest_date"] else None
        if latest_date is None:
            full_fetch.append(ticker)
            continue
        earliest_date = str(meta["earliest_date"]) if meta["earliest_date"] else latest_date
        needs_forward = latest_date < expected_trading_day
        needs_backward = earliest_date > effective_start_date
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

    def _report(phase: str) -> None:
        if progress_callback:
            progress_callback({"phase": phase, "source": "yfinance", "completed": completed, "total": total})

    _report("classifying")

    def _fetch_group(phase: str, group_tickers: list[str], start: str, end: str) -> None:
        nonlocal completed
        remaining = list(group_tickers)
        for _attempt in range(attempts):
            if not remaining:
                break
            still_remaining: list[str] = []
            for chunk_start in range(0, len(remaining), chunk_size):
                chunk = remaining[chunk_start : chunk_start + chunk_size]
                fetched = _fetch_and_store_range(chunk, start, end, interval)
                for ticker in chunk:
                    if ticker in fetched:
                        fetched_counts[phase] += 1
                        completed += 1
                    else:
                        still_remaining.append(ticker)
                _report(phase)
            remaining = still_remaining
        failed.extend(remaining)

    forward_fetch_tickers = {ticker for group in forward_by_date.values() for ticker in group}

    if full_fetch:
        _fetch_group("full_fetch", full_fetch, effective_start_date, end_exclusive)
    for earliest_date, group_tickers in backward_by_date.items():
        _fetch_group("backward_fetch", group_tickers, effective_start_date, earliest_date)
    for latest_date, group_tickers in forward_by_date.items():
        _fetch_group("forward_fetch", group_tickers, latest_date, end_exclusive)

    # Fallback pass: only full_fetch/backward_fetch failures need this -- the priority pass
    # already just tried the recent window for every ticker (including forward_fetch failures)
    # moments ago, so retrying those via bhavcopy again wouldn't find anything new.
    bhavcopy_fallback_recovered_count = 0
    dormant_marked_count = 0
    dormant_tickers: list[str] = []
    phase_c_candidates = [ticker for ticker in failed if ticker not in forward_fetch_tickers]
    if phase_c_candidates and interval == "1d":
        fallback_result = backfill_bhavcopy_history(
            phase_c_candidates,
            start_date=effective_start_date,
            end_date=today_date.isoformat(),
            interval=interval,
            progress_callback=_wrapped_progress("bhavcopy_fallback"),
        )
        scanned = int(fallback_result.get("trading_days_scanned") or 0)
        covered = int(fallback_result.get("trading_days_with_bhavcopy_data") or 0)
        coverage_pct = round(100 * covered / scanned, 2) if scanned else 0.0

        still_missing = set(_tickers_without_price_cache(interval=interval, active_only=True, tickers=phase_c_candidates))
        recovered = [ticker for ticker in phase_c_candidates if ticker not in still_missing]
        bhavcopy_fallback_recovered_count = len(recovered)
        failed = [ticker for ticker in failed if ticker not in recovered]

        if still_missing and coverage_pct >= settings.min_bhavcopy_coverage_pct:
            dormant_marked_count = _mark_tickers_dormant(list(still_missing))
            dormant_tickers = sorted(still_missing)[:50]
            failed = [ticker for ticker in failed if ticker not in still_missing]

    return {
        "requested_ticker_count": total,
        "skipped_up_to_date_count": skipped_count,
        "full_fetch_count": fetched_counts["full_fetch"],
        "backward_fetch_count": fetched_counts["backward_fetch"],
        "forward_fetch_count": fetched_counts["forward_fetch"],
        "bhavcopy_fallback_recovered_count": bhavcopy_fallback_recovered_count,
        "dormant_marked_count": dormant_marked_count,
        "dormant_tickers": dormant_tickers,
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

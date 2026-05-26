from __future__ import annotations

from collections import Counter
from datetime import datetime, time, timedelta
import logging
import os
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from stock_advisor.config.settings import settings
from stock_advisor.data.exchange_eod import get_latest_exchange_eod_rows
from stock_advisor.data.sec_edgar import get_sec_fundamentals
from stock_advisor.data.ownership import get_ownership_fundamentals
from stock_advisor.data.stooq import get_stooq_price_history
from stock_advisor.data.yahoo_chart import get_yahoo_chart_price_history

logger = logging.getLogger(__name__)
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
DAILY_MARKET_CLOSE_BUFFER = time(15, 45)
PRICE_CACHE_ENABLED = os.getenv("PRICE_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
PRICE_CACHE_MAX_AGE_HOURS = float(os.getenv("PRICE_CACHE_MAX_AGE_HOURS", "8"))
CACHEABLE_INTERVALS = {"1d", "1wk", "1mo"}

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


def get_price_history(ticker: str, period: str = "6mo", interval: str = "1d", *, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch OHLCV price history with free providers and fallback.

    Daily Indian equities prefer the official NSE/BSE bhavcopy for the latest
    EOD row before falling back to Yahoo/Stooq history.
    """
    cached = _load_cached_price_history(ticker, period, interval)
    cached = _overlay_exchange_latest_rows({ticker: cached}, period=period, interval=interval).get(_cache_ticker(ticker), cached)
    if force_refresh and _has_exchange_latest_overlay(cached):
        return cached
    if not force_refresh and _cached_history_is_fresh(cached, interval):
        return cached

    df = _drop_incomplete_price_rows(_get_yahoo_price_history(ticker, period, interval), ticker, interval=interval)
    if not df.empty:
        df = _overlay_exchange_latest_rows({ticker: df}, period=period, interval=interval).get(_cache_ticker(ticker), df)
        _store_price_history(ticker, period, interval, df)
        return df

    chart_fallback = _drop_incomplete_price_rows(get_yahoo_chart_price_history(ticker, period, interval), ticker, interval=interval)
    if not chart_fallback.empty:
        logger.info("Using Yahoo chart fallback price history for %s", ticker)
        chart_fallback = _overlay_exchange_latest_rows({ticker: chart_fallback}, period=period, interval=interval).get(_cache_ticker(ticker), chart_fallback)
        _store_price_history(ticker, period, interval, chart_fallback)
        return chart_fallback

    fallback = _drop_incomplete_price_rows(get_stooq_price_history(ticker, period, interval), ticker, interval=interval)
    if not fallback.empty:
        logger.info("Using Stooq fallback price history for %s", ticker)
        fallback = _overlay_exchange_latest_rows({ticker: fallback}, period=period, interval=interval).get(_cache_ticker(ticker), fallback)
        _store_price_history(ticker, period, interval, fallback)
        return fallback
    if not cached.empty:
        cached.attrs["provider"] = "sqlite_cache_stale"
        logger.info("Using stale cached price history for %s after provider fetch failed", ticker)
        return cached
    return fallback


def get_price_histories(
    tickers: list[str] | tuple[str, ...],
    period: str = "6mo",
    interval: str = "1d",
    *,
    chunk_size: int = 80,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV histories for many tickers using a batched Yahoo request.

    The single-ticker ``get_price_history`` path is more resilient because it has
    fallbacks. This batched path is used for broad NSE universe analytics where
    serial 500+ ticker downloads would make the UI unusable.
    """
    unique_tickers = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))
    if not unique_tickers:
        return {}

    results: dict[str, pd.DataFrame] = {}
    stale_results: dict[str, pd.DataFrame] = {}
    tickers_to_fetch = []
    for ticker in unique_tickers:
        cached = _load_cached_price_history(ticker, period, interval)
        if not force_refresh and _cached_history_is_fresh(cached, interval):
            results[ticker] = cached
        else:
            if not cached.empty:
                stale_results[ticker] = cached
            tickers_to_fetch.append(ticker)

    exchange_updated = _overlay_exchange_latest_rows({**stale_results, **results}, period=period, interval=interval)
    for ticker, frame in exchange_updated.items():
        if ticker in results or ticker in stale_results:
            results[ticker] = frame
            if ticker in tickers_to_fetch:
                tickers_to_fetch.remove(ticker)

    chunk_size = max(1, int(chunk_size or 80))
    for start in range(0, len(tickers_to_fetch), chunk_size):
        chunk = tickers_to_fetch[start : start + chunk_size]
        try:
            raw = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=30,
                group_by="ticker",
            )
        except Exception as exc:
            logger.warning("Batch price history fetch failed for %s tickers: %s", len(chunk), exc)
            continue
        fetched = _split_yahoo_batch_history(raw, chunk, interval=interval)
        fetched = _overlay_exchange_latest_rows(fetched, period=period, interval=interval)
        for fetched_ticker, frame in fetched.items():
            _store_price_history(fetched_ticker, period, interval, frame)
        results.update(fetched)

    for ticker, cached in stale_results.items():
        if ticker not in results:
            cached.attrs["provider"] = "sqlite_cache_stale"
            results[ticker] = cached
    return results


def _load_cached_price_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not _price_cache_allowed(interval):
        return pd.DataFrame()
    ticker_key = _cache_ticker(ticker)
    requested_days = _period_days(period)
    try:
        with _price_cache_connection() as conn:
            _ensure_price_cache_schema(conn)
            meta = conn.execute(
                """
                SELECT max_period_days, fetched_at
                FROM price_cache_meta
                WHERE ticker = ? AND interval = ?
                """,
                (ticker_key, interval),
            ).fetchone()
            if not meta:
                return pd.DataFrame()
            if requested_days is not None and int(meta["max_period_days"] or 0) < requested_days:
                return pd.DataFrame()
            params: list[Any] = [ticker_key, interval]
            date_filter = ""
            if requested_days is not None:
                start_date = (_current_market_datetime().date() - timedelta(days=requested_days + 7)).isoformat()
                date_filter = " AND date >= ?"
                params.append(start_date)
            rows = conn.execute(
                f"""
                SELECT date, open, high, low, close, volume, provider, fetched_at
                FROM price_history_cache
                WHERE ticker = ? AND interval = ?{date_filter}
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


def _store_price_history(ticker: str, period: str, interval: str, df: pd.DataFrame) -> None:
    if df.empty or not _price_cache_allowed(interval):
        return
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return

    ticker_key = _cache_ticker(ticker)
    provider = str(df.attrs.get("provider") or "unknown")
    fetched_at = _current_market_datetime().isoformat()
    period_days = _period_days(period) or 999999
    cache_df = df.copy()
    cache_df["date"] = pd.to_datetime(cache_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in cache_df.columns:
            cache_df[column] = None
        cache_df[column] = pd.to_numeric(cache_df[column], errors="coerce")
    cache_df = cache_df.dropna(subset=["date", "open", "high", "low", "close"])
    if cache_df.empty:
        return

    rows = [
        (
            ticker_key,
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
    latest_date = max(row[2] for row in rows)

    try:
        with _price_cache_connection() as conn:
            _ensure_price_cache_schema(conn)
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_history_cache
                (ticker, interval, date, open, high, low, close, volume, provider, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            existing = conn.execute(
                """
                SELECT max_period_days
                FROM price_cache_meta
                WHERE ticker = ? AND interval = ?
                """,
                (ticker_key, interval),
            ).fetchone()
            max_period_days = max(period_days, int(existing["max_period_days"] or 0)) if existing else period_days
            row_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM price_history_cache
                WHERE ticker = ? AND interval = ?
                """,
                (ticker_key, interval),
            ).fetchone()[0]
            conn.execute(
                """
                INSERT OR REPLACE INTO price_cache_meta
                (ticker, interval, max_period_days, latest_date, provider, fetched_at, row_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker_key, interval, max_period_days, latest_date, provider, fetched_at, int(row_count or 0)),
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Price cache store failed for %s: %s", ticker_key, exc)


def _overlay_exchange_latest_rows(frames: dict[str, pd.DataFrame], *, period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Merge latest official NSE/BSE EOD rows into existing daily histories."""
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
        _store_price_history(ticker, period, interval, exchange_frame)
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


def _has_exchange_latest_overlay(df: pd.DataFrame) -> bool:
    provider = str(df.attrs.get("provider") or "")
    return bool(df.attrs.get("exchange_trade_date")) and provider in {"nse_bhavcopy", "bse_bhavcopy"}


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history_cache (
            ticker TEXT NOT NULL,
            interval TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            provider TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (ticker, interval, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_cache_meta (
            ticker TEXT NOT NULL,
            interval TEXT NOT NULL,
            max_period_days INTEGER NOT NULL,
            latest_date TEXT,
            provider TEXT,
            fetched_at TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            PRIMARY KEY (ticker, interval)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_history_cache_lookup ON price_history_cache (ticker, interval, date)")


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
        ticker_filter = f" AND ticker IN ({placeholders})"
        params.extend(ticker_keys)
    try:
        with _price_cache_connection() as conn:
            _ensure_price_cache_schema(conn)
            rows = conn.execute(
                f"""
                SELECT ticker, row_count, latest_date, fetched_at
                FROM price_cache_meta
                WHERE interval = ?{ticker_filter}
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
    period: str = "1d",
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
        _store_price_history(ticker, period, interval, frame)

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


def warm_price_history_cache(
    tickers: list[str] | tuple[str, ...],
    *,
    period: str = "2y",
    interval: str = "1d",
    chunk_size: int = 80,
    retry_attempts: int = 2,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch/cache OHLCV data for a ticker set and report coverage."""
    unique_tickers = list(dict.fromkeys(_cache_ticker(ticker) for ticker in tickers if _cache_ticker(ticker)))
    if not unique_tickers:
        return {"requested_ticker_count": 0, "available_ticker_count": 0, "missing_ticker_count": 0, "providers": {}}
    results: dict[str, pd.DataFrame] = {}
    remaining = unique_tickers
    attempts = max(1, int(retry_attempts) + 1)
    for _attempt in range(attempts):
        if not remaining:
            break
        attempt_results = get_price_histories(
            remaining,
            period=period,
            interval=interval,
            chunk_size=chunk_size,
            force_refresh=force_refresh,
        )
        results.update(attempt_results)
        remaining = [ticker for ticker in unique_tickers if ticker not in results]
    providers: dict[str, int] = {}
    for frame in results.values():
        provider = str(frame.attrs.get("provider") or "unknown")
        providers[provider] = providers.get(provider, 0) + 1
    return {
        "requested_ticker_count": len(unique_tickers),
        "available_ticker_count": len(results),
        "missing_ticker_count": max(0, len(unique_tickers) - len(results)),
        "retry_attempts": max(0, int(retry_attempts)),
        "force_refresh": force_refresh,
        "period": period,
        "interval": interval,
        "providers": providers,
        "missing_tickers": remaining[:50],
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


def _get_yahoo_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=15,
        )
    except Exception as exc:
        logger.warning("Price history fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]

    df = df.rename(columns={c: str(c).lower().replace(" ", "_") for c in df.columns})
    df.index.name = "date"
    out = df.reset_index()

    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(out.columns):
        logger.warning("Price history for %s is missing required columns: %s", ticker, sorted(required - set(out.columns)))
        return pd.DataFrame()

    out.attrs["provider"] = "yfinance"
    return out


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

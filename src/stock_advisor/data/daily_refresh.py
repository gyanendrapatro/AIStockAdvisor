from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from stock_advisor.config.settings import PROJECT_ROOT
from stock_advisor.data.exchange_eod import clear_exchange_eod_fetch_cache
from stock_advisor.data.market_data import (
    get_price_cache_status,
    refresh_latest_exchange_eod_cache,
    warm_price_history_cache,
)
from stock_advisor.data.nse_indices import clear_nse_index_archive_cache, get_nse_index_price_history
from stock_advisor.data.universe import (
    DEFAULT_NSE_INDEX_NAME,
    list_stock_universe,
    load_stock_universe,
    refresh_bse_stock_universe,
    refresh_full_stock_universe,
    refresh_india_stock_universe,
    refresh_stock_universe,
)


DEFAULT_DAILY_REFRESH_REPORT_PATH = PROJECT_ROOT / "data" / "daily_refresh_report.json"


def run_daily_market_data_refresh(
    *,
    refresh_universes: bool = True,
    refresh_broad_universe: bool = True,
    refresh_full_nse_universe: bool = True,
    refresh_bse_universe: bool = False,
    refresh_india_universe: bool = False,
    warm_price_cache: bool = True,
    refresh_exchange_eod: bool = True,
    warm_index_cache: bool = True,
    warm_universe: str = "full_nse",
    index_name: str = DEFAULT_NSE_INDEX_NAME,
    period: str = "2y",
    interval: str = "1d",
    max_universe_symbols: int | None = None,
    max_price_symbols: int | None = None,
    chunk_size: int = 80,
    retry_attempts: int = 2,
    force_refresh_prices: bool = True,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Refresh public NSE/BSE inputs and local price cache for daily analytics.

    The daily path does three separate jobs:
    1. refreshes public universe files, including NSE equity masters;
    2. pulls the latest official NSE/BSE EOD bhavcopy row into SQLite;
    3. warms official NSE index-close archives used by RRG and index views;
    4. warms longer OHLCV history so sector, RRG, and stock analytics run fast.
    """
    started_at = datetime.now(timezone.utc)
    output_path = Path(report_path) if report_path is not None else DEFAULT_DAILY_REFRESH_REPORT_PATH
    steps: dict[str, Any] = {}
    warnings: list[str] = []

    clear_exchange_eod_fetch_cache()
    clear_nse_index_archive_cache()

    if refresh_universes:
        if refresh_broad_universe:
            steps["broad_nse_universe"] = _capture_step(
                lambda: refresh_stock_universe(index_name=index_name),
            )
        if refresh_full_nse_universe:
            steps["full_nse_universe"] = _capture_step(
                lambda: refresh_full_stock_universe(max_symbols=max_universe_symbols),
            )
        if refresh_bse_universe:
            steps["bse_universe"] = _capture_step(refresh_bse_stock_universe)
        if refresh_india_universe:
            steps["india_universe"] = _capture_step(refresh_india_stock_universe)

    universe_summary = _capture_step(lambda: list_stock_universe(universe=warm_universe, limit=0))
    steps["warm_universe_summary"] = universe_summary
    universe_df = load_stock_universe(universe=warm_universe, max_stocks=max_price_symbols)
    tickers = list(universe_df["ticker"]) if not universe_df.empty else []
    if not tickers:
        warnings.append(f"No tickers were available for warm_universe={warm_universe}.")

    exchange_eod_result: dict[str, Any] | None = None
    if refresh_exchange_eod and tickers:
        exchange_eod_result = refresh_latest_exchange_eod_cache(
            tickers,
            interval="1d",
            period="1d",
        )
        steps["exchange_eod_cache"] = {"status": "ok", "result": exchange_eod_result}

    index_warm_result: dict[str, Any] | None = None
    if warm_index_cache:
        index_warm_result = _warm_index_history_cache(period=period, interval=interval)
        steps["index_history_cache"] = {"status": "ok", "result": index_warm_result}

    price_warm_result: dict[str, Any] | None = None
    if warm_price_cache and tickers:
        price_warm_result = warm_price_history_cache(
            tickers,
            period=period,
            interval=interval,
            chunk_size=chunk_size,
            retry_attempts=retry_attempts,
            force_refresh=force_refresh_prices,
        )
        steps["price_history_cache"] = {"status": "ok", "result": price_warm_result}

    cache_status = get_price_cache_status(tickers=tickers, interval=interval) if tickers else {}
    completed_at = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "status": "ok" if all(_step_ok(step) for step in steps.values()) else "partial",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round((completed_at - started_at).total_seconds(), 2),
        "warm_universe": warm_universe,
        "price_period": period,
        "price_interval": interval,
        "universe_stock_count": int(len(universe_df)),
        "settings": {
            "refresh_universes": refresh_universes,
            "refresh_broad_universe": refresh_broad_universe,
            "refresh_full_nse_universe": refresh_full_nse_universe,
            "refresh_bse_universe": refresh_bse_universe,
            "refresh_india_universe": refresh_india_universe,
            "warm_price_cache": warm_price_cache,
            "refresh_exchange_eod": refresh_exchange_eod,
            "warm_index_cache": warm_index_cache,
            "max_universe_symbols": max_universe_symbols,
            "max_price_symbols": max_price_symbols,
            "chunk_size": chunk_size,
            "retry_attempts": retry_attempts,
            "force_refresh_prices": force_refresh_prices,
        },
        "steps": steps,
        "exchange_eod": exchange_eod_result,
        "index_cache_warm": index_warm_result,
        "price_cache_warm": price_warm_result,
        "price_cache_status": cache_status,
        "warnings": warnings,
        "report_path": str(output_path),
    }
    _write_report(output_path, report)
    return report


def load_daily_refresh_report(path: str | Path | None = None) -> dict[str, Any]:
    """Load the latest daily refresh report if it exists."""
    report_path = Path(path) if path is not None else DEFAULT_DAILY_REFRESH_REPORT_PATH
    if not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _warm_index_history_cache(*, period: str, interval: str) -> dict[str, Any]:
    """Warm official NSE index-close archives for every RRG/market-index row."""
    try:
        from stock_advisor.analysis.market_analytics import list_rrg_index_definitions
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc), "requested_index_count": 0}

    index_names = ["Nifty 50"]
    for definition in list_rrg_index_definitions().values():
        name = str(definition.get("name") or "").strip()
        if name:
            index_names.append(name)

    unique_names = list(dict.fromkeys(index_names))
    rows: dict[str, int] = {}
    missing: list[str] = []
    for name in unique_names:
        history = get_nse_index_price_history(name, period=period, interval=interval)
        if history.empty:
            missing.append(name)
        else:
            rows[name] = int(len(history))

    return {
        "status": "ok",
        "requested_index_count": len(unique_names),
        "available_index_count": len(rows),
        "missing_index_count": len(missing),
        "period": period,
        "interval": interval,
        "rows_by_index": rows,
        "missing_indexes": missing[:25],
    }


def _capture_step(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"status": "ok", "result": fn()}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _step_ok(step: Any) -> bool:
    if not isinstance(step, dict):
        return True
    return str(step.get("status") or "ok") == "ok"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

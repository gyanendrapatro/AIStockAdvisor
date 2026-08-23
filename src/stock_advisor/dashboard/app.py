from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from html import escape
import sys
import threading
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from stock_advisor.analysis.indicators import add_indicators
from stock_advisor.analysis.market_analytics import (
    SECTOR_ANALYTICS_VERSION,
    get_industry_analytics,
    get_market_breadth,
    get_market_indices,
    get_moving_average_crossover_scan,
    get_relative_rotation_graph,
    get_sector_analytics,
    get_top_gainers,
    list_industry_definitions,
    list_rrg_index_definitions,
    rank_industry_stocks,
)
from stock_advisor.analysis.pipeline import analyze_stock, rank_watchlist
from stock_advisor.analysis.sector_rotation import list_sector_definitions, rank_sector_stocks
from stock_advisor.agents.sector_rotation_workflow import run_sector_rotation_workflow
from stock_advisor.agents.stock_research_agent import run_stock_research_agent
from stock_advisor.config.settings import load_watchlists
from stock_advisor.data.daily_refresh import load_daily_refresh_report, run_daily_market_data_refresh
from stock_advisor.data.market_data import (
    CACHE_REFRESH_HINT,
    backfill_market_cap_from_yfinance,
    fill_price_cache_for_universe,
    get_price_cache_status,
    get_price_history,
    list_all_instrument_master_tickers,
    list_instrument_master_tickers,
    sync_instrument_master,
)
from stock_advisor.data.universe import list_sector_constituents, list_stock_universe


SECTOR_ANALYTICS_PERIOD = "2y"
SECTOR_ANALYTICS_INTERVAL = "1d"
SECTOR_ANALYTICS_STOCK_LIMIT = 30
MARKET_TIMEZONE_NAME = "Asia/Kolkata"
MARKET_TIMEZONE = ZoneInfo(MARKET_TIMEZONE_NAME)
AUTO_REFRESH_MIN_COVERAGE_PCT = 95.0


st.set_page_config(page_title="AI Stock Advisor", layout="wide")
st.title("AI Stock Advisor")
st.caption("Educational stock research dashboard. Not financial advice.")
st.caption("Market, sector, and industry data uses official NSE/BSE EOD rows for latest candles, with free Yahoo/Stooq history fallback; no external dashboard data source is used.")


@st.cache_resource
def _market_cap_backfill_jobs() -> dict[str, dict[str, object]]:
    return {}


def _latest_market_cap_backfill_job() -> dict[str, object] | None:
    jobs = [job for job in _market_cap_backfill_jobs().values() if job.get("scope") == "market_cap_backfill"]
    if not jobs:
        return None
    return sorted(jobs, key=lambda item: str(item.get("started_at") or ""), reverse=True)[0]


def _start_market_cap_backfill_job() -> dict[str, object]:
    jobs = _market_cap_backfill_jobs()
    for job in jobs.values():
        if job.get("scope") == "market_cap_backfill" and job.get("status") == "running":
            return job

    job_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    job: dict[str, object] = {
        "id": job_id,
        "scope": "market_cap_backfill",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    jobs[job_id] = job

    def _runner() -> None:
        try:
            result = backfill_market_cap_from_yfinance()
            job["status"] = "completed"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["result"] = result
        except Exception as exc:  # noqa: BLE001
            job["status"] = "failed"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["error"] = str(exc)

    threading.Thread(target=_runner, name=f"market-cap-backfill-{job_id}", daemon=True).start()
    return job


@st.cache_resource
def _price_cache_fill_jobs() -> dict[str, dict[str, object]]:
    return {}


def _latest_price_cache_fill_job() -> dict[str, object] | None:
    jobs = [job for job in _price_cache_fill_jobs().values() if job.get("scope") == "price_cache_fill"]
    if not jobs:
        return None
    return sorted(jobs, key=lambda item: str(item.get("started_at") or ""), reverse=True)[0]


def _start_price_cache_fill_job() -> dict[str, object]:
    jobs = _price_cache_fill_jobs()
    for job in jobs.values():
        if job.get("scope") == "price_cache_fill" and job.get("status") == "running":
            return job

    job_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    job: dict[str, object] = {
        "id": job_id,
        "scope": "price_cache_fill",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "progress": {"phase": "starting", "completed": 0, "total": 0},
    }
    jobs[job_id] = job

    def _on_progress(update: dict[str, object]) -> None:
        job["progress"] = update

    def _runner() -> None:
        try:
            tickers = list_all_instrument_master_tickers()
            result = fill_price_cache_for_universe(tickers, progress_callback=_on_progress)
            job["status"] = "completed"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["result"] = result
        except Exception as exc:  # noqa: BLE001
            job["status"] = "failed"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["error"] = str(exc)

    threading.Thread(target=_runner, name=f"price-cache-fill-{job_id}", daemon=True).start()
    return job


@st.cache_data(ttl=120, show_spinner=False)
def _global_cache_universe_tickers(universe: str) -> list[str]:
    # instrument_master.in_nifty_total_market tags NIFTY Total Market membership explicitly at
    # sync time, so "broad" is DB-backed too now — no CSV fallback needed for any universe.
    return list_instrument_master_tickers(universe)


_GLOBAL_UNIVERSE_OPTIONS = {
    "Full NSE": "full_nse",
    "Broad NSE Total Market": "broad",
    "All India NSE+BSE": "all_india",
}


with st.container():
    # Read-only status display — the price cache is only ever written to from the sidebar's
    # "Refresh price cache" button (or the daily_refresh CLI / warm_price_history_cache MCP
    # tool), never from here. This just shows current coverage for the selected universe.
    global_universe_label = st.selectbox(
        "Cache universe",
        list(_GLOBAL_UNIVERSE_OPTIONS),
        index=0,
        key="global_cache_universe",
        label_visibility="collapsed",
    )
    global_universe = _GLOBAL_UNIVERSE_OPTIONS[global_universe_label]

    global_tickers = _global_cache_universe_tickers(global_universe)
    global_cache_status = get_price_cache_status(tickers=global_tickers, interval="1d")
    if global_cache_status.get("enabled"):
        gc_cached = global_cache_status.get("cached_ticker_count", 0)
        gc_fresh = global_cache_status.get("fresh_ticker_count", 0)
        gc_stale = global_cache_status.get("stale_ticker_count", 0)
        gc_missing = global_cache_status.get("missing_ticker_count", 0)
        gc_requested = global_cache_status.get("requested_ticker_count") or len(global_tickers)
        gc_coverage = global_cache_status.get("fresh_coverage_pct")
        gc_coverage_text = f"{gc_coverage}%" if gc_coverage is not None else "n/a"
        st.caption(
            f"SQLite price cache ({global_universe_label}): {gc_fresh}/{gc_requested} fresh ({gc_coverage_text}), "
            f"{gc_stale} stale, {gc_missing} missing, {gc_cached} cached total. "
            "Use 'Refresh price cache' in the sidebar to update it."
        )

    cold_start_report = load_daily_refresh_report()
    if cold_start_report:
        st.caption(
            f"Last refresh: {str(cold_start_report.get('completed_at', 'n/a'))[:19]}Z, "
            f"status={cold_start_report.get('status', 'n/a')}."
        )
    else:
        st.caption("No refresh has been run yet.")


with st.sidebar:
    st.subheader("Instrument Master")
    st.caption("Consolidate data/*.csv NSE/BSE universes into the instrument_master security table.")
    _INSTRUMENT_MASTER_SYNC_OPTIONS = {
        "All universes (recommended)": None,
        "Broad NSE Total Market": ["broad"],
        "Full NSE Equity": ["full_nse"],
        "Full BSE Equity": ["full_bse"],
        "All India NSE+BSE": ["all_india"],
    }
    instrument_master_sync_label = st.selectbox(
        "Universe to sync from", list(_INSTRUMENT_MASTER_SYNC_OPTIONS), key="instrument_master_sync_universe"
    )
    if st.button("Update instrument_master", key="instrument_master_sync_button"):
        with st.spinner("Syncing instrument_master from selected universe CSV(s)..."):
            st.session_state["instrument_master_sync_result"] = sync_instrument_master(
                universes=_INSTRUMENT_MASTER_SYNC_OPTIONS[instrument_master_sync_label]
            )
    instrument_master_sync_result = st.session_state.get("instrument_master_sync_result")
    if instrument_master_sync_result:
        st.success(f"Synced {instrument_master_sync_result.get('synced_ticker_count', 0)} tickers into instrument_master.")

    st.divider()
    st.caption("Backfill market_cap.market_cap via yfinance (per-ticker network call, several minutes for the full universe).")
    market_cap_job = _latest_market_cap_backfill_job()
    market_cap_running = bool(market_cap_job and market_cap_job.get("status") == "running")
    if st.button("Backfill market cap (yfinance)", key="market_cap_backfill_button", disabled=market_cap_running):
        _start_market_cap_backfill_job()
        market_cap_job = _latest_market_cap_backfill_job()
        market_cap_running = True
    if market_cap_job:
        mc_status = str(market_cap_job.get("status"))
        if mc_status == "running":
            st.info("Backfilling market cap...")
            st_autorefresh(interval=2500, limit=None, key="market_cap_backfill_autorefresh")
        elif mc_status == "completed":
            mc_result = market_cap_job.get("result") or {}
            st.success(
                f"Updated {mc_result.get('updated_ticker_count', 0)}/{mc_result.get('requested_ticker_count', 0)} tickers."
            )
        elif mc_status == "failed":
            st.error(f"Market cap backfill failed: {market_cap_job.get('error')}")

    st.divider()
    st.subheader("Price History Cache")
    st.caption(
        "Fills price_history_cache for every instrument_master ticker. Already-current tickers "
        "are skipped; only missing/new dates are fetched. This is the only place prices are "
        "pulled from — everywhere else in the app just reads what's cached here."
    )
    price_fill_job = _latest_price_cache_fill_job()
    price_fill_running = bool(price_fill_job and price_fill_job.get("status") == "running")
    if st.button("Refresh price cache", key="price_cache_fill_button", disabled=price_fill_running):
        _start_price_cache_fill_job()
        price_fill_job = _latest_price_cache_fill_job()
        price_fill_running = True
    if price_fill_job:
        pf_status = str(price_fill_job.get("status"))
        if pf_status == "running":
            pf_progress = price_fill_job.get("progress") or {}
            st.info(
                f"Refreshing... phase={pf_progress.get('phase', '?')} "
                f"{pf_progress.get('completed', 0)}/{pf_progress.get('total', 0)}"
            )
            st_autorefresh(interval=2500, limit=None, key="price_cache_fill_autorefresh")
        elif pf_status == "completed":
            pf_result = price_fill_job.get("result") or {}
            st.success(
                f"Done: {pf_result.get('skipped_up_to_date_count', 0)} already current, "
                f"{pf_result.get('full_fetch_count', 0)} full-fetched, "
                f"{pf_result.get('backward_fetch_count', 0)} back-filled (start date), "
                f"{pf_result.get('forward_fetch_count', 0)} forward-updated (latest days), "
                f"{pf_result.get('failed_count', 0)} failed."
            )
        elif pf_status == "failed":
            st.error(f"Price cache refresh failed: {price_fill_job.get('error')}")


st.markdown(
    """
    <style>
    .sector-side-panel {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        background: rgba(15, 23, 42, 0.18);
    }
    .sector-side-title {
        text-align: center;
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }
    .sector-side-note {
        color: rgba(148, 163, 184, 0.95);
        text-align: center;
        font-size: 0.78rem;
        margin-bottom: 0.55rem;
    }
    .sector-side-scroll {
        max-height: 420px;
        overflow-y: auto;
        padding-right: 0.2rem;
    }
    .sector-side-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        border-radius: 7px;
        padding: 0.55rem 0.6rem;
        margin: 0.38rem 0;
        background: rgba(148, 163, 184, 0.12);
    }
    .sector-side-main {
        min-width: 0;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.3rem;
    }
    .sector-side-name {
        font-weight: 750;
        color: inherit;
    }
    .sector-side-pill {
        border-radius: 999px;
        padding: 0.1rem 0.42rem;
        background: rgba(16, 185, 129, 0.86);
        color: #ffffff;
        font-size: 0.76rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .sector-side-value {
        flex: 0 0 auto;
        font-weight: 850;
        font-size: 1rem;
    }
    .sector-click-caption {
        color: rgba(148, 163, 184, 0.95);
        font-size: 0.82rem;
        margin-top: -0.2rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _daily_refresh_universe(universe: str | None) -> str:
    normalized = str(universe or "").strip().lower()
    if normalized in {"all_india", "india", "nse_bse", "bse_nse"}:
        return "all_india"
    return "full_nse"


def _expected_latest_complete_eod_date() -> str:
    now = datetime.now(MARKET_TIMEZONE)
    expected = now.date()
    if now.time() < time(17, 30):
        expected -= timedelta(days=1)
    while expected.weekday() >= 5:
        expected -= timedelta(days=1)
    return expected.isoformat()


def _universe_tickers_for_refresh_check(warm_universe: str) -> list[str]:
    try:
        universe_summary = list_stock_universe(universe=warm_universe, limit=None)
    except Exception:  # noqa: BLE001
        return []
    return [
        str(row.get("ticker") or "").strip().upper()
        for row in universe_summary.get("stocks", [])
        if str(row.get("ticker") or "").strip()
    ]


def _latest_cache_ready_for_universe(warm_universe: str) -> tuple[bool, dict[str, object]]:
    tickers = _universe_tickers_for_refresh_check(warm_universe)
    if not tickers:
        return False, {"reason": "No tickers were available for cache freshness check."}

    cache_status = get_price_cache_status(tickers=tickers, interval="1d")
    expected_date = _expected_latest_complete_eod_date()
    requested_count = int(cache_status.get("requested_ticker_count") or len(tickers))
    distribution = cache_status.get("latest_price_date_distribution") or {}
    expected_date_count = int(distribution.get(expected_date) or 0)
    latest_date = str(cache_status.get("latest_price_date") or "")
    latest_date_count = int(cache_status.get("latest_price_date_count") or 0)
    latest_available_coverage_pct = round(100 * latest_date_count / requested_count, 2) if requested_count else 0.0
    best_available_date = latest_date if latest_date >= expected_date else expected_date
    best_available_count = latest_date_count if latest_date >= expected_date else expected_date_count
    coverage_pct = round(100 * best_available_count / requested_count, 2) if requested_count else 0.0

    ready = bool(cache_status.get("enabled")) and not cache_status.get("error") and coverage_pct >= AUTO_REFRESH_MIN_COVERAGE_PCT
    reason = (
        f"{coverage_pct}% of {requested_count} tickers have latest EOD {best_available_date}."
        if ready
        else f"Only {coverage_pct}% of {requested_count} tickers have latest EOD {best_available_date}."
    )
    if cache_status.get("error"):
        reason = str(cache_status.get("error"))
    if not cache_status.get("enabled"):
        reason = "SQLite price cache is disabled."
    return ready, {
        "reason": reason,
        "warm_universe": warm_universe,
        "expected_latest_eod_date": expected_date,
        "latest_price_date": latest_date or None,
        "latest_price_date_count": latest_date_count,
        "latest_available_coverage_pct": latest_available_coverage_pct,
        "expected_date_count": expected_date_count,
        "requested_ticker_count": requested_count,
        "coverage_pct": coverage_pct,
        "min_coverage_pct": AUTO_REFRESH_MIN_COVERAGE_PCT,
        "cache_status": cache_status,
    }


def _warn_if_cache_not_ready(
    label: str,
    *,
    universe: str = "full_nse",
    period: str = "2y",
    interval: str = "1d",
    include_bse: bool | None = None,
) -> dict[str, object]:
    """Read-only cache-readiness check. Never starts a refresh — the price cache is only ever
    written to from the sidebar's "Refresh price cache" button (or the daily_refresh CLI /
    warm_price_history_cache MCP tool). If coverage is short, just warn and proceed with
    whatever's cached."""
    warm_universe = _daily_refresh_universe(universe)
    cache_ready, cache_readiness = _latest_cache_ready_for_universe(warm_universe)
    if cache_ready:
        st.caption(
            f"{cache_readiness.get('coverage_pct')}% of {warm_universe} already has latest EOD "
            f"{cache_readiness.get('latest_price_date') or cache_readiness.get('expected_latest_eod_date')}."
        )
    else:
        st.warning(
            f"Price cache for {warm_universe} isn't fully up to date "
            f"({cache_readiness.get('coverage_pct')}% at the latest EOD date). "
            "Refresh it from 'Price History Cache' in the sidebar. Continuing with available cached data."
        )
    return cache_readiness


stock_tab, sector_tab, universe_tab, sector_analytics_tab, rrg_tab, industry_tab, indices_tab, breadth_tab, crossover_tab, gainers_tab = st.tabs(
    [
        "Stock Scan",
        "Sector Rotation",
        "Universe",
        "Sector Analytics",
        "RRG",
        "Industry Analytics",
        "Market Indices",
        "Market Breadth",
        "MA Crossovers",
        "Top Gainers",
    ]
)


def _pct(value):
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return None


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display_ticker(ticker: object) -> str:
    value = str(ticker or "")
    for suffix in (".NS", ".BO"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _plotly_selection_points(state: object) -> list[dict]:
    if not state:
        return []
    selection = state.get("selection") if isinstance(state, dict) else getattr(state, "selection", None)
    if not selection:
        return []
    points = selection.get("points") if isinstance(selection, dict) else getattr(selection, "points", None)
    return points or []


def _sector_id_from_plotly_state(state: object, sectors: list[dict]) -> str | None:
    sector_id_by_name = {str(row.get("name")): str(row.get("sector_id")) for row in sectors if row.get("sector_id")}
    for point in _plotly_selection_points(state):
        customdata = point.get("customdata") if isinstance(point, dict) else None
        if isinstance(customdata, (list, tuple)) and customdata:
            return str(customdata[0])
        if customdata:
            return str(customdata)
        y_value = point.get("y") if isinstance(point, dict) else None
        if y_value in sector_id_by_name:
            return sector_id_by_name[str(y_value)]
    return None


def _sector_row_by_id(sectors: list[dict], sector_id: str | None) -> dict:
    if sector_id:
        for row in sectors:
            if str(row.get("sector_id")) == str(sector_id):
                return row
    return sectors[0] if sectors else {}


def _sector_card_panel(title: str, note: str, rows_html: str, *, scroll: bool = False) -> None:
    body_class = "sector-side-scroll" if scroll else ""
    if body_class:
        rows_html = f"<div class='{body_class}'>{rows_html}</div>"
    st.markdown(
        (
            '<div class="sector-side-panel">'
            f'<div class="sector-side-title">{escape(title)}</div>'
            f'<div class="sector-side-note">{escape(note)}</div>'
            f"{rows_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _safe_streamlit_key(value: object) -> str:
    text = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in cleaned.split("_") if part) or "item"


def _average_numeric(rows: list[dict], field: str) -> float | None:
    values = [_number(row.get(field)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _market_cap_cr(value: object) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if abs(number) > 10_000_000:
        return number / 10_000_000
    return number


def _industry_stock_rows(constituent_stocks: list[dict], industry: str) -> list[dict]:
    industry_key = str(industry or "").strip().casefold()
    rows = [
        row
        for row in constituent_stocks
        if str(row.get("industry") or row.get("basic_industry") or "").strip().casefold() == industry_key
    ]
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("passes_filter")),
            _number(row.get("rs_rating")) or 0,
            _number(row.get("return_20d_pct")) or -999,
        ),
        reverse=True,
    )


def _industry_stock_frame(rows: list[dict]) -> pd.DataFrame:
    data = []
    for row in rows:
        market_cap = _market_cap_cr(row.get("market_cap"))
        data.append(
            {
                "Stock Name": _display_ticker(row.get("ticker")),
                "Company": row.get("name"),
                "RS Rating": row.get("rs_rating"),
                "Market Cap (Cr)": None if market_cap is None else round(market_cap, 2),
                "1 Day Return (%)": row.get("return_1d_pct"),
                "1 Week Return (%)": row.get("return_5d_pct"),
                "1 Month Returns (%)": row.get("return_20d_pct"),
                "3 Month Returns (%)": row.get("return_60d_pct"),
                "% from 52W High": row.get("distance_from_52w_high_pct"),
                "Passes Filter": row.get("passes_filter"),
                "Latest Date": row.get("latest_date"),
            }
        )
    return pd.DataFrame(data)


def _render_sector_industry_overview(
    industry_row: dict,
    industry_stocks: list[dict],
    *,
    selected_sector_row: dict,
    metric_label: str,
) -> None:
    industry_name = str(industry_row.get("industry") or "Industry")
    sector_name = str(industry_row.get("sector") or selected_sector_row.get("name") or "Sector")
    key_prefix = _safe_streamlit_key(f"industry_detail_{sector_name}_{industry_name}")

    st.divider()
    title_col, action_col = st.columns([4, 1])
    title_col.subheader(f"{industry_name} Overview")
    if action_col.button("Close overview", key=f"{key_prefix}_close", use_container_width=True):
        st.session_state.pop("sector_analytics_industry_detail", None)
        st.rerun()

    group_market_cap = sum(
        value
        for value in (_market_cap_cr(row.get("market_cap")) for row in industry_stocks)
        if value is not None
    )
    passing_count = int(_number(industry_row.get("passing_count")) or 0)
    eligible_count = int(_number(industry_row.get("eligible_count")) or 0)
    stock_count = int(_number(industry_row.get("stock_count")) or len(industry_stocks))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Industry", industry_name)
    c2.metric("Number of Stocks", stock_count)
    c3.metric("Group Market Cap", f"{group_market_cap:,.0f} Cr" if group_market_cap else "n/a")
    c4.metric("Contribution", f"{_number(industry_row.get('contribution_pct')) or 0:.1f}%")
    c5.metric("Passing", passing_count, delta=f"of {eligible_count} eligible")

    st.caption(
        f"{metric_label}. Industry contribution = passing stocks in this industry / eligible stocks in {sector_name}."
    )

    perf_rows = [
        {"Time Frame": "1D", "Industry Performance (%)": _average_numeric(industry_stocks, "return_1d_pct")},
        {"Time Frame": "1W", "Industry Performance (%)": _average_numeric(industry_stocks, "return_5d_pct")},
        {"Time Frame": "1M", "Industry Performance (%)": _average_numeric(industry_stocks, "return_20d_pct")},
        {"Time Frame": "3M", "Industry Performance (%)": _average_numeric(industry_stocks, "return_60d_pct")},
    ]
    perf_rows = [row for row in perf_rows if row["Industry Performance (%)"] is not None]
    chart_col, breadth_col = st.columns([1.5, 1])
    with chart_col:
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            perf_fig = go.Figure()
            perf_fig.add_trace(
                go.Bar(
                    x=perf_df["Time Frame"],
                    y=perf_df["Industry Performance (%)"],
                    marker_color=[
                        "#16a34a" if (_number(value) or 0) >= 0 else "#dc2626"
                        for value in perf_df["Industry Performance (%)"]
                    ],
                    text=perf_df["Industry Performance (%)"].map(lambda value: f"{value:.1f}%"),
                    textposition="auto",
                )
            )
            perf_fig.update_layout(
                title="Industry Performance",
                yaxis_title="Return (%)",
                height=360,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(perf_fig, width="stretch")
        else:
            st.info("No return window data was available for this industry.")
    with breadth_col:
        if eligible_count:
            breadth_fig = go.Figure(
                go.Pie(
                    labels=["Passing", "Not passing"],
                    values=[passing_count, max(eligible_count - passing_count, 0)],
                    hole=0.58,
                    marker_colors=["#10b981", "#64748b"],
                    textinfo="percent+label",
                )
            )
            breadth_fig.update_layout(
                title="Filter Breadth",
                height=360,
                margin=dict(l=10, r=10, t=45, b=10),
                showlegend=False,
            )
            st.plotly_chart(breadth_fig, width="stretch")
        else:
            st.info("No eligible stocks for the active filter.")

    stock_frame = _industry_stock_frame(industry_stocks)
    if stock_frame.empty:
        st.info("No stock rows were available for this industry.")
        return

    st.markdown("**Industry Stocks**")
    st.dataframe(stock_frame, width="stretch", hide_index=True)

    rs_fig = go.Figure()
    rs_fig.add_trace(
        go.Bar(
            x=stock_frame.head(25)["RS Rating"],
            y=stock_frame.head(25)["Stock Name"],
            orientation="h",
            marker_color="#10b981",
            text=stock_frame.head(25)["RS Rating"].map(lambda value: "n/a" if pd.isna(value) else f"{value:.0f}"),
            textposition="auto",
        )
    )
    rs_fig.update_layout(
        title=f"Top RS Stocks in {industry_name}",
        xaxis_title="RS Rating",
        yaxis=dict(autorange="reversed"),
        height=max(340, min(760, 30 * len(stock_frame.head(25)) + 90)),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(rs_fig, width="stretch")


def _render_sector_industry_cards(industries: list[dict], *, key_prefix: str) -> dict | None:
    if not industries:
        _sector_card_panel(
            "STRONG Industries",
            "% indicates the contribution of industry towards sector's performance",
            '<div class="sector-side-note">No industry contribution data for this sector.</div>',
        )
        return None

    st.markdown("**STRONG Industries**")
    st.caption("% indicates the contribution of industry towards sector's performance")
    clicked: dict | None = None
    active_state = st.session_state.get("sector_analytics_industry_detail") or {}
    active_industry = str(active_state.get("industry") or "")
    for idx, row in enumerate(industries[:8]):
        industry = str(row.get("industry") or "n/a")
        sector = str(row.get("sector") or "n/a")
        contribution = _number(row.get("contribution_pct")) or 0.0
        button_label = f"{industry} · {sector}"
        if industry == active_industry:
            button_label = f"Selected: {button_label}"
        c1, c2 = st.columns([3.2, 0.8])
        if c1.button(
            button_label,
            key=f"{key_prefix}_industry_{idx}_{_safe_streamlit_key(industry)}",
            use_container_width=True,
        ):
            clicked = row
        c2.metric(" ", f"{contribution:.1f}%")
    return clicked


def _render_sector_stock_cards(stocks: list[dict]) -> None:
    if not stocks:
        _sector_card_panel(
            "STRONG Stocks",
            "Number after stock name indicates RS",
            '<div class="sector-side-note">No stock rows were available for this sector.</div>',
        )
        return
    rows = []
    for row in stocks[:30]:
        ticker = escape(_display_ticker(row.get("ticker")))
        industry = escape(str(row.get("industry") or "n/a"))
        sector = escape(str(row.get("sector") or "n/a"))
        rs_rating = _number(row.get("rs_rating"))
        rs_text = "n/a" if rs_rating is None else f"{rs_rating:.0f}"
        rows.append(
            '<div class="sector-side-row">'
            '<div class="sector-side-main">'
            f'<span class="sector-side-name">{ticker}</span>'
            f'<span class="sector-side-pill">{industry}</span>'
            f'<span class="sector-side-pill">{sector}</span>'
            "</div>"
            f'<div class="sector-side-value">{rs_text}</div>'
            "</div>"
        )
    _sector_card_panel("STRONG Stocks", "Number after stock name indicates RS", "".join(rows), scroll=True)


def _render_sector_stock_action_buttons(stocks: list[dict], *, key_prefix: str) -> str | None:
    if not stocks:
        return None
    st.caption("Click a stock to run the full research agent.")
    clicked: str | None = None
    for idx, row in enumerate(stocks[:12]):
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        display = _display_ticker(ticker)
        industry = str(row.get("industry") or "n/a")
        rs_rating = _number(row.get("rs_rating"))
        rs_text = "n/a" if rs_rating is None else f"{rs_rating:.0f}"
        c1, c2, c3 = st.columns([0.9, 1.8, 0.45])
        if c1.button(display, key=f"{key_prefix}_research_{idx}_{ticker}", use_container_width=True):
            clicked = ticker
        c2.caption(industry)
        c3.metric("RS", rs_text)
    return clicked


def _render_stock_research_agent_result(result: dict) -> None:
    verdict = result.get("verdict") or {}
    analysis = result.get("analysis") or {}
    llm = result.get("llm") or {}
    workflow = result.get("workflow") or {}
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Agent verdict", verdict.get("action") or "n/a")
    c2.metric("Score", verdict.get("score") or analysis.get("final_score"))
    c3.metric("Risk", verdict.get("risk_score") or analysis.get("risk_score"))
    c4.metric("Confidence", f"{verdict.get('confidence', analysis.get('confidence', 0))}%")
    c5.metric("Workflow", workflow.get("backend") or "n/a")
    c6.metric("LLM", "Used" if llm.get("used") else "Missing/off")

    st.markdown("**Research Summary**")
    st.write(result.get("executive_summary") or result.get("deterministic_summary") or "No summary generated.")

    warnings = result.get("warnings") or []
    if warnings:
        with st.expander("Data gaps and warnings", expanded=False):
            st.write(warnings)

    t1, t2, t3, t4, t5, t6 = st.tabs(["Technicals", "Fundamentals", "News", "Exchange filings", "Intelligence", "Raw"])
    with t1:
        st.json(analysis.get("latest_indicators", {}))
        st.json(analysis.get("chart_patterns", {}))
    with t2:
        st.json(analysis.get("fundamentals", {}))
        st.json((analysis.get("analyst_insights") or {}).get("consensus", {}))
        st.json(analysis.get("stock_events", {}))
    with t3:
        news_rows = analysis.get("news") or []
        st.dataframe(pd.DataFrame(news_rows), width="stretch") if news_rows else st.info("No recent news rows returned.")
    with t4:
        exchange = result.get("exchange_announcements") or {}
        rows = exchange.get("announcements") or []
        if rows:
            display_rows = [
                {
                    "exchange": row.get("exchange"),
                    "published_at": row.get("published_at"),
                    "headline": row.get("headline"),
                    "categories": ", ".join(row.get("material_categories") or []),
                    "attachment_url": row.get("attachment_url"),
                    "has_pdf_text": bool(row.get("attachment_text_excerpt")),
                }
                for row in rows
            ]
            st.dataframe(pd.DataFrame(display_rows), width="stretch")
            parsed_rows = [row for row in rows if row.get("attachment_text_excerpt")]
            for row in parsed_rows[:2]:
                with st.expander(str(row.get("headline") or "Parsed PDF excerpt"), expanded=False):
                    st.write(row.get("attachment_text_excerpt"))
        else:
            st.info("No NSE/BSE announcement rows returned for this stock.")
    with t5:
        st.json(analysis.get("company_intelligence", {}))
        st.write("Watch items")
        st.write(result.get("watch_items", []))
    with t6:
        st.json(result)


def _top_gainer_industry_from_plotly_state(state: object, industries: list[dict]) -> str | None:
    label_to_industry = {str(row.get("label")): str(row.get("industry")) for row in industries if row.get("industry")}
    for point in _plotly_selection_points(state):
        customdata = point.get("customdata") if isinstance(point, dict) else None
        if isinstance(customdata, (list, tuple)) and customdata:
            return str(customdata[0])
        if customdata:
            return str(customdata)
        y_value = point.get("y") if isinstance(point, dict) else None
        if y_value in label_to_industry:
            return label_to_industry[str(y_value)]
    return None


def _top_gainer_stock_frame(rows: list[dict], *, include_context: bool = False) -> pd.DataFrame:
    data = []
    for row in rows or []:
        item = {
            "Stock": _display_ticker(row.get("ticker")),
            "Return (%)": row.get("selected_return_pct"),
        }
        if include_context:
            item.update(
                {
                    "RS": row.get("rs_rating"),
                    "Industry": row.get("industry"),
                    "Sector": row.get("sector"),
                    "Market Cap (Cr)": row.get("market_cap_cr"),
                    "Latest Date": row.get("latest_date"),
                }
            )
        data.append(item)
    return pd.DataFrame(data)


def _top_gainer_industry_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Industry": row.get("industry"),
                "Sector": row.get("sector"),
                "Industry %": row.get("industry_gainer_pct"),
                "Avg Return (%)": row.get("avg_selected_return_pct"),
                "Top Return (%)": row.get("top_return_pct"),
                "Gainers": row.get("passing_count"),
                "Eligible": row.get("eligible_count"),
                "Formula": row.get("formula"),
                "Top Stock": _display_ticker(row.get("top_stock")),
            }
            for row in rows
        ]
    )


def _ma_crossover_stock_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Stock": _display_ticker(row.get("ticker")),
                "Signal": row.get("signal"),
                "Cross Date": row.get("cross_date"),
                "Days Since Cross": row.get("periods_since_cross"),
                "Sector": row.get("sector"),
                "Industry": row.get("industry"),
                "RS": row.get("rs_rating"),
                "Close": row.get("close"),
                "50 DMA": row.get("sma_50"),
                "200 DMA": row.get("sma_200"),
                "DMA Gap (%)": row.get("sma_gap_pct"),
                "Return Since Cross (%)": row.get("return_since_cross_pct"),
                "1M Return (%)": row.get("return_20d_pct"),
                "3M Return (%)": row.get("return_60d_pct"),
                "Market Cap (Cr)": row.get("market_cap_cr"),
                "Latest Date": row.get("latest_date"),
            }
            for row in rows
        ]
    )


def _ma_crossover_summary_frame(rows: list[dict], group_key: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    title = "Sector" if group_key == "sector" else "Industry"
    return pd.DataFrame(
        [
            {
                title: row.get(group_key),
                "Signals": row.get("signal_count"),
                "Bullish": row.get("bullish_count"),
                "Bearish": row.get("bearish_count"),
                "Avg RS": row.get("avg_rs_rating"),
                "Avg Return Since Cross (%)": row.get("avg_return_since_cross_pct"),
                "Latest Cross Date": row.get("latest_cross_date"),
                "Top Stock": _display_ticker(row.get("top_stock")),
            }
            for row in rows
        ]
    )


def _movement_arrow(value, threshold: float = 0.001) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    if number > threshold:
        return "↑"
    if number < -threshold:
        return "↓"
    return "→"


def _movement_display(value, threshold: float = 0.001) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return f"{_movement_arrow(number, threshold)} {number * 100:.2f}%"


def _score_display(value, strong: float = 60.0, weak: float = 40.0) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    if number >= strong:
        arrow = "↑"
    elif number <= weak:
        arrow = "↓"
    else:
        arrow = "→"
    return f"{arrow} {number:.1f}"


def _color_for_status(status: str | None) -> str:
    return {
        "currently_running": "#16a34a",
        "upcoming": "#2563eb",
        "watch": "#ca8a04",
        "avoid": "#dc2626",
    }.get(str(status or "").lower(), "#64748b")


def _sector_decision_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Sector": row.get("name"),
                "Status": row.get("movement_status"),
                "Stage": row.get("stage"),
                "Score": row.get("rotation_score"),
                "5D move": _movement_display(row.get("return_5d")),
                "20D move": _movement_display(row.get("return_20d")),
                "60D move": _movement_display(row.get("return_60d")),
                "20D RS": _movement_display(row.get("rs_20d")),
                "60D RS": _movement_display(row.get("rs_60d")),
                "Trend": _score_display(row.get("trend_score"), strong=65, weak=45),
                "Breadth": _score_display(row.get("breadth_score"), strong=60, weak=40),
            }
            for row in rows or []
        ]
    )


def _analytics_table(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for source, target in {
        "return_1d": "1D",
        "return_5d": "1W",
        "return_20d": "1M",
        "return_60d": "3M",
        "return_120d": "6M",
        "rs_20d": "20D RS",
        "rs_60d": "60D RS",
    }.items():
        if source in df.columns:
            df[target] = df[source].apply(_movement_display)
    for source, target in {
        "trend_score": "Trend",
        "acceleration_score": "Accel",
        "breadth_score": "Breadth",
        "relative_strength_score": "RS Score",
        "composite_score": "Score",
    }.items():
        if source in df.columns:
            df[target] = df[source].apply(_score_display)
    available = [column for column in columns if column in df.columns]
    return df[available]


def _pct_bar_frame(rows: list[dict], label_key: str, fields: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    data = []
    for row in rows:
        label = row.get(label_key)
        for field in fields:
            data.append({"name": label, "metric": field, "value": row.get(field)})
    return pd.DataFrame(data)


def _rrg_quadrant_color(quadrant: str | None) -> str:
    return {
        "Leading": "#15803d",
        "Improving": "#2563eb",
        "Lagging": "#dc2626",
        "Weakening": "#d97706",
    }.get(str(quadrant or ""), "#64748b")


def _rrg_axis_range(values: list[float]) -> list[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return [96.5, 103.5]
    low = min(min(clean), 100) - 0.4
    high = max(max(clean), 100) + 0.4
    span = max(high - low, 5.5)
    return [round(low, 2), round(low + span, 2)]


def _rrg_figure(trails: list[dict]) -> go.Figure:
    all_x = []
    all_y = []
    for trail in trails:
        for point in trail.get("points", []):
            all_x.append(_number(point.get("rs_ratio")))
            all_y.append(_number(point.get("rs_momentum")))
    x_range = _rrg_axis_range(all_x)
    y_range = _rrg_axis_range(all_y)
    fig = go.Figure()
    quadrant_shapes = [
        (x_range[0], 100, 100, y_range[1], "#d7e3ff", "Improving", x_range[0] + 0.08, y_range[1] - 0.08, "left", "top", "#0000cc"),
        (100, x_range[1], 100, y_range[1], "#d8f3d6", "Leading", x_range[1] - 0.08, y_range[1] - 0.08, "right", "top", "#17652c"),
        (x_range[0], 100, y_range[0], 100, "#f5c6c6", "Lagging", x_range[0] + 0.08, y_range[0] + 0.08, "left", "bottom", "#cc1f1a"),
        (100, x_range[1], y_range[0], 100, "#ffe5bf", "Weakening", x_range[1] - 0.08, y_range[0] + 0.08, "right", "bottom", "#f59e0b"),
    ]
    for x0, x1, y0, y1, color, label, text_x, text_y, xanchor, yanchor, font_color in quadrant_shapes:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, opacity=0.8, line_width=0, layer="below")
        fig.add_annotation(
            x=text_x,
            y=text_y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=14, color=font_color),
            xanchor=xanchor,
            yanchor=yanchor,
        )
    for trail in trails:
        points = trail.get("points", [])
        if not points:
            continue
        df = pd.DataFrame(points)
        current = points[-1]
        color = _rrg_quadrant_color(current.get("quadrant"))
        fig.add_trace(
            go.Scatter(
                x=df["rs_ratio"],
                y=df["rs_momentum"],
                mode="lines+markers",
                name=trail.get("name"),
                line=dict(color=color, width=2.2),
                marker=dict(size=5, color=color),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Date: %{customdata[1]}<br>"
                    "RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<br>Quadrant: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
                customdata=df[["name", "date", "quadrant"]],
            )
        )
        if len(points) >= 2:
            previous = points[-2]
            fig.add_annotation(
                x=current.get("rs_ratio"),
                y=current.get("rs_momentum"),
                ax=previous.get("rs_ratio"),
                ay=previous.get("rs_momentum"),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=color,
            )
        fig.add_trace(
            go.Scatter(
                x=[current.get("rs_ratio")],
                y=[current.get("rs_momentum")],
                mode="markers+text",
                text=[trail.get("name")],
                textposition="top center",
                showlegend=False,
                textfont=dict(size=11, color="#111827"),
                marker=dict(size=9, color=color, symbol="triangle-up", line=dict(color=color, width=1)),
                hovertemplate=f"<b>{trail.get('name')}</b><extra></extra>",
            )
        )
    fig.add_vline(x=100, line_color="#b7b7b7", line_width=1.2)
    fig.add_hline(y=100, line_color="#b7b7b7", line_width=1.2)
    fig.add_annotation(
        x=0.5,
        y=0.03,
        xref="paper",
        yref="paper",
        text="<b>JdK RS-Ratio</b>",
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.add_annotation(
        x=0.02,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>JdK RS-Momentum</b>",
        textangle=-90,
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(
            range=x_range,
            showgrid=True,
            gridcolor="#d1d5db",
            zeroline=False,
            tickfont=dict(color="#64748b"),
            linecolor="#4b5563",
            mirror=True,
        ),
        yaxis=dict(
            range=y_range,
            showgrid=True,
            gridcolor="#d1d5db",
            zeroline=False,
            tickfont=dict(color="#64748b"),
            linecolor="#4b5563",
            mirror=True,
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#111827"),
        height=640,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def _rrg_points_frame(points: list[dict]) -> pd.DataFrame:
    if not points:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "sector_id": row.get("sector_id"),
                "name": row.get("name"),
                "quadrant": row.get("quadrant"),
                "stage": row.get("stage"),
                "status": row.get("status"),
                "direction": row.get("direction"),
                "rotation_score": row.get("rotation_score"),
                "rs_ratio": row.get("rs_ratio"),
                "rs_momentum": row.get("rs_momentum"),
                "RS-Ratio vs 100": f"{row.get('x_rs_60d_pct')}%",
                "RS-Momentum vs 100": f"{row.get('y_rs_momentum_pct')}%",
                "20D return": f"{row.get('return_20d_pct')}%",
                "60D return": f"{row.get('return_60d_pct')}%",
                "Trend": _score_display(row.get("trend_score")),
                "Data": row.get("price_method"),
            }
            for row in points
        ]
    )

with stock_tab:
    watchlists = load_watchlists()
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
    group = c1.selectbox("Watchlist", ["all"] + list(watchlists.keys()))
    period = c2.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
    interval = c3.selectbox("Interval", ["1d", "1wk"], index=0)
    include_news = c4.checkbox("Include news sentiment", value=True)
    c1, c2, c3 = st.columns([1, 2.2, 1])
    run = c1.button("Run scan")
    custom_ticker = c2.text_input("Analyze ticker", placeholder="AAPL or RELIANCE.NS")
    analyze_one = c3.button("Analyze ticker")

    if run or analyze_one:
        _warn_if_cache_not_ready("stock scan", period=period, interval=interval)
        with st.spinner("Scanning..."):
            if analyze_one and custom_ticker.strip():
                rows = [
                    analyze_stock(
                        custom_ticker,
                        period=period,
                        interval=interval,
                        include_news=include_news,
                    )
                ]
            else:
                rows = rank_watchlist(
                    None if group == "all" else group,
                    period=period,
                    interval=interval,
                    include_news=include_news,
                )
        if not rows:
            st.warning("No tickers are configured for this selection.")
            st.stop()

        df = pd.DataFrame(
            [
                {
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "latest_indicators",
                        "fundamentals",
                        "chart_patterns",
                        "news",
                        "company_intelligence",
                        "analyst_insights",
                        "stock_events",
                        "reasons",
                        "risks",
                        "metadata",
                    }
                }
                for row in rows
            ]
        )
        st.dataframe(df, width="stretch")

        selected = st.selectbox("Inspect ticker", [row["ticker"] for row in rows])
        detail = next(row for row in rows if row["ticker"] == selected)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Signal", detail.get("signal"))
        c2.metric("Score", detail.get("final_score"))
        c3.metric("Confidence", f"{detail.get('confidence')}%")
        c4.metric("Risk Score", detail.get("risk_score"))
        c5.metric("Data Points", detail.get("metadata", {}).get("data_points", 0))

        warnings = detail.get("metadata", {}).get("warnings", [])
        if warnings:
            st.warning("\n".join(warnings))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Reasons")
            st.write(detail.get("reasons", []))
        with c2:
            st.subheader("Risks")
            st.write(detail.get("risks", []))

        with st.expander("Fundamentals", expanded=False):
            st.json(detail.get("fundamentals", {}))
        with st.expander("Latest indicators", expanded=False):
            st.json(detail.get("latest_indicators", {}))
        with st.expander("Chart patterns", expanded=False):
            st.json(detail.get("chart_patterns", {}))
        with st.expander("News", expanded=False):
            st.write(detail.get("news", []))

        prices = add_indicators(get_price_history(selected, period=period, interval=interval))
        if not prices.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=prices["date"],
                    open=prices["open"],
                    high=prices["high"],
                    low=prices["low"],
                    close=prices["close"],
                    name="Price",
                )
            )
            fig.add_trace(go.Scatter(x=prices["date"], y=prices["sma_20"], name="SMA 20"))
            fig.add_trace(go.Scatter(x=prices["date"], y=prices["sma_50"], name="SMA 50"))
            fig.add_trace(go.Scatter(x=prices["date"], y=prices["sma_200"], name="SMA 200"))
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(f"No price history is cached for {selected}. {CACHE_REFRESH_HINT}")
    else:
        st.info("Choose a watchlist and run a scan, or enter a ticker.")

with sector_tab:
    c1, c2, c3, c4 = st.columns(4)
    sector_period_choice = c1.selectbox(
        "Sector period",
        ["Auto", "6mo", "1y", "2y"],
        index=0,
        help=(
            "Historical window for sector movement. Auto uses fresh 1y price history and compares "
            "5D, 20D, 60D, and 120D movement, relative strength, trend, acceleration, and breadth."
        ),
    )
    sector_auto_period = sector_period_choice == "Auto"
    sector_period = "auto" if sector_auto_period else sector_period_choice
    sector_analysis_period = "1y" if sector_auto_period else sector_period_choice
    sector_interval = c2.selectbox(
        "Sector interval",
        ["1d", "1wk"],
        index=0,
        help="Daily candles detect current rotation earlier; weekly candles smooth noise for slower sector trends.",
    )
    sector_limit = c3.slider(
        "Sectors",
        min_value=5,
        max_value=13,
        value=10,
        help="How many sectors to rank and show in the rotation charts.",
    )
    include_fundamentals = c4.checkbox(
        "Stock fundamentals",
        value=True,
        help="Include available free fundamentals while ranking stock candidates inside the selected sector.",
    )
    c1, c2, c3 = st.columns([1, 1, 1.2])
    breadth_sample = c1.slider(
        "Breadth sample",
        min_value=3,
        max_value=10,
        value=6,
        help="Number of constituent stocks sampled to check whether a sector move is broad or narrow.",
    )
    stock_sample = c2.slider(
        "Stocks per sector",
        min_value=3,
        max_value=10,
        value=6,
        help="Number of stock candidates ranked inside the selected or top sector.",
    )
    run_sector = c3.button("Run sector rotation", help="Use latest cached data; start a background exchange refresh only if latest EOD is missing.")
    st.caption("Auto period recalculates from the latest available local data. The table arrows show whether each movement or RS value is rising, falling, or flat.")

    sector_definitions = list_sector_definitions()
    sector_options = {f"{row['name']} ({sector_id})": sector_id for sector_id, row in sector_definitions.items()}
    selected_label = st.selectbox("Sector stock candidates", list(sector_options), index=0)
    selected_sector = sector_options[selected_label]
    run_selected_sector = st.button("Run selected sector")

    if run_sector or run_selected_sector:
        _warn_if_cache_not_ready("sector rotation", period=sector_analysis_period, interval=sector_interval)
        with st.spinner("Running sector workflow..."):
            st.session_state["sector_rotation_workflow_result"] = run_sector_rotation_workflow(
                period=sector_period,
                interval=sector_interval,
                auto_period=sector_auto_period,
                max_sectors=sector_limit,
                max_breadth_stocks=breadth_sample,
                stocks_per_sector=stock_sample,
                include_fundamentals=include_fundamentals,
                selected_sector=selected_sector if run_selected_sector else None,
            )

    workflow_result = st.session_state.get("sector_rotation_workflow_result")
    if workflow_result:
        rotation = workflow_result.get("rotation", {})
        ranked = workflow_result.get("ranked_stocks", {})
        decision = workflow_result.get("decision_summary", {})
        explanations = workflow_result.get("indicator_explanations", {})
        inputs = workflow_result.get("inputs", {})
        period_label = (
            f"Auto -> {inputs.get('analysis_period')} with {', '.join(inputs.get('movement_windows', []))} windows"
            if inputs.get("auto_period")
            else inputs.get("analysis_period")
        )
        st.caption(
            f"Last workflow run completed at {workflow_result.get('completed_at')} | "
            f"period: {period_label} | interval: {inputs.get('interval')} | latest available local data"
        )

        with st.expander("Indicator, Filter, and Column Guide", expanded=False):
            guide_tabs = st.tabs(["Filters", "Sector Columns", "Stock Columns", "Charts"])
            with guide_tabs[0]:
                st.dataframe(pd.DataFrame(explanations.get("sector_filters", [])), width="stretch")
            with guide_tabs[1]:
                st.dataframe(pd.DataFrame(explanations.get("sector_columns", [])), width="stretch")
            with guide_tabs[2]:
                st.dataframe(pd.DataFrame(explanations.get("stock_columns", [])), width="stretch")
            with guide_tabs[3]:
                st.dataframe(pd.DataFrame(explanations.get("chart_explanations", [])), width="stretch")

        sectors = rotation.get("sectors", [])
        if not sectors:
            st.warning("No sector index data was returned by the market data provider.")
            st.stop()

        st.subheader("Sector Decision")
        target_sector = decision.get("target_next_sector") or {}
        top_stock = decision.get("top_stock_target") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target next sector", target_sector.get("name") or "n/a")
        c2.metric("Sector status", target_sector.get("movement_status") or "n/a")
        c3.metric("Sector score", target_sector.get("rotation_score") or "n/a")
        c4.metric("Top stock to inspect", top_stock.get("ticker") or "n/a")
        st.info(decision.get("headline", "Run sector rotation to get the current sector decision."))

        current_running = _sector_decision_frame(decision.get("currently_running_sectors", []))
        upcoming = _sector_decision_frame(decision.get("upcoming_sectors", []))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Currently Running Sectors**")
            if not current_running.empty:
                st.dataframe(current_running, width="stretch")
            else:
                st.write("No fully confirmed leadership sector in this run.")
        with c2:
            st.markdown("**Upcoming Sectors**")
            if not upcoming.empty:
                st.dataframe(upcoming, width="stretch")
            else:
                st.write("No emerging sector qualified in this run.")

        with st.expander("Root cause for target sector", expanded=True):
            for cause in target_sector.get("root_causes", []):
                st.write(f"- {cause}")

        sector_rows = []
        for row in sectors:
            metrics = row.get("metrics", {})
            breadth = row.get("breadth", {})
            relative = row.get("relative_strength", {})
            sector_rows.append(
                {
                    "sector_id": row.get("sector_id"),
                    "name": row.get("name"),
                    "stage": row.get("stage"),
                    "rotation_score": row.get("rotation_score"),
                    "movement_arrow": _movement_arrow(metrics.get("return_20d")),
                    "return_5d": metrics.get("return_5d"),
                    "return_20d": metrics.get("return_20d"),
                    "return_60d": metrics.get("return_60d"),
                    "return_5d_pct": _pct(metrics.get("return_5d")),
                    "return_20d_pct": _pct(metrics.get("return_20d")),
                    "return_60d_pct": _pct(metrics.get("return_60d")),
                    "5D move": _movement_display(metrics.get("return_5d")),
                    "20D move": _movement_display(metrics.get("return_20d")),
                    "60D move": _movement_display(metrics.get("return_60d")),
                    "rs_20d": relative.get("vs_benchmark_20d"),
                    "rs_60d": relative.get("vs_benchmark_60d"),
                    "rs_20d_pct": _pct(relative.get("vs_benchmark_20d")),
                    "rs_60d_pct": _pct(relative.get("vs_benchmark_60d")),
                    "20D RS": _movement_display(relative.get("vs_benchmark_20d")),
                    "60D RS": _movement_display(relative.get("vs_benchmark_60d")),
                    "trend_score": row.get("trend_score"),
                    "acceleration_score": row.get("acceleration_score"),
                    "breadth_score": breadth.get("breadth_score"),
                    "Trend": _score_display(row.get("trend_score"), strong=65, weak=45),
                    "Acceleration": _score_display(row.get("acceleration_score"), strong=55, weak=45),
                    "Breadth": _score_display(breadth.get("breadth_score"), strong=60, weak=40),
                    "above_20_pct": breadth.get("above_sma_20_percent"),
                    "above_50_pct": breadth.get("above_sma_50_percent"),
                    "above_200_pct": breadth.get("above_sma_200_percent"),
                    "movement_status": next(
                        (
                            item.get("movement_status")
                            for item in [
                                *(decision.get("currently_running_sectors", []) or []),
                                *(decision.get("upcoming_sectors", []) or []),
                                *(decision.get("avoid_or_weak_sectors", []) or []),
                            ]
                            if item.get("sector_id") == row.get("sector_id")
                        ),
                        "watch",
                    ),
                }
            )
        sector_df = pd.DataFrame(sector_rows)
        sector_display_cols = [
            "sector_id",
            "name",
            "movement_arrow",
            "movement_status",
            "stage",
            "rotation_score",
            "5D move",
            "20D move",
            "60D move",
            "20D RS",
            "60D RS",
            "Trend",
            "Acceleration",
            "Breadth",
            "above_20_pct",
            "above_50_pct",
            "above_200_pct",
        ]

        c1, c2, c3 = st.columns(3)
        top = sectors[0]
        top_metrics = top.get("metrics", {}) or {}
        top_relative = top.get("relative_strength", {}) or {}
        c1.metric("Top Sector", top.get("name"), delta=f"20D move {_movement_display(top_metrics.get('return_20d'))}")
        c2.metric("Rotation Score", top.get("rotation_score"), delta=f"20D RS {_movement_display(top_relative.get('vs_benchmark_20d'))}")
        c3.metric("Stage", top.get("stage"))

        stage_counts = sector_df["stage"].value_counts().reset_index()
        stage_counts.columns = ["stage", "count"]
        c1, c2 = st.columns([1.2, 1])
        with c1:
            bar = go.Figure()
            bar.add_trace(
                go.Bar(
                    x=sector_df["rotation_score"],
                    y=sector_df["name"],
                    orientation="h",
                    marker_color=[_color_for_status(status) for status in sector_df["movement_status"]],
                    text=sector_df["movement_status"],
                    textposition="auto",
                )
            )
            bar.update_layout(
                title="Sector Rotation Score",
                height=440,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(bar, width="stretch")
        with c2:
            pie = go.Figure()
            pie.add_trace(go.Pie(labels=stage_counts["stage"], values=stage_counts["count"], hole=0.45))
            pie.update_layout(title="Sector Stage Mix", height=440, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(pie, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            scatter = go.Figure()
            scatter.add_trace(
                go.Scatter(
                    x=sector_df["rs_20d_pct"],
                    y=sector_df["rs_60d_pct"],
                    mode="markers+text",
                    text=sector_df["sector_id"],
                    textposition="top center",
                    marker=dict(
                        size=sector_df["rotation_score"].clip(lower=20) / 2,
                        color=sector_df["acceleration_score"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Accel"),
                    ),
                )
            )
            scatter.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
            scatter.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            scatter.update_layout(
                title="Relative Strength Map",
                xaxis_title="20D RS vs Nifty (%)",
                yaxis_title="60D RS vs Nifty (%)",
                height=430,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(scatter, width="stretch")
        with c2:
            returns = go.Figure()
            returns.add_trace(go.Bar(x=sector_df["name"], y=sector_df["return_5d_pct"], name="5D"))
            returns.add_trace(go.Bar(x=sector_df["name"], y=sector_df["return_20d_pct"], name="20D"))
            returns.add_trace(go.Bar(x=sector_df["name"], y=sector_df["return_60d_pct"], name="60D"))
            returns.update_layout(
                title="Sector Returns",
                yaxis_title="Return (%)",
                barmode="group",
                height=430,
                margin=dict(l=10, r=10, t=45, b=100),
            )
            st.plotly_chart(returns, width="stretch")

        breadth_fig = go.Figure()
        breadth_fig.add_trace(go.Bar(x=sector_df["name"], y=sector_df["above_20_pct"], name="Above 20-DMA"))
        breadth_fig.add_trace(go.Bar(x=sector_df["name"], y=sector_df["above_50_pct"], name="Above 50-DMA"))
        breadth_fig.add_trace(go.Bar(x=sector_df["name"], y=sector_df["above_200_pct"], name="Above 200-DMA"))
        breadth_fig.update_layout(
            title="Sector Breadth Participation",
            yaxis_title="% constituents",
            barmode="group",
            height=430,
            margin=dict(l=10, r=10, t=45, b=100),
        )
        st.plotly_chart(breadth_fig, width="stretch")

        st.dataframe(sector_df[sector_display_cols], width="stretch")

        stocks = ranked.get("stocks", [])
        if stocks:
            stock_df = pd.DataFrame(
                [
                    {
                        "ticker": row.get("ticker"),
                        "stage": row.get("stage"),
                        "stock_score": row.get("stock_score"),
                        "rs_20d": row.get("relative_strength", {}).get("vs_sector_20d"),
                        "rs_60d": row.get("relative_strength", {}).get("vs_sector_60d"),
                        "rs_20d_pct": _pct(row.get("relative_strength", {}).get("vs_sector_20d")),
                        "rs_60d_pct": _pct(row.get("relative_strength", {}).get("vs_sector_60d")),
                        "trend_score": row.get("trend_score"),
                        "pattern_score": row.get("pattern_score"),
                        "volume_score": row.get("volume_score"),
                        "risk_quality_score": row.get("risk_quality_score"),
                        "dominant_chart_pattern": row.get("dominant_chart_pattern"),
                    }
                    for row in stocks
                ]
            )
            c1, c2 = st.columns(2)
            with c1:
                stock_score_fig = go.Figure()
                stock_score_fig.add_trace(
                    go.Bar(
                        x=stock_df["stock_score"],
                        y=stock_df["ticker"],
                        orientation="h",
                        marker_color="#0f766e",
                    )
                )
                stock_score_fig.update_layout(
                    title="Stock Targets in Selected Sector",
                    yaxis=dict(autorange="reversed"),
                    height=380,
                    margin=dict(l=10, r=10, t=45, b=10),
                )
                st.plotly_chart(stock_score_fig, width="stretch")
            with c2:
                stock_rs_fig = go.Figure()
                stock_rs_fig.add_trace(go.Bar(x=stock_df["ticker"], y=stock_df["rs_20d_pct"], name="20D RS"))
                stock_rs_fig.add_trace(go.Bar(x=stock_df["ticker"], y=stock_df["rs_60d_pct"], name="60D RS"))
                stock_rs_fig.update_layout(
                    title="Stock Relative Strength vs Sector",
                    yaxis_title="RS (%)",
                    barmode="group",
                    height=380,
                    margin=dict(l=10, r=10, t=45, b=80),
                )
                st.plotly_chart(stock_rs_fig, width="stretch")
            st.dataframe(stock_df, width="stretch")

            pick = st.selectbox("Inspect stock candidate", [row["ticker"] for row in stocks])
            pick_detail = next(row for row in stocks if row["ticker"] == pick)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stock Score", pick_detail.get("stock_score"))
            c2.metric("Stage", pick_detail.get("stage"))
            c3.metric("20D RS", pick_detail.get("relative_strength", {}).get("vs_sector_20d"))
            c4.metric("Pattern", pick_detail.get("dominant_chart_pattern") or "None")
            st.write(pick_detail.get("reasons", []))

            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=[
                        pick_detail.get("stock_score") or 0,
                        pick_detail.get("trend_score") or 0,
                        pick_detail.get("pattern_score") or 0,
                        pick_detail.get("volume_score") or 0,
                        pick_detail.get("risk_quality_score") or 0,
                    ],
                    theta=["Score", "Trend", "Pattern", "Volume", "Risk"],
                    fill="toself",
                    name=pick,
                )
            )
            radar.update_layout(
                title=f"{pick} Setup Quality",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=360,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(radar, width="stretch")

            chart_period = inputs.get("analysis_period") or sector_analysis_period
            stock_prices = add_indicators(get_price_history(pick, period=chart_period, interval=sector_interval))
            if not stock_prices.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Candlestick(
                        x=stock_prices["date"],
                        open=stock_prices["open"],
                        high=stock_prices["high"],
                        low=stock_prices["low"],
                        close=stock_prices["close"],
                        name=pick,
                    )
                )
                fig.add_trace(go.Scatter(x=stock_prices["date"], y=stock_prices["sma_20"], name="SMA 20"))
                fig.add_trace(go.Scatter(x=stock_prices["date"], y=stock_prices["sma_50"], name="SMA 50"))
                fig.add_trace(go.Scatter(x=stock_prices["date"], y=stock_prices["sma_200"], name="SMA 200"))
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No stock candidates were available for this sector.")

        warnings = workflow_result.get("warnings", [])
        if warnings:
            with st.expander("Provider warnings", expanded=False):
                st.write(warnings)
    else:
        st.info("Run sector rotation to view sector leadership and stock candidates.")

with universe_tab:
    st.subheader("Universe")
    st.caption("Audit the exact stock universe used for broad sector and industry breadth calculations.")
    universe_source_label = st.selectbox(
        "Universe source",
        ["Broad NSE Total Market", "Full NSE Equity Master", "Full BSE Equity Master", "All India NSE+BSE Master"],
        index=0,
        key="universe_source",
        help="All India uses Dhan's free public NSE+BSE instrument master and NSE sector metadata where an ISIN match exists.",
    )
    universe_source = {
        "Broad NSE Total Market": "broad",
        "Full NSE Equity Master": "full_nse",
        "Full BSE Equity Master": "full_bse",
        "All India NSE+BSE Master": "all_india",
    }[universe_source_label]
    universe_summary = list_sector_constituents(universe=universe_source)
    c1, c2, c3 = st.columns(3)
    c1.metric("Stocks", universe_summary.get("count", 0))
    c2.metric("Sectors", universe_summary.get("sector_count", 0))
    c3.metric("Refreshed", str(universe_summary.get("refreshed_at") or "n/a")[:10])
    if universe_source == "full_nse" and not universe_summary.get("count"):
        st.warning("Full NSE universe file is not built yet. Run `python scripts/refresh_universe.py --universe full` from the project directory, then refresh this page.")
    if universe_source == "full_bse" and not universe_summary.get("count"):
        st.warning("Full BSE universe file is not built yet. Run `python scripts/refresh_universe.py --universe bse` from the project directory, then refresh this page.")
    if universe_source == "all_india" and not universe_summary.get("count"):
        st.warning("All India universe file is not built yet. Run `python scripts/refresh_universe.py --universe india` from the project directory, then refresh this page.")

    with st.expander("Daily NSE/BSE universe sync", expanded=False):
        st.caption(
            "Syncs public NSE/BSE universe CSVs and instrument_master identity data only — "
            "it never touches price_history_cache. Refresh prices from the sidebar instead."
        )
        last_refresh = st.session_state.get("daily_refresh_result") or load_daily_refresh_report()
        r1, r2, r3 = st.columns(3)
        r1.metric("Last status", last_refresh.get("status", "n/a") if last_refresh else "n/a")
        r2.metric("Universe stocks", last_refresh.get("universe_stock_count", 0) if last_refresh else 0)
        r3.metric("Completed", str(last_refresh.get("completed_at", "n/a"))[:10] if last_refresh else "n/a")

        d1, d2, d3 = st.columns(3)
        daily_refresh_full_nse = d1.checkbox("NSE full file", value=True, key="daily_refresh_full_nse")
        daily_refresh_broad = d2.checkbox("NSE total market", value=True, key="daily_refresh_broad")
        daily_refresh_bse = d3.checkbox("BSE master", value=False, key="daily_refresh_bse")
        d4, d5 = st.columns(2)
        daily_refresh_india = d4.checkbox("India master", value=False, key="daily_refresh_india")
        daily_run = d5.button("Run universe sync now", key="daily_refresh_run")

        if daily_run:
            with st.spinner("Syncing NSE/BSE universe data..."):
                st.session_state["daily_refresh_result"] = run_daily_market_data_refresh(
                    refresh_broad_universe=daily_refresh_broad,
                    refresh_full_nse_universe=daily_refresh_full_nse,
                    refresh_bse_universe=daily_refresh_bse,
                    refresh_india_universe=daily_refresh_india,
                    refresh_exchange_eod=False,
                    warm_price_cache=False,
                    warm_index_cache=False,
                )
            st.rerun()

        current_refresh = st.session_state.get("daily_refresh_result") or last_refresh
        if current_refresh:
            with st.expander("Latest sync report", expanded=False):
                st.json(current_refresh)

    sector_constituent_options = {"All sectors": None}
    sector_constituent_options.update(
        {
            f"{row['sector']} ({row['stock_count']} stocks)": row["sector"]
            for row in universe_summary.get("sectors", [])
        }
    )
    selected_constituent_label = st.selectbox("Sector", list(sector_constituent_options), key="universe_sector")
    selected_constituent_sector = sector_constituent_options[selected_constituent_label]
    constituent_result = (
        list_sector_constituents(universe=universe_source, sector=selected_constituent_sector)
        if selected_constituent_sector
        else universe_summary
    )

    sectors_for_display = constituent_result.get("sectors", [])
    if sectors_for_display:
        overview_rows = [
            {
                "sector": row.get("sector"),
                "stock_count": row.get("stock_count"),
                "industry_count": row.get("industry_count"),
            }
            for row in sectors_for_display
        ]
        st.dataframe(pd.DataFrame(overview_rows), width="stretch")

        if selected_constituent_sector:
            sector_row = sectors_for_display[0]
            industry_rows = [
                {
                    "basic_industry": row.get("basic_industry"),
                    "stock_count": row.get("stock_count"),
                    "stocks": ", ".join(stock.get("symbol") or stock.get("ticker") or "" for stock in row.get("stocks", [])),
                }
                for row in sector_row.get("industries", [])
            ]
            st.markdown("**Industry Breakdown**")
            st.dataframe(pd.DataFrame(industry_rows), width="stretch")
            st.markdown("**Stocks In Selected Sector**")
            st.dataframe(pd.DataFrame(sector_row.get("stocks", [])), width="stretch")
    else:
        st.info("No stock universe rows were available.")


with sector_analytics_tab:
    st.subheader("Sector Analytics")
    st.caption(
        "ChartsMaze-style sector breadth: sector/industry membership comes from data/sectors, while RS, returns, and moving averages are recalculated from fresh or SQLite-cached price data."
    )
    sector_universe_label = st.selectbox(
        "Sector universe",
        ["Sector CSV Taxonomy (Full NSE)", "Broad NSE Total Market", "All India NSE+BSE Master", "Configured Nifty baskets"],
        index=0,
        help="Sector CSV Taxonomy reads folder names as sectors, CSV filenames as industries, and Stock Name rows as constituents. Price indicators are still recalculated live.",
        key="sector_analytics_universe",
    )
    sector_analytics_universe = {
        "Sector CSV Taxonomy (Full NSE)": "full_nse",
        "Broad NSE Total Market": "broad",
        "All India NSE+BSE Master": "all_india",
        "Configured Nifty baskets": "local",
    }[sector_universe_label]
    sector_mode_label = st.radio(
        "Sector analytics mode",
        ["Moving Average", "Relative Strength", "Near 52w High"],
        horizontal=True,
        key="sector_analytics_mode",
    )
    sector_mode = {
        "Moving Average": "moving_average",
        "Relative Strength": "relative_strength",
        "Near 52w High": "near_52w_high",
    }[sector_mode_label]

    c1, _ = st.columns([1, 3])
    if sector_mode == "moving_average":
        sector_analytics_ma = c1.selectbox("MA Type", ["200 MA", "50 MA", "20 MA", "21 EMA"], index=0, key="sector_analytics_ma_type")
        sector_analytics_rs = 80
        sector_analytics_near_high = 5.0
    elif sector_mode == "relative_strength":
        sector_analytics_ma = "200 MA"
        sector_analytics_rs = c1.slider("RS Rating Cutoff", min_value=40, max_value=95, value=80, key="sector_analytics_rs")
        sector_analytics_near_high = 5.0
    else:
        sector_analytics_ma = "200 MA"
        sector_analytics_rs = 80
        sector_analytics_near_high = c1.slider("% from 52w High <", min_value=1.0, max_value=20.0, value=5.0, step=0.5, key="sector_analytics_near_high")

    with st.expander("Stock Research Agent Settings", expanded=False):
        st.caption(
            "The UI calls the internal research agent directly. MCP exposes the same shared agent for external tools, but the dashboard does not require an MCP server."
        )
        a1, a2, a3, a4 = st.columns(4)
        stock_agent_llm_mode = a1.selectbox(
            "LLM synthesis",
            ["Off", "Auto from .env", "OpenAI-compatible", "Ollama/local"],
            index=1,
            key="sector_stock_agent_llm_mode",
        )
        stock_agent_llm_model = a2.text_input("LLM model override", value="", key="sector_stock_agent_llm_model")
        stock_agent_llm_base_url = a3.text_input("LLM base URL override", value="", key="sector_stock_agent_llm_base_url")
        stock_agent_parse_pdfs = a4.checkbox("Parse result PDFs", value=True, key="sector_stock_agent_parse_pdfs")
        st.caption(
            "Cloud mode uses STOCK_ADVISOR_LLM_PROVIDER=openai_compatible plus STOCK_ADVISOR_LLM_API_KEY or OPENAI_API_KEY. "
            "Local mode uses STOCK_ADVISOR_LLM_PROVIDER=ollama and a local Ollama-compatible server."
        )

    sector_analytics_options = {"Auto: top sector from this run": None}
    sector_universe_tickers: list[str] = []
    if sector_analytics_universe in {"broad", "full_nse", "all_india"}:
        broad_universe = list_stock_universe(universe=sector_analytics_universe, limit=None)
        sector_universe_tickers = [row["ticker"] for row in broad_universe.get("stocks", []) if row.get("ticker")]
        cache_status = get_price_cache_status(
            tickers=sector_universe_tickers,
            interval=SECTOR_ANALYTICS_INTERVAL,
        )
        if cache_status.get("enabled"):
            cached = cache_status.get("cached_ticker_count", 0)
            fresh = cache_status.get("fresh_ticker_count", 0)
            stale = cache_status.get("stale_ticker_count", 0)
            missing = cache_status.get("missing_ticker_count", 0)
            requested = cache_status.get("requested_ticker_count") or broad_universe.get("count", 0)
            coverage = cache_status.get("fresh_coverage_pct")
            coverage_text = f"{coverage}%" if coverage is not None else "n/a"
            latest_date = cache_status.get("latest_price_date")
            latest_count = cache_status.get("latest_price_date_count", 0)
            dominant_date = cache_status.get("dominant_latest_price_date")
            dominant_count = cache_status.get("dominant_latest_price_date_count", 0)
            st.caption(
                f"SQLite price cache: {fresh}/{requested} fresh ({coverage_text}), {stale} stale, {missing} missing, {cached} cached total."
            )
            if latest_date:
                st.caption(
                    f"Latest complete candles: {latest_count}/{requested} at {latest_date}; "
                    f"most symbols {dominant_count}/{requested} at {dominant_date}."
                )
                if dominant_date and latest_date != dominant_date:
                    st.warning(
                        "Cached data is split across dates. Sector analytics will use each stock's latest complete candle; "
                        "refresh the price cache from the sidebar once Yahoo finishes publishing complete NSE closes."
                    )
        sector_analytics_options.update(
            {
                f"{row['sector']} ({row['stock_count']} stocks)": str(row["sector"])
                for row in broad_universe.get("sectors", [])
            }
        )
    else:
        sector_analytics_defs = list_sector_definitions()
        sector_analytics_options.update({f"{row['name']} ({sector_id})": sector_id for sector_id, row in sector_analytics_defs.items()})
    c1, c2 = st.columns([2.2, 1])
    sector_analytics_selected_label = c1.selectbox("Sector drill-down", list(sector_analytics_options), key="sector_analytics_selected")
    run_sector_analytics = c2.button("Run sector analytics")

    if run_sector_analytics:
        _warn_if_cache_not_ready(
            "sector analytics",
            universe=sector_analytics_universe,
            period=SECTOR_ANALYTICS_PERIOD,
            interval=SECTOR_ANALYTICS_INTERVAL,
            include_bse=sector_analytics_universe == "all_india",
        )
        with st.spinner("Running sector analytics..."):
            sector_analytics_result = get_sector_analytics(
                mode=sector_mode,
                period=SECTOR_ANALYTICS_PERIOD,
                interval=SECTOR_ANALYTICS_INTERVAL,
                ma_type=sector_analytics_ma,
                rs_cutoff=sector_analytics_rs,
                near_high_pct=sector_analytics_near_high,
                selected_sector=sector_analytics_options[sector_analytics_selected_label],
                max_stocks=SECTOR_ANALYTICS_STOCK_LIMIT,
                universe=sector_analytics_universe,
                refresh_universe=sector_analytics_universe == "broad",
            )
            st.session_state["sector_analytics_result"] = sector_analytics_result
            if sector_analytics_result.get("selected_sector_id"):
                st.session_state["sector_analytics_drilldown_sector_id"] = sector_analytics_result["selected_sector_id"]

    sector_analytics_result = st.session_state.get("sector_analytics_result")
    if sector_analytics_result and sector_analytics_result.get("calculation_version") != SECTOR_ANALYTICS_VERSION:
        st.session_state.pop("sector_analytics_result", None)
        sector_analytics_result = None
    if sector_analytics_result:
        sectors = sector_analytics_result.get("sectors", [])
        top_sector_row = sector_analytics_result.get("top_sector") or {}
        metric_label = sector_analytics_result.get("metric_label", "Sector breadth")
        clicked_sector_id = _sector_id_from_plotly_state(st.session_state.get("sector_analytics_sector_chart"), sectors)
        if clicked_sector_id:
            st.session_state["sector_analytics_drilldown_sector_id"] = clicked_sector_id
        selected_sector_id = st.session_state.get("sector_analytics_drilldown_sector_id") or sector_analytics_result.get("selected_sector_id")
        selected_sector_row = _sector_row_by_id(sectors, selected_sector_id)
        selected_sector_id = selected_sector_row.get("sector_id")
        if selected_sector_id:
            st.session_state["sector_analytics_drilldown_sector_id"] = selected_sector_id
        industries_by_sector = sector_analytics_result.get("industries_by_sector", {})
        stocks_by_sector = sector_analytics_result.get("stocks_by_sector", {})
        constituent_stocks_by_sector = sector_analytics_result.get("constituent_stocks_by_sector", {})
        industries = industries_by_sector.get(selected_sector_id, sector_analytics_result.get("industries", []))
        stocks = stocks_by_sector.get(selected_sector_id, sector_analytics_result.get("stocks", []))
        constituent_stocks = constituent_stocks_by_sector.get(selected_sector_id, sector_analytics_result.get("constituent_stocks", []))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mode", sector_analytics_result.get("mode_label"))
        c2.metric("Top sector", top_sector_row.get("name") or "n/a", delta=f"{top_sector_row.get('metric_pct', 0)}%")
        c3.metric("Drill-down sector", selected_sector_row.get("name") or "n/a", delta=f"{selected_sector_row.get('metric_pct', 0)}%")
        selected_eligible_count = selected_sector_row.get("eligible_count", selected_sector_row.get("stock_count", 0))
        c4.metric("Stocks passing", selected_sector_row.get("passing_count", 0), delta=f"of {selected_eligible_count} eligible")
        freshness_parts = []
        if sector_analytics_result.get("universe_refreshed_at"):
            freshness_parts.append(f"NSE universe refreshed: {sector_analytics_result['universe_refreshed_at']}")
        if sector_analytics_result.get("price_history_end_date"):
            freshness_parts.append(f"Price history through: {sector_analytics_result['price_history_end_date']}")
        if sector_analytics_result.get("min_sector_stocks_for_ranking"):
            freshness_parts.append(f"Top-sector ranking ignores sectors below {sector_analytics_result['min_sector_stocks_for_ranking']} eligible stocks")
        if freshness_parts:
            st.caption(" | ".join(freshness_parts))

        with st.expander("Indicator, Filter, and Column Guide", expanded=False):
            st.write(sector_analytics_result.get("methodology"))
            st.dataframe(pd.DataFrame(sector_analytics_result.get("column_explanations", [])), width="stretch")

        industry_detail_state = st.session_state.get("sector_analytics_industry_detail") or {}
        selected_industry_name = str(industry_detail_state.get("industry") or "")
        if industry_detail_state.get("sector_id") != selected_sector_id:
            selected_industry_name = ""
        if selected_industry_name:
            selected_industry_row = next(
                (
                    row
                    for row in industries
                    if str(row.get("industry") or "").strip().casefold() == selected_industry_name.strip().casefold()
                ),
                None,
            )
            selected_industry_stocks = _industry_stock_rows(constituent_stocks, selected_industry_name)
            if selected_industry_row:
                _render_sector_industry_overview(
                    selected_industry_row,
                    selected_industry_stocks,
                    selected_sector_row=selected_sector_row,
                    metric_label=metric_label,
                )

        if sectors:
            sector_analytics_df = pd.DataFrame(sectors)
            selected_sector_name = selected_sector_row.get("name")
            bar_colors = [
                "#34d399" if row.get("sector_id") == selected_sector_id else "#10b981"
                for row in sectors
            ]
            sector_chart = go.Figure()
            sector_chart.add_trace(
                go.Bar(
                    x=sector_analytics_df["metric_pct"],
                    y=sector_analytics_df["name"],
                    orientation="h",
                    customdata=sector_analytics_df["sector_id"],
                    marker_color=bar_colors,
                    marker_line=dict(
                        color=["#ffffff" if name == selected_sector_name else "rgba(0,0,0,0)" for name in sector_analytics_df["name"]],
                        width=[2 if name == selected_sector_name else 0 for name in sector_analytics_df["name"]],
                    ),
                    text=sector_analytics_df["metric_pct"].map(lambda value: f"{value:.0f}%"),
                    textposition="auto",
                    hovertemplate="<b>%{y}</b><br>%{x:.2f}% passing<br>Click to drill down<extra></extra>",
                )
            )
            sector_chart.update_layout(
                title=metric_label,
                xaxis_title="% eligible stocks passing selected rule",
                yaxis=dict(autorange="reversed"),
                height=690,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            left_col, right_col = st.columns([1.85, 1], gap="large")
            with left_col:
                st.markdown('<div class="sector-click-caption">Click a sector bar to update the industries and stocks on the right.</div>', unsafe_allow_html=True)
                sector_chart_state = st.plotly_chart(
                    sector_chart,
                    width="stretch",
                    key="sector_analytics_sector_chart",
                    on_select="rerun",
                    selection_mode="points",
                    config={"displayModeBar": False},
                )
                chart_selected_sector_id = _sector_id_from_plotly_state(sector_chart_state, sectors)
                if chart_selected_sector_id:
                    st.session_state["sector_analytics_drilldown_sector_id"] = chart_selected_sector_id
                    selected_sector_id = chart_selected_sector_id
                    selected_sector_row = _sector_row_by_id(sectors, selected_sector_id)
                    industries = industries_by_sector.get(selected_sector_id, [])
                    stocks = stocks_by_sector.get(selected_sector_id, [])
                    constituent_stocks = constituent_stocks_by_sector.get(selected_sector_id, [])
            with right_col:
                clicked_industry = _render_sector_industry_cards(
                    industries,
                    key_prefix=f"sector_{selected_sector_id or 'all'}",
                )
                if clicked_industry:
                    st.session_state["sector_analytics_industry_detail"] = {
                        "sector_id": selected_sector_id,
                        "industry": clicked_industry.get("industry"),
                    }
                    st.rerun()
                _render_sector_stock_cards(stocks)

            sector_display_columns = [
                "sector_id",
                "name",
                "stage",
                "metric_pct",
                "ranking_eligible",
                "sample_confidence",
                "passing_count",
                "eligible_count",
                "stock_count",
                "coverage_pct",
                "avg_rs_rating",
                "avg_return_20d_pct",
                "above_21_ema_pct",
                "above_50_pct",
                "above_200_pct",
                "near_52w_high_pct",
            ]
            with st.expander("Sector ranking table", expanded=False):
                st.dataframe(sector_analytics_df[[column for column in sector_display_columns if column in sector_analytics_df.columns]], width="stretch")

        if stocks:
            with st.expander("Stock Research Agent", expanded=bool(st.session_state.get("sector_stock_agent_result"))):
                st.caption(
                    "This runs the LangGraph stock-research workflow: market refresh, technical/fundamental research, news, NSE/BSE filings, optional PDF parsing, and LLM synthesis."
                )
                clicked_stock_ticker = _render_sector_stock_action_buttons(
                    stocks,
                    key_prefix=f"sector_{selected_sector_id or 'all'}",
                )
                c1, c2 = st.columns([2.5, 1])
                manual_stock_ticker = c1.text_input("Or enter any NSE/BSE ticker", placeholder="RELIANCE.NS", key="sector_stock_agent_manual_ticker")
                run_stock_agent = c2.button("Run stock research agent", key="sector_stock_agent_run")
                target_ticker = clicked_stock_ticker or manual_stock_ticker.strip()
                if (clicked_stock_ticker or run_stock_agent) and not target_ticker:
                    st.warning("Choose a stock button or enter a ticker.")
                if (clicked_stock_ticker or run_stock_agent) and target_ticker:
                    llm_provider = {
                        "Off": "disabled",
                        "Auto from .env": None,
                        "OpenAI-compatible": "openai_compatible",
                        "Ollama/local": "ollama",
                    }[stock_agent_llm_mode]
                    _warn_if_cache_not_ready(
                        f"stock research for {_display_ticker(target_ticker)}",
                        universe=sector_analytics_universe,
                        period="2y",
                        interval=SECTOR_ANALYTICS_INTERVAL,
                        include_bse=sector_analytics_universe == "all_india",
                    )
                    with st.spinner(f"Running stock research agent for {_display_ticker(target_ticker)}..."):
                        st.session_state["sector_stock_agent_result"] = run_stock_research_agent(
                            target_ticker,
                            period="1y",
                            interval=SECTOR_ANALYTICS_INTERVAL,
                            intelligence_days=45,
                            intelligence_strategic_days=365,
                            include_exchange_announcements=True,
                            parse_exchange_pdfs=stock_agent_parse_pdfs,
                            include_llm=stock_agent_llm_mode != "Off",
                            llm_provider=llm_provider,
                            llm_model=stock_agent_llm_model.strip() or None,
                            llm_base_url=stock_agent_llm_base_url.strip() or None,
                        )

                stock_agent_result = st.session_state.get("sector_stock_agent_result")
                if stock_agent_result:
                    _render_stock_research_agent_result(stock_agent_result)

        if industries:
            with st.expander(f"Industry details: {selected_sector_row.get('name') or 'selected sector'}", expanded=False):
                industry_df = pd.DataFrame(industries)
                industry_fig = go.Figure()
                industry_fig.add_trace(
                    go.Bar(
                        x=industry_df["contribution_pct"],
                        y=industry_df["industry"],
                        orientation="h",
                        marker_color="#14b8a6",
                        text=industry_df["contribution_pct"].map(lambda value: f"{value:.1f}%"),
                        textposition="auto",
                    )
                )
                industry_fig.update_layout(
                    title="Industry Contribution",
                    xaxis_title="Contribution (%)",
                    yaxis=dict(autorange="reversed"),
                    height=360,
                    margin=dict(l=10, r=10, t=45, b=10),
                )
                st.plotly_chart(industry_fig, width="stretch")
                industry_display = industry_df.copy()
                if "passing_tickers" in industry_display.columns:
                    industry_display["passing_tickers"] = industry_display["passing_tickers"].apply(lambda rows: ", ".join(rows or []))
                if "stock_tickers" in industry_display.columns:
                    industry_display["stock_tickers"] = industry_display["stock_tickers"].apply(lambda rows: ", ".join(rows or []))
                st.dataframe(
                    industry_display[
                        [
                            "industry",
                            "sector",
                            "contribution_pct",
                            "formula",
                            "passing_count",
                            "eligible_count",
                            "stock_count",
                            "pass_pct_within_industry",
                            "avg_rs_rating",
                            "top_stock",
                            "passing_tickers",
                            "stock_tickers",
                        ]
                    ],
                    width="stretch",
                )

        if stocks:
            with st.expander(f"Strong stock details: {selected_sector_row.get('name') or 'selected sector'}", expanded=False):
                stock_df = pd.DataFrame(stocks)
                stock_fig = go.Figure()
                stock_fig.add_trace(
                    go.Bar(
                        x=stock_df["rs_rating"],
                        y=stock_df["ticker"],
                        orientation="h",
                        marker_color=["#10b981" if passed else "#94a3b8" for passed in stock_df["passes_filter"]],
                        text=stock_df["rs_rating"].map(lambda value: f"{value:.0f}"),
                        textposition="auto",
                    )
                )
                stock_fig.update_layout(
                    title="Strong Stocks by RS",
                    xaxis_title="RS rating",
                    yaxis=dict(autorange="reversed"),
                    height=max(360, min(900, 28 * len(stock_df))),
                    margin=dict(l=10, r=10, t=45, b=10),
                )
                st.plotly_chart(stock_fig, width="stretch")
                st.dataframe(stock_df, width="stretch")

        if constituent_stocks:
            with st.expander("All selected-sector stock rows", expanded=False):
                st.caption("This is the full selected-sector stock-breadth table used for the sector and industry calculations.")
                st.dataframe(pd.DataFrame(constituent_stocks), width="stretch")

        warnings = sector_analytics_result.get("warnings", [])
        if warnings:
            with st.expander("Provider warnings", expanded=False):
                st.write(warnings)
    else:
        st.info("Run sector analytics to rank sectors by moving-average breadth, relative-strength breadth, or proximity to 52-week highs.")

with rrg_tab:
    st.subheader("Relative Rotation Graph")
    st.caption("ChartsMaze-style RRG using this tool's own free price providers. It fetches fresh sector index history when you run it.")
    rrg_sector_defs = list_rrg_index_definitions()
    rrg_benchmark_options = {"Nifty 50": "^NSEI"}
    rrg_benchmark_options.update({f"{row['name']}": sector_id for sector_id, row in rrg_sector_defs.items()})
    c1, c2, c3, c4 = st.columns(4)
    rrg_benchmark_label = c1.selectbox("Benchmark", list(rrg_benchmark_options), index=0, key="rrg_benchmark")
    rrg_tail_length = c2.slider("Tail Length", min_value=3, max_value=30, value=5, key="rrg_tail_length")
    rrg_interval_label = c3.selectbox("Interval", ["Daily", "Weekly"], index=0, key="rrg_interval")
    rrg_data_mode = c4.selectbox(
        "Data Mode",
        ["Sector Index"],
        index=0,
        help="Sector Index plots NSE sector index or free ETF fallback data. Stock-level RRG can be added later as a separate mode.",
        key="rrg_data_mode",
    )
    c1, c2, c3 = st.columns([1, 1.4, 1])
    rrg_period = c1.selectbox(
        "History window",
        ["6mo", "1y", "2y"],
        index=1,
        key="rrg_period",
        help="How much historical data to fetch before calculating the latest RRG tail. Tail Length controls how many final daily/weekly points are drawn.",
    )
    rrg_zones = c2.multiselect(
        "Zone",
        ["Leading", "Improving", "Lagging", "Weakening"],
        default=["Leading", "Improving", "Lagging", "Weakening"],
        key="rrg_zones",
    )
    run_rrg = c3.button("Run RRG")

    rrg_sector_options = {f"{row['name']} ({sector_id})": sector_id for sector_id, row in rrg_sector_defs.items()}
    selected_rrg_sector_labels = st.multiselect(
        "Indexes",
        list(rrg_sector_options),
        default=list(rrg_sector_options),
        help="Search and choose which sector indices to plot. These are fetched from our free provider pipeline.",
        key="rrg_selected_indexes",
    )

    if run_rrg:
        _warn_if_cache_not_ready(
            "RRG",
            period=rrg_period,
            interval={"Daily": "1d", "Weekly": "1wk"}[rrg_interval_label],
        )
        with st.spinner("Calculating fresh RRG trails..."):
            st.session_state["standalone_rrg_result"] = get_relative_rotation_graph(
                period=rrg_period,
                interval={"Daily": "1d", "Weekly": "1wk"}[rrg_interval_label],
                benchmark=rrg_benchmark_options[rrg_benchmark_label],
                trail_length=rrg_tail_length,
                selected_sectors=[rrg_sector_options[label] for label in selected_rrg_sector_labels],
                zone=rrg_zones,
            )

    rrg_result = st.session_state.get("standalone_rrg_result")
    if rrg_result and rrg_result.get("calculation_version") != "rrg_nse_index_calibrated_v5":
        st.session_state.pop("standalone_rrg_result", None)
        rrg_result = None
    if rrg_result:
        points = rrg_result.get("points", [])
        trails = rrg_result.get("trails", [])
        if not points or not trails:
            st.warning("No RRG points were available for the selected filters.")
        else:
            top_leader = rrg_result.get("top_leading_sector") or {}
            top_upcoming = rrg_result.get("top_upcoming_sector") or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Benchmark", rrg_result.get("benchmark_name") or rrg_result.get("benchmark"))
            c2.metric("Top leading", top_leader.get("name") or "n/a", delta=top_leader.get("rotation_score"))
            c3.metric("Top upcoming", top_upcoming.get("name") or "n/a", delta=top_upcoming.get("rotation_score"))
            c4.metric("Tail ending", str(rrg_result.get("end_date") or "n/a")[:10], delta=f"{rrg_result.get('trail_length')} points")

            with st.expander("Indicator, Filter, and Column Guide", expanded=False):
                st.write(rrg_result.get("methodology"))
                st.dataframe(pd.DataFrame(rrg_result.get("column_explanations", [])), width="stretch")
                st.write(
                    f"Data mode: {rrg_data_mode}. Period controls the history fetched; interval controls whether each tail point is daily or weekly."
                )

            c1, c2 = st.columns([2.4, 1])
            with c1:
                st.plotly_chart(_rrg_figure(trails), width="stretch")
            with c2:
                quadrant_counts = rrg_result.get("quadrant_counts", {})
                quadrant_df = pd.DataFrame(
                    [{"quadrant": key, "count": value} for key, value in quadrant_counts.items() if value]
                )
                if not quadrant_df.empty:
                    quadrant_fig = go.Figure()
                    quadrant_fig.add_trace(
                        go.Pie(
                            labels=quadrant_df["quadrant"],
                            values=quadrant_df["count"],
                            marker=dict(colors=[_rrg_quadrant_color(label) for label in quadrant_df["quadrant"]]),
                            hole=0.45,
                        )
                    )
                    quadrant_fig.update_layout(title="Zone Mix", height=310, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(quadrant_fig, width="stretch")
                point_df = _rrg_points_frame(points)
                if not point_df.empty:
                    score_fig = go.Figure()
                    top_scores = point_df.head(10)
                    score_fig.add_trace(
                        go.Bar(
                            x=top_scores["rotation_score"],
                            y=top_scores["name"],
                            orientation="h",
                            marker_color=[_rrg_quadrant_color(value) for value in top_scores["quadrant"]],
                        )
                    )
                    score_fig.update_layout(
                        title="Rotation Score",
                        yaxis=dict(autorange="reversed"),
                        height=310,
                        margin=dict(l=10, r=10, t=45, b=10),
                    )
                    st.plotly_chart(score_fig, width="stretch")

            point_df = _rrg_points_frame(points)
            st.dataframe(point_df, width="stretch")

            rrg_drill_options = {f"{row['name']} ({row['quadrant']})": row["sector_id"] for row in points}
            c1, c2 = st.columns([2, 1])
            rrg_drill_label = c1.selectbox("Dive into Sector Index", list(rrg_drill_options), key="rrg_drill_sector")
            run_rrg_drill = c2.button("Rank stocks in RRG sector")
            if run_rrg_drill:
                _warn_if_cache_not_ready(
                    "RRG stock drill-down",
                    period=rrg_period,
                    interval={"Daily": "1d", "Weekly": "1wk"}[rrg_interval_label],
                )
                with st.spinner("Ranking stocks inside selected RRG sector..."):
                    st.session_state["rrg_stock_result"] = rank_sector_stocks(
                        rrg_drill_options[rrg_drill_label],
                        period=rrg_period,
                        interval={"Daily": "1d", "Weekly": "1wk"}[rrg_interval_label],
                        max_stocks=10,
                        include_fundamentals=True,
                    )

            rrg_stock_result = st.session_state.get("rrg_stock_result")
            if rrg_stock_result:
                st.markdown("**RRG Sector Stock Candidates**")
                stocks = rrg_stock_result.get("stocks", [])
                if stocks:
                    stock_df = pd.DataFrame(
                        [
                            {
                                "ticker": row.get("ticker"),
                                "stage": row.get("stage"),
                                "stock_score": row.get("stock_score"),
                                "20D RS vs sector": _movement_display((row.get("relative_strength", {}) or {}).get("vs_sector_20d")),
                                "60D RS vs sector": _movement_display((row.get("relative_strength", {}) or {}).get("vs_sector_60d")),
                                "Trend": _score_display(row.get("trend_score")),
                                "Pattern": _score_display(row.get("pattern_score")),
                                "Volume": _score_display(row.get("volume_score")),
                                "Risk": _score_display(row.get("risk_quality_score")),
                                "dominant_chart_pattern": row.get("dominant_chart_pattern"),
                            }
                            for row in stocks
                        ]
                    )
                    stock_fig = go.Figure()
                    stock_fig.add_trace(
                        go.Bar(x=stock_df["stock_score"], y=stock_df["ticker"], orientation="h", marker_color="#0f766e")
                    )
                    stock_fig.update_layout(
                        title=f"Stock Targets in {rrg_stock_result.get('sector_name')}",
                        yaxis=dict(autorange="reversed"),
                        height=360,
                        margin=dict(l=10, r=10, t=45, b=10),
                    )
                    st.plotly_chart(stock_fig, width="stretch")
                    st.dataframe(stock_df, width="stretch")
                else:
                    st.info("No stock candidates were available for this sector.")

            warnings = rrg_result.get("warnings", [])
            if warnings:
                with st.expander("RRG provider warnings", expanded=False):
                    st.write(warnings)
    else:
        st.info("Run RRG to plot sector rotation tails against the selected benchmark.")

with industry_tab:
    st.subheader("Industry Analytics")
    st.caption("Ranks NSE basic industries across 1W, 1M, and 3M movement with RRG-style stage labels.")
    industry_universe_label = st.selectbox(
        "Industry universe",
        ["Broad NSE Total Market", "Full NSE Equity Master", "Configured industry groups"],
        index=1,
        help="Full NSE Equity Master is closest to ChartsMaze, but requires a full universe refresh. Broad mode uses NSE basic-industry metadata for 750 total-market stocks.",
        key="industry_universe",
    )
    industry_universe = {
        "Broad NSE Total Market": "broad",
        "Full NSE Equity Master": "full_nse",
        "Configured industry groups": "local",
    }[industry_universe_label]
    c1, c2, c3, c4 = st.columns(4)
    industry_period = c1.selectbox("Industry period", ["6mo", "1y", "2y"], index=1, key="industry_period")
    industry_interval = c2.selectbox("Industry interval", ["1d", "1wk"], index=0, key="industry_interval")
    industry_min_stocks = c3.slider("Minimum stocks", min_value=2, max_value=12, value=3)
    industry_weighting = c4.selectbox("Weighting", ["equal", "market_cap"], index=0)
    c1, c2, c3 = st.columns([1, 1, 1.2])
    industry_include_fundamentals = c1.checkbox("Use market cap data", value=False)
    run_industry = c2.button("Run industry analytics")
    st.caption("This mirrors the top-down industry ranking workflow: performance windows, ranks, stock count, breadth, and RRG quadrant.")

    industry_definitions = list_industry_definitions(universe=industry_universe)
    industry_options = {f"{row['name']} ({industry_id})": industry_id for industry_id, row in industry_definitions.items()}
    selected_industry_label = st.selectbox("Industry stock candidates", list(industry_options), index=0) if industry_options else None
    selected_industry = industry_options[selected_industry_label] if selected_industry_label else None
    run_industry_stocks = c3.button("Rank selected industry stocks")

    if run_industry:
        _warn_if_cache_not_ready(
            "industry analytics",
            universe=industry_universe,
            period=industry_period,
            interval=industry_interval,
        )
        with st.spinner("Running industry analytics..."):
            st.session_state["industry_analytics_result"] = get_industry_analytics(
                period=industry_period,
                interval=industry_interval,
                min_stocks=industry_min_stocks,
                weighting=industry_weighting,
                include_fundamentals=industry_include_fundamentals,
                universe=industry_universe,
                refresh_universe=industry_universe == "broad",
            )

    if run_industry_stocks and selected_industry:
        _warn_if_cache_not_ready(
            "industry stock ranking",
            universe=industry_universe,
            period=industry_period,
            interval=industry_interval,
        )
        with st.spinner("Ranking stocks inside selected industry..."):
            st.session_state["industry_stock_result"] = rank_industry_stocks(
                selected_industry,
                period=industry_period,
                interval=industry_interval,
                max_stocks=10,
                include_fundamentals=True,
                universe=industry_universe,
                refresh_universe=industry_universe == "broad",
            )

    industry_result = st.session_state.get("industry_analytics_result")
    if industry_result:
        industries = industry_result.get("industries", [])
        if not industries:
            st.warning("No industry data was returned by the market data provider.")
        else:
            top_industry = industry_result.get("top_industry") or industries[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top industry", top_industry.get("name"))
            c2.metric("Composite score", top_industry.get("composite_score"))
            c3.metric("Stage", top_industry.get("stage"))
            c4.metric("Stocks", top_industry.get("stock_count"))

            industry_search = st.text_input("Search industries", placeholder="Pharma, infra, banks...", key="industry_search")
            filtered_industries = [
                row
                for row in industries
                if not industry_search
                or industry_search.lower() in str(row.get("name", "")).lower()
                or industry_search.lower() in str(row.get("sector", "")).lower()
            ]
            table_columns = st.multiselect(
                "Industry columns",
                [
                    "name",
                    "sector",
                    "Score",
                    "stage",
                    "1W",
                    "1M",
                    "3M",
                    "20D RS",
                    "60D RS",
                    "rank_1w",
                    "rank_1m",
                    "rank_3m",
                    "stock_count",
                    "Breadth",
                    "above_50_pct",
                    "near_52w_high_pct",
                ],
                default=["name", "sector", "Score", "stage", "1W", "1M", "3M", "rank_1w", "rank_1m", "rank_3m", "stock_count"],
            )
            industry_df = _analytics_table(filtered_industries, table_columns)
            st.dataframe(industry_df, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                top_rows = pd.DataFrame(filtered_industries[:12])
                if not top_rows.empty:
                    returns_fig = go.Figure()
                    returns_fig.add_trace(go.Bar(x=top_rows["name"], y=top_rows["return_5d"] * 100, name="1W"))
                    returns_fig.add_trace(go.Bar(x=top_rows["name"], y=top_rows["return_20d"] * 100, name="1M"))
                    returns_fig.add_trace(go.Bar(x=top_rows["name"], y=top_rows["return_60d"] * 100, name="3M"))
                    returns_fig.update_layout(
                        title="Industry Performance Windows",
                        yaxis_title="Return (%)",
                        barmode="group",
                        height=430,
                        margin=dict(l=10, r=10, t=45, b=110),
                    )
                    st.plotly_chart(returns_fig, width="stretch")
            with c2:
                quadrant_counts = pd.DataFrame(filtered_industries)["stage"].value_counts().reset_index()
                quadrant_counts.columns = ["stage", "count"]
                stage_fig = go.Figure()
                stage_fig.add_trace(go.Pie(labels=quadrant_counts["stage"], values=quadrant_counts["count"], hole=0.45))
                stage_fig.update_layout(title="Industry Stage Mix", height=430, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(stage_fig, width="stretch")

            with st.expander("Root cause for top industry", expanded=False):
                for cause in top_industry.get("root_causes", []):
                    st.write(f"- {cause}")

            warnings = industry_result.get("warnings", [])
            if warnings:
                with st.expander("Industry provider warnings", expanded=False):
                    st.write(warnings)
    else:
        st.info("Run industry analytics to rank industries by short, medium, and longer movement.")

    industry_stock_result = st.session_state.get("industry_stock_result")
    if industry_stock_result:
        st.markdown("**Selected Industry Stock Candidates**")
        industry_stocks = industry_stock_result.get("stocks", [])
        if industry_stocks:
            stock_rows = pd.DataFrame(
                [
                    {
                        "ticker": row.get("ticker"),
                        "stage": row.get("stage"),
                        "stock_score": row.get("stock_score"),
                        "20D RS": _movement_display((row.get("relative_strength", {}) or {}).get("vs_sector_20d")),
                        "60D RS": _movement_display((row.get("relative_strength", {}) or {}).get("vs_sector_60d")),
                        "Trend": _score_display(row.get("trend_score")),
                        "Pattern": _score_display(row.get("pattern_score")),
                        "Volume": _score_display(row.get("volume_score")),
                        "Risk": _score_display(row.get("risk_quality_score")),
                        "dominant_chart_pattern": row.get("dominant_chart_pattern"),
                    }
                    for row in industry_stocks
                ]
            )
            st.dataframe(stock_rows, width="stretch")
            stock_fig = go.Figure()
            stock_fig.add_trace(go.Bar(x=stock_rows["stock_score"], y=stock_rows["ticker"], orientation="h", marker_color="#0f766e"))
            stock_fig.update_layout(
                title=f"Stock Targets in {industry_stock_result.get('industry_name')}",
                yaxis=dict(autorange="reversed"),
                height=360,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(stock_fig, width="stretch")
        else:
            st.info("No stock candidates were available for the selected industry.")

with indices_tab:
    st.subheader("Market Indices")
    st.caption("Broad and sector index performance with relative strength, trend health, and RRG-style rotation.")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    indices_period = c1.selectbox("Index period", ["6mo", "1y", "2y"], index=1)
    indices_interval = c2.selectbox("Index interval", ["1d", "1wk"], index=0)
    run_indices = c3.button("Run market indices")

    if run_indices:
        _warn_if_cache_not_ready("market indices", period=indices_period, interval=indices_interval)
        with st.spinner("Fetching latest available index data..."):
            st.session_state["market_indices_result"] = get_market_indices(
                period=indices_period,
                interval=indices_interval,
            )
            st.session_state["rrg_result"] = get_relative_rotation_graph(
                period=indices_period,
                interval=indices_interval,
            )

    indices_result = st.session_state.get("market_indices_result")
    rrg_result = st.session_state.get("rrg_result")
    if indices_result:
        indices = indices_result.get("indices", [])
        if not indices:
            st.warning("No market index data was returned.")
        else:
            index_table = _analytics_table(
                indices,
                ["name", "ticker", "category", "1D", "1W", "1M", "3M", "20D RS", "60D RS", "RS Score", "Trend", "rsi_14", "adx_14", "above_sma_50", "above_sma_200"],
            )
            st.dataframe(index_table, width="stretch")

            index_df = pd.DataFrame(indices)
            c1, c2 = st.columns(2)
            with c1:
                perf_fig = go.Figure()
                perf_fig.add_trace(go.Bar(x=index_df["name"], y=index_df["return_5d"] * 100, name="1W"))
                perf_fig.add_trace(go.Bar(x=index_df["name"], y=index_df["return_20d"] * 100, name="1M"))
                perf_fig.add_trace(go.Bar(x=index_df["name"], y=index_df["return_60d"] * 100, name="3M"))
                perf_fig.update_layout(title="Index Performance", yaxis_title="Return (%)", barmode="group", height=430, margin=dict(l=10, r=10, t=45, b=110))
                st.plotly_chart(perf_fig, width="stretch")
            with c2:
                health_fig = go.Figure()
                health_fig.add_trace(
                    go.Scatter(
                        x=index_df["relative_strength_score"],
                        y=index_df["trend_score"],
                        mode="markers+text",
                        text=index_df["name"],
                        textposition="top center",
                        marker=dict(size=14, color=index_df["return_20d"] * 100, colorscale="Viridis", showscale=True),
                    )
                )
                health_fig.update_layout(title="RS Score vs Trend", xaxis_title="RS score", yaxis_title="Trend score", height=430, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(health_fig, width="stretch")

            if rrg_result and rrg_result.get("points"):
                rrg_df = pd.DataFrame(rrg_result.get("points", []))
                rrg_fig = go.Figure()
                rrg_fig.add_trace(
                    go.Scatter(
                        x=rrg_df["x_rs_60d_pct"],
                        y=rrg_df["y_rs_momentum_pct"],
                        mode="markers+text",
                        text=rrg_df["sector_id"],
                        textposition="top center",
                        marker=dict(size=rrg_df["rotation_score"].clip(lower=20) / 2, color=rrg_df["quadrant"].astype("category").cat.codes, colorscale="Turbo"),
                    )
                )
                rrg_fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
                rrg_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                rrg_fig.update_layout(
                    title="Relative Rotation Graph Style Map",
                    xaxis_title="RS-Ratio distance from 100",
                    yaxis_title="RS-Momentum distance from 100",
                    height=520,
                    margin=dict(l=10, r=10, t=45, b=10),
                )
                st.plotly_chart(rrg_fig, width="stretch")
                st.dataframe(rrg_df, width="stretch")

            warnings = [*indices_result.get("warnings", []), *(rrg_result or {}).get("warnings", [])]
            if warnings:
                with st.expander("Index provider warnings", expanded=False):
                    st.write(warnings)
    else:
        st.info("Run market indices to compare sector indices and RRG-style movement.")

with breadth_tab:
    st.subheader("Market Breadth")
    st.caption("Participation view: how many configured stocks are above moving averages, positive over recent windows, and near highs.")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    breadth_period = c1.selectbox("Breadth period", ["6mo", "1y", "2y"], index=1)
    breadth_interval = c2.selectbox("Breadth interval", ["1d", "1wk"], index=0)
    breadth_limit = c3.slider("Max stocks", min_value=20, max_value=120, value=80)
    run_breadth = st.button("Run market breadth")

    if run_breadth:
        _warn_if_cache_not_ready("market breadth", period=breadth_period, interval=breadth_interval)
        with st.spinner("Checking market participation..."):
            st.session_state["market_breadth_result"] = get_market_breadth(
                period=breadth_period,
                interval=breadth_interval,
                max_stocks=breadth_limit,
            )

    breadth_result = st.session_state.get("market_breadth_result")
    if breadth_result:
        summary = breadth_result.get("summary", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Stocks checked", summary.get("stock_count"))
        c2.metric("Above 20-DMA", f"{summary.get('above_20_pct')}%")
        c3.metric("Above 50-DMA", f"{summary.get('above_50_pct')}%")
        c4.metric("Above 200-DMA", f"{summary.get('above_200_pct')}%")
        c5.metric("Positive 20D", f"{summary.get('positive_20d_pct')}%")

        sector_breadth = breadth_result.get("sectors", [])
        if sector_breadth:
            breadth_df = pd.DataFrame(sector_breadth)
            breadth_fig = go.Figure()
            breadth_fig.add_trace(go.Bar(x=breadth_df["sector"], y=breadth_df["above_20_pct"], name="Above 20-DMA"))
            breadth_fig.add_trace(go.Bar(x=breadth_df["sector"], y=breadth_df["above_50_pct"], name="Above 50-DMA"))
            breadth_fig.add_trace(go.Bar(x=breadth_df["sector"], y=breadth_df["above_200_pct"], name="Above 200-DMA"))
            breadth_fig.add_trace(go.Bar(x=breadth_df["sector"], y=breadth_df["positive_20d_pct"], name="Positive 20D"))
            breadth_fig.update_layout(title="Sector Breadth", yaxis_title="% stocks", barmode="group", height=460, margin=dict(l=10, r=10, t=45, b=100))
            st.plotly_chart(breadth_fig, width="stretch")
            st.dataframe(breadth_df, width="stretch")

        stocks = breadth_result.get("stocks", [])
        if stocks:
            stock_breadth = pd.DataFrame(
                [
                    {
                        "ticker": row.get("ticker"),
                        "sector": row.get("sector"),
                        "industry": row.get("industry"),
                        "1W": _movement_display(row.get("return_5d")),
                        "1M": _movement_display(row.get("return_20d")),
                        "above_sma_20": row.get("above_sma_20"),
                        "above_sma_50": row.get("above_sma_50"),
                        "above_sma_200": row.get("above_sma_200"),
                        "distance_from_52w_high": _movement_display(row.get("distance_from_52w_high")),
                    }
                    for row in stocks
                ]
            )
            st.dataframe(stock_breadth, width="stretch")
    else:
        st.info("Run market breadth to see whether market movement has broad participation.")

with crossover_tab:
    st.subheader("50 DMA / 200 DMA Crossovers")
    st.caption("Find stocks where the 50-day simple moving average recently crossed the 200-day simple moving average.")
    c1, c2, c3, c4 = st.columns(4)
    crossover_universe_label = c1.selectbox(
        "Universe",
        ["Full NSE Equity Master", "Nifty Total Market", "Local configured baskets", "All India NSE+BSE"],
        index=0,
        key="ma_cross_universe",
    )
    crossover_universe = {
        "Full NSE Equity Master": "full_nse",
        "Nifty Total Market": "broad",
        "Local configured baskets": "local",
        "All India NSE+BSE": "all_india",
    }[crossover_universe_label]
    crossover_direction_label = c2.selectbox(
        "Cross type",
        ["Bullish golden cross", "Bearish death cross", "Both"],
        index=0,
        key="ma_cross_direction",
    )
    crossover_direction = {
        "Bullish golden cross": "bullish",
        "Bearish death cross": "bearish",
        "Both": "both",
    }[crossover_direction_label]
    crossover_lookback = c3.slider("Cross happened within", min_value=1, max_value=120, value=20, step=1, key="ma_cross_lookback")
    crossover_min_market_cap = c4.number_input("Market Cap >", min_value=0.0, value=0.0, step=250.0, key="ma_cross_market_cap")

    s1, s2, s3 = st.columns(3)
    crossover_period = s1.selectbox("Price period", ["1y", "2y", "5y"], index=1, key="ma_cross_period")
    crossover_interval = s2.selectbox("Price interval", ["1d"], index=0, key="ma_cross_interval")
    crossover_max_rows = s3.slider("Rows", min_value=25, max_value=300, value=150, step=25, key="ma_cross_rows")
    run_crossover_scan = st.button("Run crossover scan", key="run_ma_crossover_scan")

    if run_crossover_scan:
        _warn_if_cache_not_ready(
            "50/200 moving-average crossover scan",
            universe=crossover_universe,
            period=crossover_period,
            interval=crossover_interval,
            include_bse=crossover_universe == "all_india",
        )
        with st.spinner("Scanning 50 DMA / 200 DMA crossovers..."):
            st.session_state["ma_crossover_result"] = get_moving_average_crossover_scan(
                period=crossover_period,
                interval=crossover_interval,
                universe=crossover_universe,
                direction=crossover_direction,
                lookback_periods=int(crossover_lookback),
                min_market_cap_cr=float(crossover_min_market_cap),
                max_rows=int(crossover_max_rows),
            )

    crossover_result = st.session_state.get("ma_crossover_result")
    if crossover_result:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Universe stocks", crossover_result.get("universe_stock_count"))
        c2.metric("Price data", crossover_result.get("price_data_stock_count"))
        c3.metric("Eligible 50/200 DMA", crossover_result.get("eligible_stock_count"))
        c4.metric("Signals", crossover_result.get("signal_stock_count"))
        c5.metric("Data through", crossover_result.get("price_history_end_date") or "n/a")

        if crossover_result.get("warnings"):
            for warning in crossover_result.get("warnings", []):
                st.warning(warning)

        st.caption(crossover_result.get("methodology"))
        crossover_rows = crossover_result.get("stocks", [])
        sector_summary = crossover_result.get("sector_summary", [])
        industry_summary = crossover_result.get("industry_summary", [])

        if crossover_rows:
            stock_df = _ma_crossover_stock_frame(crossover_rows)
            left, right = st.columns([1.4, 1])
            with left:
                sector_df = _ma_crossover_summary_frame(sector_summary, "sector")
                if not sector_df.empty:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=sector_df["Signals"],
                            y=sector_df["Sector"],
                            orientation="h",
                            marker_color="#10b981" if crossover_direction != "bearish" else "#ef4444",
                            text=sector_df["Signals"],
                            textposition="auto",
                        )
                    )
                    fig.update_layout(
                        title="Signals by Sector",
                        xaxis_title="Stocks",
                        yaxis=dict(autorange="reversed"),
                        height=max(360, 30 * len(sector_df) + 90),
                        margin=dict(l=10, r=10, t=45, b=30),
                    )
                    st.plotly_chart(fig, width="stretch")
            with right:
                st.markdown("**Top Sectors**")
                sector_table = _ma_crossover_summary_frame(sector_summary[:12], "sector")
                st.dataframe(sector_table, width="stretch", hide_index=True)

            st.markdown("**Stocks Matching 50/200 DMA Cross**")
            st.dataframe(stock_df, width="stretch", hide_index=True)
            st.download_button(
                "Download crossover stocks",
                stock_df.to_csv(index=False),
                file_name="ma_50_200_crossovers.csv",
                mime="text/csv",
                width="stretch",
            )

            with st.expander("Industry summary", expanded=False):
                industry_df = _ma_crossover_summary_frame(industry_summary, "industry")
                st.dataframe(industry_df, width="stretch", hide_index=True)
        else:
            st.info("No stocks matched the selected 50/200 DMA crossover filters. Increase the lookback window or choose Both.")
    else:
        st.info("Run the crossover scan to list stocks with recent bullish or bearish 50 DMA / 200 DMA crosses.")

with gainers_tab:
    st.subheader("Top Gainers")
    st.caption("Be the first to unpack the next sector or industry move from a single glance.")
    c1, c2, c3, c4 = st.columns(4)
    return_type = c1.selectbox("Returns Type", ["1 Day Return", "1 Week Return", "1 Month Return"], index=0)
    return_window = {
        "1 Day Return": "1d",
        "1 Week Return": "1w",
        "1 Month Return": "1m",
    }[return_type]
    market_cap_min = c2.number_input("Market Cap >", min_value=0.0, value=1000.0, step=250.0)
    min_return = c3.number_input("Stock Return(%) >", min_value=-50.0, max_value=100.0, value=5.0, step=0.5)
    min_industry_stocks = c4.number_input("No. of stock in Industry >=", min_value=1, max_value=50, value=3, step=1)

    with st.expander("Data settings", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        gainers_universe_label = s1.selectbox("Universe", ["Full NSE Equity Master", "Nifty Total Market", "Local configured baskets"], index=0)
        gainers_universe = {
            "Full NSE Equity Master": "full_nse",
            "Nifty Total Market": "broad",
            "Local configured baskets": "local",
        }[gainers_universe_label]
        gainers_period = s2.selectbox("Price period", ["3mo", "6mo", "1y", "2y"], index=2)
        gainers_interval = s3.selectbox("Price interval", ["1d", "1wk"], index=0)
        max_gainers = s4.slider("Overall table rows", min_value=10, max_value=150, value=60)

    run_gainers = st.button("Run top gainers")

    if run_gainers:
        _warn_if_cache_not_ready(
            "top gainers",
            universe=gainers_universe,
            period=gainers_period,
            interval=gainers_interval,
        )
        with st.spinner("Ranking top gainers..."):
            st.session_state["top_gainers_result"] = get_top_gainers(
                period=gainers_period,
                interval=gainers_interval,
                return_window=return_window,
                min_return_pct=min_return,
                market_cap_min=market_cap_min,
                min_industry_stocks=int(min_industry_stocks),
                max_rows=max_gainers,
                universe=gainers_universe,
            )

    gainers_result = st.session_state.get("top_gainers_result")
    if gainers_result:
        gainers = gainers_result.get("stocks", [])
        industry_summary = gainers_result.get("industry_summary", [])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe stocks", gainers_result.get("universe_stock_count"))
        c2.metric("Eligible stocks", gainers_result.get("eligible_stock_count"))
        c3.metric("Stocks passing", gainers_result.get("gainer_stock_count"))
        c4.metric("Price data through", gainers_result.get("price_history_end_date") or "n/a")
        if gainers_result.get("warnings"):
            for warning in gainers_result.get("warnings", []):
                st.warning(warning)

        if gainers:
            overall_df = _top_gainer_stock_frame(gainers, include_context=True)
            industry_df = _top_gainer_industry_frame(industry_summary)
            if not industry_summary:
                st.info("Stocks matched the return filter, but no industry met the minimum industry stock count.")
                st.markdown("**Overall Top Performers**")
                st.dataframe(overall_df, width="stretch", hide_index=True)
                st.download_button(
                    "Download",
                    overall_df.to_csv(index=False),
                    file_name="top_gainers.csv",
                    mime="text/csv",
                    width="stretch",
                )
                st.stop()
            left, right = st.columns([1.55, 1])
            with left:
                industry_fig = go.Figure()
                industry_fig.add_trace(
                    go.Bar(
                        x=[row.get("avg_selected_return_pct") for row in industry_summary],
                        y=[row.get("label") for row in industry_summary],
                        customdata=[[row.get("industry")] for row in industry_summary],
                        orientation="h",
                        marker_color="#5cc59b",
                        text=[f"{row.get('avg_selected_return_pct')}%" for row in industry_summary],
                        textposition="inside",
                    )
                )
                industry_fig.update_layout(
                    title="Industry(% of Total Stocks in the Industry)",
                    xaxis_title="Returns(%)",
                    yaxis=dict(autorange="reversed"),
                    height=max(460, 34 * len(industry_summary) + 130),
                    margin=dict(l=10, r=10, t=55, b=35),
                )
                chart_state = st.plotly_chart(
                    industry_fig,
                    width="stretch",
                    on_select="rerun",
                    selection_mode="points",
                    key="top_gainers_industry_chart",
                )
                clicked_industry = _top_gainer_industry_from_plotly_state(chart_state, industry_summary)
                if clicked_industry:
                    st.session_state["top_gainers_selected_industry"] = clicked_industry
                st.caption("Click an industry bar to update the industry performer table.")

            with right:
                st.markdown("**Overall Top Performers**")
                st.dataframe(overall_df[["Stock", "Return (%)"]], width="stretch", hide_index=True, height=420)
                csv_data = overall_df.to_csv(index=False)
                stock_list = ",".join(f"NSE:{stock}" for stock in overall_df["Stock"].dropna().astype(str).tolist())
                st.download_button(
                    "Download",
                    csv_data,
                    file_name="top_gainers.csv",
                    mime="text/csv",
                    width="stretch",
                )
                with st.expander("Export to TradingView / Add to Watchlist", expanded=False):
                    st.text_area("TradingView symbols", stock_list, height=110)

            industry_names = [row.get("industry") for row in industry_summary if row.get("industry")]
            selected_industry = st.session_state.get("top_gainers_selected_industry")
            if selected_industry not in industry_names:
                selected_industry = industry_names[0] if industry_names else None
            if selected_industry:
                selected_index = industry_names.index(selected_industry) if selected_industry in industry_names else 0
                selected_industry = st.selectbox("Industry performers", industry_names, index=selected_index)
                st.session_state["top_gainers_selected_industry"] = selected_industry
                industry_rows = gainers_result.get("performers_by_industry", {}).get(selected_industry, [])
                selected_df = _top_gainer_stock_frame(industry_rows, include_context=True)
                st.markdown(f"**Top Performers in {selected_industry}**")
                st.dataframe(selected_df, width="stretch", hide_index=True)

            with st.expander("Industry breakdown and full performer data", expanded=False):
                st.dataframe(industry_df, width="stretch", hide_index=True)
                st.dataframe(overall_df, width="stretch", hide_index=True)
        else:
            st.info("No stocks met the top-gainer filter.")
    else:
        st.info("Run top gainers to identify stocks and industries with current momentum.")

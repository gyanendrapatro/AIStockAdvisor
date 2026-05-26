from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
import re
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_analyst_insights(ticker: str, *, max_rows: int = 20) -> dict[str, Any]:
    """Return free analyst consensus, price targets, and rating-change history."""
    normalized = ticker.strip().upper()
    max_rows = max(1, min(int(max_rows), 100))
    yahoo = yf.Ticker(normalized)
    info = _safe_call(lambda: yahoo.info or {}, "info", normalized) or {}
    price_targets = _clean_mapping(
        _safe_call(yahoo.get_analyst_price_targets, "analyst price targets", normalized)
        or _safe_getattr(yahoo, "analyst_price_targets", normalized)
        or {}
    )
    recommendation_summary = _frame_records(
        _safe_call(yahoo.get_recommendations_summary, "recommendations summary", normalized),
        max_rows=max_rows,
    )
    upgrades_downgrades = _frame_records(
        _safe_call(yahoo.get_upgrades_downgrades, "upgrades/downgrades", normalized),
        max_rows=max_rows,
    )

    current_price = _first_number(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        info.get("previousClose"),
    )
    target_mean = _first_number(
        info.get("targetMeanPrice"),
        price_targets.get("mean"),
        price_targets.get("targetMeanPrice"),
    )
    consensus = {
        "current_price": current_price,
        "recommendation_key": info.get("recommendationKey"),
        "recommendation_mean": _clean_scalar(info.get("recommendationMean")),
        "number_of_analyst_opinions": _clean_scalar(info.get("numberOfAnalystOpinions")),
        "target_low_price": _first_number(info.get("targetLowPrice"), price_targets.get("low")),
        "target_mean_price": target_mean,
        "target_median_price": _first_number(info.get("targetMedianPrice"), price_targets.get("median")),
        "target_high_price": _first_number(info.get("targetHighPrice"), price_targets.get("high")),
        "target_upside_percent": _percent_upside(current_price, target_mean),
    }

    providers = []
    if any(value is not None for value in consensus.values()) or recommendation_summary or upgrades_downgrades:
        providers.append("yfinance")

    warnings = []
    if not providers:
        warnings.append("No free analyst consensus data returned for this ticker.")
    warnings.append("Analyst consensus and targets are free Yahoo Finance fields and may be incomplete or stale.")

    return {
        "ticker": normalized,
        "consensus": consensus,
        "price_targets": price_targets,
        "recommendation_summary": recommendation_summary,
        "upgrades_downgrades": upgrades_downgrades,
        "providers": providers,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
    }


def get_stock_events(ticker: str, *, max_rows: int = 20) -> dict[str, Any]:
    """Return free corporate action, calendar, dividend/split, and earnings-date data."""
    normalized = ticker.strip().upper()
    max_rows = max(1, min(int(max_rows), 100))
    yahoo = yf.Ticker(normalized)

    calendar = _safe_call(yahoo.get_calendar, "calendar", normalized)
    if not calendar:
        calendar = _safe_getattr(yahoo, "calendar", normalized)
    calendar_events = _calendar_events(calendar)

    actions = _frame_records(_safe_call(yahoo.get_actions, "actions", normalized), max_rows=max_rows)
    dividends = _frame_records(_safe_call(yahoo.get_dividends, "dividends", normalized), max_rows=max_rows)
    splits = _frame_records(_safe_call(yahoo.get_splits, "splits", normalized), max_rows=max_rows)
    earnings_dates = _frame_records(
        _safe_call(lambda: yahoo.get_earnings_dates(limit=max_rows), "earnings dates", normalized),
        max_rows=max_rows,
    )
    earnings_history = _frame_records(
        _safe_call(yahoo.get_earnings_history, "earnings history", normalized),
        max_rows=max_rows,
    )

    providers = []
    if calendar_events or actions or dividends or splits or earnings_dates or earnings_history:
        providers.append("yfinance")

    warnings = []
    if not providers:
        warnings.append("No free corporate-event/calendar data returned for this ticker.")
    warnings.append("Event dates are free Yahoo Finance fields and must be verified against exchange filings/company announcements.")

    return {
        "ticker": normalized,
        "calendar_events": calendar_events,
        "recent_actions": actions,
        "recent_dividends": dividends,
        "recent_splits": splits,
        "earnings_dates": earnings_dates,
        "earnings_history": earnings_history,
        "providers": providers,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
    }


def _safe_call(fn: Any, label: str, ticker: str) -> Any:
    try:
        return fn()
    except Exception as exc:
        logger.info("Yahoo Finance %s fetch failed for %s: %s", label, ticker, exc)
        return None


def _safe_getattr(obj: Any, name: str, ticker: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception as exc:
        logger.info("Yahoo Finance %s fetch failed for %s: %s", name, ticker, exc)
        return None


def _calendar_events(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame | pd.Series):
        return _frame_records(value, max_rows=30)
    if not isinstance(value, dict):
        return []

    events = []
    for key, raw in value.items():
        values = raw if isinstance(raw, list | tuple | set) else [raw]
        for item in values:
            cleaned = _clean_scalar(item)
            if cleaned is None:
                continue
            events.append({"event": _humanize_key(str(key)), "date": cleaned})
    return events


def _frame_records(value: Any, *, max_rows: int) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.Series):
        if value.empty:
            return []
        frame = value.to_frame(name=_humanize_key(str(value.name or "value")))
    elif isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        frame = value.copy()
    elif isinstance(value, dict):
        return [_clean_mapping(value)]
    else:
        return []

    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    records = frame.tail(max_rows).to_dict(orient="records")
    return [_clean_mapping(record) for record in records]


def _clean_mapping(record: dict[Any, Any]) -> dict[str, Any]:
    return {_humanize_key(str(key)): _clean_scalar(value) for key, value in record.items()}


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp | datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _clean_scalar(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _first_number(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None or value == "":
                continue
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number) or math.isinf(number):
            continue
        return round(number, 4)
    return None


def _percent_upside(current_price: float | None, target_price: float | None) -> float | None:
    if not current_price or not target_price:
        return None
    return round(((target_price / current_price) - 1) * 100, 2)


def _humanize_key(key: str) -> str:
    key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    key = re.sub(r"[^A-Za-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_").lower()
    key = key.replace("e_p_s", "eps")
    return key or "value"

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pivot:
    index: int
    value: float
    date: Any


def detect_chart_patterns(df: pd.DataFrame, lookback: int = 120) -> dict[str, Any]:
    """Detect common OHLCV chart patterns using deterministic price-action rules."""
    window_size = max(40, min(int(lookback or 120), 300))
    prices = _prepare_price_frame(df).tail(window_size).reset_index(drop=True)
    warnings: list[str] = []

    if len(prices) < 40:
        return {
            "lookback": int(len(prices)),
            "patterns": [],
            "dominant_pattern": None,
            "pattern_score": 50,
            "chart_pattern_direction": "neutral",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "warnings": ["At least 40 complete OHLC rows are required for chart-pattern detection."],
        }

    close = prices["close"].astype(float)
    high = prices["high"].astype(float)
    low = prices["low"].astype(float)
    volume = prices.get("volume", pd.Series(np.nan, index=prices.index)).astype(float)
    last_close = float(close.iloc[-1])
    tolerance = _price_tolerance(high, low, close)
    pivot_highs, pivot_lows = _find_pivots(prices, order=3)
    patterns: list[dict[str, Any]] = []

    _detect_trend_structure(patterns, pivot_highs, pivot_lows, close, tolerance)
    _detect_double_patterns(patterns, prices, pivot_highs, pivot_lows, last_close, tolerance, volume)
    _detect_rectangle(patterns, prices, last_close, tolerance, volume)
    _detect_triangles(patterns, pivot_highs, pivot_lows, last_close, tolerance, volume)
    _detect_channels(patterns, prices)
    _detect_head_shoulders(patterns, prices, pivot_highs, pivot_lows, last_close, tolerance, volume)
    _detect_cup_handle(patterns, prices, last_close, tolerance, volume)
    _detect_flags(patterns, prices, last_close, volume)

    if not pivot_highs or not pivot_lows:
        warnings.append("Only limited swing-pivot evidence was available.")

    patterns = sorted(patterns, key=lambda row: float(row.get("confidence", 0)), reverse=True)[:10]
    dominant = patterns[0] if patterns else None
    score = _pattern_score(patterns)
    direction = _score_direction(score)
    bullish_count = sum(1 for item in patterns if item.get("direction") == "bullish")
    bearish_count = sum(1 for item in patterns if item.get("direction") == "bearish")
    neutral_count = sum(1 for item in patterns if item.get("direction") == "neutral")

    return _json_safe(
        {
            "lookback": int(len(prices)),
            "patterns": patterns,
            "dominant_pattern": dominant,
            "pattern_score": score,
            "chart_pattern_direction": direction,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "support": _round_or_none(float(low.tail(20).min())),
            "resistance": _round_or_none(float(high.tail(20).max())),
            "warnings": warnings,
        }
    )


def chart_pattern_indicator_fields(patterns: dict[str, Any]) -> dict[str, Any]:
    """Return compact scoring fields derived from a chart-pattern payload."""
    dominant = patterns.get("dominant_pattern") or {}
    return {
        "chart_pattern_score": patterns.get("pattern_score"),
        "chart_pattern_direction": patterns.get("chart_pattern_direction"),
        "dominant_chart_pattern": dominant.get("pattern"),
        "chart_pattern_confidence": dominant.get("confidence"),
    }


def _prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    required = ["open", "high", "low", "close"]
    if not set(required).issubset(df.columns):
        return pd.DataFrame()

    cols = [column for column in ["date", "open", "high", "low", "close", "volume"] if column in df.columns]
    out = df.loc[:, cols].copy()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=required)
    if "date" in out.columns:
        out = out.sort_values("date")
    return out.reset_index(drop=True)


def _find_pivots(prices: pd.DataFrame, order: int = 3) -> tuple[list[Pivot], list[Pivot]]:
    high = prices["high"].astype(float).reset_index(drop=True)
    low = prices["low"].astype(float).reset_index(drop=True)
    dates = prices["date"].tolist() if "date" in prices.columns else list(range(len(prices)))
    highs: list[Pivot] = []
    lows: list[Pivot] = []

    for idx in range(order, len(prices) - order):
        high_window = high.iloc[idx - order : idx + order + 1]
        low_window = low.iloc[idx - order : idx + order + 1]
        if high.iloc[idx] >= high_window.max():
            highs.append(Pivot(idx, float(high.iloc[idx]), dates[idx]))
        if low.iloc[idx] <= low_window.min():
            lows.append(Pivot(idx, float(low.iloc[idx]), dates[idx]))

    return _dedupe_pivots(highs, "high", order), _dedupe_pivots(lows, "low", order)


def _dedupe_pivots(pivots: list[Pivot], kind: str, min_gap: int) -> list[Pivot]:
    if not pivots:
        return []

    deduped: list[Pivot] = [pivots[0]]
    for pivot in pivots[1:]:
        previous = deduped[-1]
        if pivot.index - previous.index <= min_gap:
            keep_new = pivot.value > previous.value if kind == "high" else pivot.value < previous.value
            if keep_new:
                deduped[-1] = pivot
        else:
            deduped.append(pivot)
    return deduped


def _detect_trend_structure(
    patterns: list[dict[str, Any]],
    highs: list[Pivot],
    lows: list[Pivot],
    close: pd.Series,
    tolerance: float,
) -> None:
    if len(highs) < 3 or len(lows) < 3:
        return

    recent_highs = highs[-3:]
    recent_lows = lows[-3:]
    high_values = [pivot.value for pivot in recent_highs]
    low_values = [pivot.value for pivot in recent_lows]
    ret20 = _pct_change(close, 20)

    if _rising(high_values, tolerance / 2) and _rising(low_values, tolerance / 2):
        confidence = 60 + min(18, max(0, ret20) * 100)
        _add_pattern(
            patterns,
            "higher_high_higher_low",
            "Higher high / higher low trend",
            "bullish",
            confidence,
            "active",
            "Recent swing highs and swing lows are rising.",
            support=recent_lows[-1].value,
            invalidation_level=recent_lows[-1].value,
        )
    elif _falling(high_values, tolerance / 2) and _falling(low_values, tolerance / 2):
        confidence = 60 + min(18, abs(min(0, ret20)) * 100)
        _add_pattern(
            patterns,
            "lower_high_lower_low",
            "Lower high / lower low trend",
            "bearish",
            confidence,
            "active",
            "Recent swing highs and swing lows are falling.",
            resistance=recent_highs[-1].value,
            invalidation_level=recent_highs[-1].value,
        )


def _detect_double_patterns(
    patterns: list[dict[str, Any]],
    prices: pd.DataFrame,
    highs: list[Pivot],
    lows: list[Pivot],
    last_close: float,
    tolerance: float,
    volume: pd.Series,
) -> None:
    min_depth = max(0.04, tolerance * 1.4)
    best_bottom = None
    for first, second in _pivot_pairs(lows[-7:]):
        separation = second.index - first.index
        if not 7 <= separation <= 90:
            continue
        similarity = _relative_gap(first.value, second.value)
        if similarity > max(0.045, tolerance * 1.4):
            continue
        between = prices.iloc[first.index : second.index + 1]
        neckline = float(between["high"].max())
        depth = neckline / max(min(first.value, second.value), 1e-9) - 1
        if depth < min_depth:
            continue
        status = "breakout_confirmed" if last_close > neckline * 1.005 else "forming"
        confidence = 58 + min(22, depth * 120) + (8 if status == "breakout_confirmed" else 0)
        candidate = (confidence, first, second, neckline, status, depth)
        if best_bottom is None or confidence > best_bottom[0]:
            best_bottom = candidate

    if best_bottom:
        confidence, first, second, neckline, status, depth = best_bottom
        _add_pattern(
            patterns,
            "double_bottom",
            "Double bottom",
            "bullish",
            confidence,
            status,
            f"Two similar swing lows with a neckline around {_format_price(neckline)}.",
            breakout_level=neckline,
            invalidation_level=min(first.value, second.value) * 0.985,
            volume_confirmation=_volume_confirmation(volume),
            depth_percent=depth * 100,
        )

    best_top = None
    for first, second in _pivot_pairs(highs[-7:]):
        separation = second.index - first.index
        if not 7 <= separation <= 90:
            continue
        similarity = _relative_gap(first.value, second.value)
        if similarity > max(0.045, tolerance * 1.4):
            continue
        between = prices.iloc[first.index : second.index + 1]
        neckline = float(between["low"].min())
        depth = min(first.value, second.value) / max(neckline, 1e-9) - 1
        if depth < min_depth:
            continue
        status = "breakdown_confirmed" if last_close < neckline * 0.995 else "forming"
        confidence = 58 + min(22, depth * 120) + (8 if status == "breakdown_confirmed" else 0)
        candidate = (confidence, first, second, neckline, status, depth)
        if best_top is None or confidence > best_top[0]:
            best_top = candidate

    if best_top:
        confidence, first, second, neckline, status, depth = best_top
        _add_pattern(
            patterns,
            "double_top",
            "Double top",
            "bearish",
            confidence,
            status,
            f"Two similar swing highs with a neckline around {_format_price(neckline)}.",
            breakdown_level=neckline,
            invalidation_level=max(first.value, second.value) * 1.015,
            volume_confirmation=_volume_confirmation(volume),
            depth_percent=depth * 100,
        )


def _detect_rectangle(
    patterns: list[dict[str, Any]],
    prices: pd.DataFrame,
    last_close: float,
    tolerance: float,
    volume: pd.Series,
) -> None:
    if len(prices) < 35:
        return

    base = prices.iloc[-61:-1] if len(prices) > 61 else prices.iloc[:-1]
    if len(base) < 30:
        return

    resistance = float(base["high"].max())
    support = float(base["low"].min())
    midpoint = (resistance + support) / 2
    if midpoint <= 0:
        return

    width = (resistance - support) / midpoint
    high_tests = int((base["high"] >= resistance * (1 - max(0.015, tolerance))).sum())
    low_tests = int((base["low"] <= support * (1 + max(0.015, tolerance))).sum())
    if width > max(0.16, tolerance * 4) or high_tests < 2 or low_tests < 2:
        return

    if last_close > resistance * 1.005:
        _add_pattern(
            patterns,
            "rectangle_breakout",
            "Rectangle breakout",
            "bullish",
            68 + min(12, width * 50),
            "breakout_confirmed",
            "Price closed above a multi-week horizontal resistance zone.",
            breakout_level=resistance,
            support=support,
            volume_confirmation=_volume_confirmation(volume),
        )
    elif last_close < support * 0.995:
        _add_pattern(
            patterns,
            "rectangle_breakdown",
            "Rectangle breakdown",
            "bearish",
            68 + min(12, width * 50),
            "breakdown_confirmed",
            "Price closed below a multi-week horizontal support zone.",
            breakdown_level=support,
            resistance=resistance,
            volume_confirmation=_volume_confirmation(volume),
        )
    else:
        _add_pattern(
            patterns,
            "rectangle_consolidation",
            "Rectangle consolidation",
            "neutral",
            58 + min(10, high_tests + low_tests),
            "range_bound",
            "Price is trading inside a tested horizontal range.",
            support=support,
            resistance=resistance,
        )


def _detect_triangles(
    patterns: list[dict[str, Any]],
    highs: list[Pivot],
    lows: list[Pivot],
    last_close: float,
    tolerance: float,
    volume: pd.Series,
) -> None:
    recent_highs = _recent_pivots(highs, 80)[-4:]
    recent_lows = _recent_pivots(lows, 80)[-4:]
    if len(recent_highs) < 3 or len(recent_lows) < 3:
        return

    high_values = [pivot.value for pivot in recent_highs]
    low_values = [pivot.value for pivot in recent_lows]
    high_flat = _relative_gap(max(high_values), min(high_values)) <= max(0.035, tolerance * 1.2)
    low_flat = _relative_gap(max(low_values), min(low_values)) <= max(0.035, tolerance * 1.2)
    highs_falling = _falling(high_values, tolerance / 3)
    lows_rising = _rising(low_values, tolerance / 3)
    highs_rising = _rising(high_values, tolerance / 3)
    lows_falling = _falling(low_values, tolerance / 3)

    if high_flat and lows_rising:
        resistance = float(np.mean(high_values[-3:]))
        status = "breakout_confirmed" if last_close > resistance * 1.005 else "breakout_watch"
        _add_pattern(
            patterns,
            "ascending_triangle",
            "Ascending triangle",
            "bullish",
            63 + (7 if status == "breakout_confirmed" else 0),
            status,
            "Horizontal resistance with rising swing lows.",
            breakout_level=resistance,
            support=recent_lows[-1].value,
            volume_confirmation=_volume_confirmation(volume),
        )
    elif low_flat and highs_falling:
        support = float(np.mean(low_values[-3:]))
        status = "breakdown_confirmed" if last_close < support * 0.995 else "breakdown_watch"
        _add_pattern(
            patterns,
            "descending_triangle",
            "Descending triangle",
            "bearish",
            63 + (7 if status == "breakdown_confirmed" else 0),
            status,
            "Horizontal support with falling swing highs.",
            breakdown_level=support,
            resistance=recent_highs[-1].value,
            volume_confirmation=_volume_confirmation(volume),
        )
    elif highs_falling and lows_rising:
        _add_pattern(
            patterns,
            "symmetrical_triangle",
            "Symmetrical triangle",
            "neutral",
            61,
            "compression",
            "Lower swing highs and higher swing lows show price compression.",
            breakout_level=recent_highs[-1].value,
            breakdown_level=recent_lows[-1].value,
        )
    elif highs_rising and lows_falling:
        _add_pattern(
            patterns,
            "broadening_formation",
            "Broadening formation",
            "neutral",
            58,
            "volatile_range",
            "Higher swing highs and lower swing lows show expanding volatility.",
            breakout_level=recent_highs[-1].value,
            breakdown_level=recent_lows[-1].value,
        )


def _detect_channels(patterns: list[dict[str, Any]], prices: pd.DataFrame) -> None:
    if len(prices) < 45:
        return

    window = prices.tail(min(80, len(prices))).reset_index(drop=True)
    high = window["high"].astype(float)
    low = window["low"].astype(float)
    close = window["close"].astype(float)
    high_slope, high_line = _linear_fit(high)
    low_slope, low_line = _linear_fit(low)
    mean_price = float(close.mean())
    if mean_price <= 0:
        return

    high_move = high_slope * len(window) / mean_price
    low_move = low_slope * len(window) / mean_price
    parallel_gap = abs(high_move - low_move)
    channel_width = float((high_line[-1] - low_line[-1]) / mean_price)
    if channel_width <= 0 or channel_width > 0.30 or parallel_gap > 0.08:
        return

    if high_move > 0.04 and low_move > 0.04:
        _add_pattern(
            patterns,
            "ascending_channel",
            "Ascending channel",
            "bullish",
            59 + min(10, high_move * 100),
            "active",
            "Regression bands for highs and lows are rising in parallel.",
            support=float(low_line[-1]),
            resistance=float(high_line[-1]),
        )
    elif high_move < -0.04 and low_move < -0.04:
        _add_pattern(
            patterns,
            "descending_channel",
            "Descending channel",
            "bearish",
            59 + min(10, abs(high_move) * 100),
            "active",
            "Regression bands for highs and lows are falling in parallel.",
            support=float(low_line[-1]),
            resistance=float(high_line[-1]),
        )


def _detect_head_shoulders(
    patterns: list[dict[str, Any]],
    prices: pd.DataFrame,
    highs: list[Pivot],
    lows: list[Pivot],
    last_close: float,
    tolerance: float,
    volume: pd.Series,
) -> None:
    recent_highs = _recent_pivots(highs, 100)
    if len(recent_highs) >= 3:
        left, head, right = recent_highs[-3:]
        shoulder_gap = _relative_gap(left.value, right.value)
        head_gap = head.value / max(left.value, right.value, 1e-9) - 1
        neckline_lows = [pivot.value for pivot in lows if left.index < pivot.index < right.index]
        if head_gap >= max(0.04, tolerance) and shoulder_gap <= max(0.08, tolerance * 2.2) and neckline_lows:
            neckline = float(np.mean(sorted(neckline_lows)[:2]))
            status = "breakdown_confirmed" if last_close < neckline * 0.995 else "forming"
            _add_pattern(
                patterns,
                "head_and_shoulders",
                "Head and shoulders",
                "bearish",
                62 + (8 if status == "breakdown_confirmed" else 0),
                status,
                "Middle swing high is materially above two similar shoulder highs.",
                breakdown_level=neckline,
                invalidation_level=head.value * 1.01,
                volume_confirmation=_volume_confirmation(volume),
            )

    recent_lows = _recent_pivots(lows, 100)
    if len(recent_lows) >= 3:
        left, head, right = recent_lows[-3:]
        shoulder_gap = _relative_gap(left.value, right.value)
        head_gap = min(left.value, right.value) / max(head.value, 1e-9) - 1
        neckline_highs = [pivot.value for pivot in highs if left.index < pivot.index < right.index]
        if head_gap >= max(0.04, tolerance) and shoulder_gap <= max(0.08, tolerance * 2.2) and neckline_highs:
            neckline = float(np.mean(sorted(neckline_highs, reverse=True)[:2]))
            status = "breakout_confirmed" if last_close > neckline * 1.005 else "forming"
            _add_pattern(
                patterns,
                "inverse_head_and_shoulders",
                "Inverse head and shoulders",
                "bullish",
                62 + (8 if status == "breakout_confirmed" else 0),
                status,
                "Middle swing low is materially below two similar shoulder lows.",
                breakout_level=neckline,
                invalidation_level=head.value * 0.99,
                volume_confirmation=_volume_confirmation(volume),
            )


def _detect_cup_handle(
    patterns: list[dict[str, Any]],
    prices: pd.DataFrame,
    last_close: float,
    tolerance: float,
    volume: pd.Series,
) -> None:
    if len(prices) < 80:
        return

    window = prices.tail(min(140, len(prices))).reset_index(drop=True)
    close = window["close"].astype(float)
    third = len(window) // 3
    left_idx = int(close.iloc[:third].idxmax())
    trough_idx = int(close.iloc[left_idx : len(window) - third].idxmin())
    right_idx = int(close.iloc[trough_idx:].idxmax())
    left_high = float(close.iloc[left_idx])
    trough = float(close.iloc[trough_idx])
    right_high = float(close.iloc[right_idx])
    if not (left_idx < trough_idx < right_idx):
        return

    depth = left_high / max(trough, 1e-9) - 1
    rim_gap = _relative_gap(left_high, right_high)
    handle = close.tail(15)
    handle_pullback = float(handle.max() / max(handle.min(), 1e-9) - 1)
    if 0.12 <= depth <= 0.55 and rim_gap <= max(0.12, tolerance * 2.5) and handle_pullback <= 0.16:
        breakout_level = max(left_high, right_high)
        status = "breakout_confirmed" if last_close > breakout_level * 1.005 else "forming"
        _add_pattern(
            patterns,
            "cup_and_handle",
            "Cup and handle",
            "bullish",
            60 + min(10, depth * 50) + (7 if status == "breakout_confirmed" else 0),
            status,
            "Rounded recovery with a shallow recent handle.",
            breakout_level=breakout_level,
            invalidation_level=float(handle.min()),
            volume_confirmation=_volume_confirmation(volume),
            depth_percent=depth * 100,
        )


def _detect_flags(
    patterns: list[dict[str, Any]],
    prices: pd.DataFrame,
    last_close: float,
    volume: pd.Series,
) -> None:
    if len(prices) < 45:
        return

    close = prices["close"].astype(float).reset_index(drop=True)
    impulse = close.iloc[-16] / close.iloc[-36] - 1
    consolidation = close.tail(15)
    range_pct = float((consolidation.max() - consolidation.min()) / max(last_close, 1e-9))
    drift = close.iloc[-1] / close.iloc[-16] - 1

    if impulse > 0.14 and range_pct < 0.10 and drift > -0.08:
        _add_pattern(
            patterns,
            "bull_flag",
            "Bull flag",
            "bullish",
            61 + min(10, impulse * 50),
            "continuation_watch",
            "Strong prior advance followed by tight consolidation.",
            breakout_level=float(consolidation.max()),
            invalidation_level=float(consolidation.min()),
            volume_confirmation=_volume_confirmation(volume),
        )
    elif impulse < -0.14 and range_pct < 0.10 and drift < 0.08:
        _add_pattern(
            patterns,
            "bear_flag",
            "Bear flag",
            "bearish",
            61 + min(10, abs(impulse) * 50),
            "continuation_watch",
            "Strong prior decline followed by tight consolidation.",
            breakdown_level=float(consolidation.min()),
            invalidation_level=float(consolidation.max()),
            volume_confirmation=_volume_confirmation(volume),
        )


def _add_pattern(
    patterns: list[dict[str, Any]],
    pattern: str,
    name: str,
    direction: str,
    confidence: float,
    status: str,
    notes: str,
    **levels: Any,
) -> None:
    row = {
        "pattern": pattern,
        "name": name,
        "direction": direction,
        "confidence": round(max(0, min(100, float(confidence))), 2),
        "status": status,
        "notes": notes,
    }
    for key, value in levels.items():
        if value is not None:
            if isinstance(value, bool | np.bool_):
                row[key] = bool(value)
            elif isinstance(value, int | float | np.number):
                row[key] = _round_or_none(value)
            else:
                row[key] = value
    patterns.append(row)


def _pattern_score(patterns: list[dict[str, Any]]) -> float:
    if not patterns:
        return 50

    bullish = [
        (float(item.get("confidence", 50)) - 50) * _status_weight(item)
        for item in patterns
        if item.get("direction") == "bullish"
    ]
    bearish = [
        (float(item.get("confidence", 50)) - 50) * _status_weight(item)
        for item in patterns
        if item.get("direction") == "bearish"
    ]
    bullish_strength = (sum(bullish) / len(bullish)) if bullish else 0
    bearish_strength = (sum(bearish) / len(bearish)) if bearish else 0
    score = 50 + min(35, bullish_strength * 0.65 + len(bullish) * 2) - min(35, bearish_strength * 0.65 + len(bearish) * 2)
    return round(max(0, min(100, score)), 2)


def _status_weight(item: dict[str, Any]) -> float:
    status = str(item.get("status", "")).lower()
    if "confirmed" in status:
        return 1.2
    if status == "active":
        return 1.0
    if status in {"forming", "breakout_watch", "breakdown_watch", "continuation_watch"}:
        return 0.7
    return 0.8


def _score_direction(score: float) -> str:
    if score >= 58:
        return "bullish"
    if score <= 42:
        return "bearish"
    return "neutral"


def _price_tolerance(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    last_close = float(close.iloc[-1])
    if not np.isfinite(atr) or last_close <= 0:
        return 0.035
    return max(0.025, min(0.08, (atr / last_close) * 1.8))


def _volume_confirmation(volume: pd.Series) -> bool | None:
    cleaned = pd.to_numeric(volume, errors="coerce").dropna()
    if len(cleaned) < 20:
        return None
    avg = float(cleaned.tail(20).mean())
    if avg <= 0:
        return None
    return bool(float(cleaned.iloc[-1]) >= avg * 1.15)


def _pivot_pairs(pivots: list[Pivot]) -> list[tuple[Pivot, Pivot]]:
    pairs: list[tuple[Pivot, Pivot]] = []
    for i, first in enumerate(pivots):
        for second in pivots[i + 1 :]:
            pairs.append((first, second))
    return pairs


def _recent_pivots(pivots: list[Pivot], bars: int) -> list[Pivot]:
    if not pivots:
        return []
    last_idx = pivots[-1].index
    return [pivot for pivot in pivots if last_idx - pivot.index <= bars]


def _relative_gap(a: float, b: float) -> float:
    mean = (abs(float(a)) + abs(float(b))) / 2
    if mean <= 0:
        return 0
    return abs(float(a) - float(b)) / mean


def _rising(values: list[float], tolerance: float) -> bool:
    return all(curr >= prev * (1 + tolerance) for prev, curr in zip(values, values[1:]))


def _falling(values: list[float], tolerance: float) -> bool:
    return all(curr <= prev * (1 - tolerance) for prev, curr in zip(values, values[1:]))


def _linear_fit(series: pd.Series) -> tuple[float, np.ndarray]:
    y = series.astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), slope * x + intercept


def _pct_change(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return 0
    start = float(close.iloc[-periods])
    if start == 0:
        return 0
    return float(close.iloc[-1] / start - 1)


def _format_price(value: float) -> str:
    return f"{value:.2f}"


def _round_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, 4)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _round_or_none(value)
    if isinstance(value, float):
        return _round_or_none(value)
    return value

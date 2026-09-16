from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from stock_advisor.analysis.indicators import add_indicators
from stock_advisor.data.market_data import (
    INDEX_BACKFILL_DEFINITIONS,
    SECTOR_TO_INDEX_NAME,
    compute_fundamentals_ratios,
    get_fundamentals_history,
    get_scanner_benchmark_frame,
    get_scanner_price_frame,
)

# This module is the QUANTITATIVE FILTER only -- it identifies "stocks exhibiting technical/
# market characteristics consistent with a potential turnaround," never a buy/sell/best/worst
# recommendation. Fundamental/management/valuation judgment is left entirely to the separate
# research stage that consumes this module's exported candidate list.

MODES = ("EARLY", "CONFIRMED", "DEEP")
LOOKBACK_DAYS_DEFAULT = 620  # ~200DMA + its 20-day-ago slope comparison + 52w/12m windows, with margin
MIN_HISTORY_DAYS_DEFAULT = 250
CROSSOVER_LOOKBACK_DAYS = 60  # "most recent crossover" search window, matches the existing MA-crossover tab's magnitude
DEEP_RECENT_CROSS_DAYS = 10  # DEEP mode's "crossed recently" window
RETURN_PERIODS = {"return_5d": 5, "return_1m": 20, "return_3m": 60, "return_6m": 120, "return_12m": 252}
FUNDAMENTALS_MIN_QUARTERS = 4
FUNDAMENTALS_REPORTING_LAG_DAYS = 45  # Indian companies typically report ~45 days after quarter-end

INDEX_TICKER_BY_ARCHIVE_NAME: dict[str, str] = {name: ticker for name, ticker, _, _ in INDEX_BACKFILL_DEFINITIONS}

ALL_FALSE_TURNAROUND_FLAG_KEYS = [
    "one_quarter_only_improvement",
    "low_base_effect",
    "pat_growth_without_revenue_growth",
    "pat_growth_without_ebitda_improvement",
    "profit_without_ocf_improvement",
    "margin_improvement_without_revenue_stabilization",
    "working_capital_deterioration",
    "receivables_increasing_disproportionately",
    "inventory_deterioration",
    "exceptional_income_dependence",
    "other_income_dependence",
    "tax_reversal_dependence",
    "asset_sale_dependence",
    "debt_restructuring",
    "dilution",
    "negative_cash_conversion",
]
# These have no source columns anywhere in fundamentals_history/fundamentals_snapshot (no
# balance-sheet detail, no cash-flow-statement line items beyond operating_cash_flow, no
# shares-outstanding time series) -- always UNKNOWN, never guessed.
_UNCOMPUTABLE_FLAG_KEYS = [
    "working_capital_deterioration",
    "receivables_increasing_disproportionately",
    "inventory_deterioration",
    "exceptional_income_dependence",
    "other_income_dependence",
    "tax_reversal_dependence",
    "asset_sale_dependence",
    "debt_restructuring",
    "dilution",
    "negative_cash_conversion",
]

STAGE_ORDER = {"EARLY DISCOVERY": 0, "DEVELOPING": 1, "CONFIRMED": 2, "MATURE / RECOGNIZED": 3, "FALSE / UNCONFIRMED": 4}

DEFAULT_SHORTLIST_THRESHOLDS: dict[str, dict[str, Any]] = {
    "EARLY": {"min_technical_score": 55.0, "recognition_in": ("LOW", "MODERATE")},
    "CONFIRMED": {"min_technical_score": 65.0, "require_ma_confirmation": True, "require_positive_rs_3m": True},
    "DEEP": {"min_technical_score": 50.0},
}


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, value)))


def _n(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _safe_return(close: pd.Series, periods: int) -> float | None:
    """Same convention as sector_rotation._return(): trading-day-count lookback, not calendar days."""
    if close is None or len(close) <= periods:
        return None
    start = float(close.iloc[-periods - 1])
    end = float(close.iloc[-1])
    if start <= 0:
        return None
    return round(end / start - 1, 4)


def _slope(series: pd.Series, periods: int) -> float | None:
    if series is None or len(series) <= periods:
        return None
    cur, prior = series.iloc[-1], series.iloc[-1 - periods]
    if pd.isna(cur) or pd.isna(prior) or prior == 0:
        return None
    return round(float(cur / prior - 1), 4)


def _cross_events(fast: pd.Series, slow: pd.Series) -> tuple[pd.Series, pd.Series]:
    current = np.sign(fast - slow)
    previous = np.sign(fast.shift(1) - slow.shift(1))
    crossed_up = (current > 0) & (previous <= 0)
    crossed_down = (current < 0) & (previous >= 0)
    return crossed_up, crossed_down


# ---------------------------------------------------------------------------
# Step 5: price-data preparation
# ---------------------------------------------------------------------------

def prepare_universe_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, list[str]], dict[str, int]]:
    """Sort, drop invalid OHLCV rows, and dedupe dual NSE/BSE listings that share one bare
    symbol -- returns the cleaned frame plus a per-ticker_id exclusion-reason log and summary
    counts. Never interpolates, forward-fills, or fabricates a trading day; a row is either
    exactly what's in price_history_cache or it's dropped and flagged, never repaired."""
    exclusions: dict[int, list[str]] = defaultdict(list)
    if raw_df.empty:
        return raw_df, exclusions, {"invalid_rows_dropped": 0, "duplicate_listings_dropped": 0}

    df = raw_df.sort_values(["ticker_id", "date"]).reset_index(drop=True)

    valid_mask = (
        (df["high"] >= df["low"])
        & (df["open"] <= df["high"]) & (df["open"] >= df["low"])
        & (df["close"] <= df["high"]) & (df["close"] >= df["low"])
        & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
        & (df["volume"] >= 0)
    )
    invalid_rows = int((~valid_mask).sum())
    for tid in df.loc[~valid_mask, "ticker_id"].unique().tolist():
        exclusions[tid].append("invalid_ohlcv_dropped")
    df = df[valid_mask].copy()

    counts = df.groupby("ticker_id")["date"].transform("count")
    df["_row_count"] = counts
    df["_nse_pref"] = df["exchange"].fillna("").str.contains("NSE").astype(int)
    key = df[["symbol", "ticker_id", "_nse_pref", "_row_count"]].drop_duplicates("ticker_id")
    key = key.sort_values(["symbol", "_nse_pref", "_row_count"], ascending=[True, False, False])
    keep_ids = set(key.drop_duplicates("symbol", keep="first")["ticker_id"])
    dropped_ids = set(key["ticker_id"]) - keep_ids
    for tid in dropped_ids:
        exclusions[tid].append("duplicate_listing_dropped")
    df = df[df["ticker_id"].isin(keep_ids)].drop(columns=["_row_count", "_nse_pref"])

    return df, exclusions, {"invalid_rows_dropped": invalid_rows, "duplicate_listings_dropped": len(dropped_ids)}


# ---------------------------------------------------------------------------
# Step 6-15: per-ticker technical indicators (as-of-date bound, no look-ahead)
# ---------------------------------------------------------------------------

def compute_ticker_indicators(
    ticker_df: pd.DataFrame,
    benchmark_returns: dict[str, float | None],
    sector_index_returns: dict[str, float | None] | None,
) -> dict[str, Any] | None:
    """All fields computed only from ticker_df, which the caller already bounded to
    date <= as_of_date -- that SQL bound is the scanner's entire no-look-ahead guarantee;
    nothing here looks past the last row it's handed."""
    tdf = ticker_df.sort_values("date").reset_index(drop=True)
    ind = add_indicators(tdf[["date", "open", "high", "low", "close", "volume"]])
    if ind.empty:
        return None
    ind["volume_sma_50"] = ind["volume"].rolling(50).mean()
    close, volume = ind["close"], ind["volume"]
    last = ind.iloc[-1]

    up20, _ = _cross_events(close, ind["sma_20"])
    up50, _ = _cross_events(close, ind["sma_50"])
    up2050, _ = _cross_events(ind["sma_20"], ind["sma_50"])
    up50200, _ = _cross_events(ind["sma_50"], ind["sma_200"])

    crossovers: list[tuple[str, Any]] = []
    for label, up_series in [
        ("close_crossed_above_20dma", up20),
        ("close_crossed_above_50dma", up50),
        ("20dma_crossed_above_50dma", up2050),
        ("50dma_crossed_above_200dma", up50200),
    ]:
        recent = up_series.tail(CROSSOVER_LOOKBACK_DAYS)
        hits = ind.loc[recent[recent].index]
        if not hits.empty:
            crossovers.append((label, hits["date"].iloc[-1]))
    latest_crossover = crossover_date = None
    if crossovers:
        crossovers.sort(key=lambda pair: pair[1])
        latest_crossover, raw_date = crossovers[-1]
        crossover_date = str(pd.Timestamp(raw_date).date())

    returns = {key: _safe_return(close, periods) for key, periods in RETURN_PERIODS.items()}
    return_1d = _safe_return(close, 1)

    vol5 = volume.tail(5).mean()
    vol_prior20 = volume.iloc[-25:-5].mean() if len(volume) >= 25 else None
    volume_trend = round(float(vol5 / vol_prior20), 3) if vol_prior20 and vol_prior20 > 0 and pd.notna(vol5) else None

    recent20 = ind.tail(20)
    up_days = int((recent20["close"] > recent20["close"].shift(1)).sum())
    pct_up_volume_days = round(up_days / len(recent20), 3) if len(recent20) else None

    def _rel(period_key: str) -> float | None:
        stock_r, bench_r = returns.get(period_key), benchmark_returns.get(period_key)
        return round(stock_r - bench_r, 4) if stock_r is not None and bench_r is not None else None

    sector_rs_3m = None
    if sector_index_returns and returns.get("return_3m") is not None and sector_index_returns.get("return_3m") is not None:
        sector_rs_3m = round(returns["return_3m"] - sector_index_returns["return_3m"], 4)

    dma20, dma50, dma200 = _n(last["sma_20"]), _n(last["sma_50"]), _n(last["sma_200"])
    close_val, high_52w = _n(last["close"]), _n(last["high_52w"])
    vol_val, avg_vol_20 = _n(last["volume"]), _n(last["volume_sma_20"])
    avg_vol_50 = _n(last["volume_sma_50"])

    return {
        "close": close_val,
        "dma20": dma20, "dma50": dma50, "dma200": dma200,
        "dma20_slope": _slope(ind["sma_20"], 5),
        "dma50_slope": _slope(ind["sma_50"], 10),
        "dma200_slope": _slope(ind["sma_200"], 20),
        **returns,
        "high_52w": high_52w, "low_52w": _n(last["low_52w"]),
        "distance_from_52w_high": _n(last["distance_from_52w_high"]),
        "distance_from_52w_low": _n(last["distance_from_52w_low"]),
        "volume": vol_val, "avg_volume_20d": avg_vol_20, "avg_volume_50d": avg_vol_50,
        "volume_ratio_20d": _n(last["volume_ratio"]),
        "volume_ratio_50d": round(vol_val / avg_vol_50, 3) if vol_val is not None and avg_vol_50 else None,
        "volume_trend": volume_trend,
        "pct_up_volume_days": pct_up_volume_days,
        "rs_1m": _rel("return_1m"), "rs_3m": _rel("return_3m"), "rs_6m": _rel("return_6m"),
        "sector_rs_3m": sector_rs_3m,
        "latest_crossover": latest_crossover, "crossover_date": crossover_date,
        "data_points": int(len(ind)),
        # internal-only fields (rule evaluation), stripped before export
        "_return_1d": return_1d,
        "_close_above_dma20": bool(close_val is not None and dma20 is not None and close_val > dma20),
        "_close_above_dma50": bool(close_val is not None and dma50 is not None and close_val > dma50),
        "_close_above_dma200": bool(close_val is not None and dma200 is not None and close_val > dma200),
        "_dma20_above_dma50": bool(dma20 is not None and dma50 is not None and dma20 > dma50),
        "_dma50_above_dma200": bool(dma50 is not None and dma200 is not None and dma50 > dma200),
        "_dma50_slope_20d": _slope(ind["sma_50"], 20),
        "_close_crossed_20dma_recent": bool(up20.tail(DEEP_RECENT_CROSS_DAYS).any()),
        "_close_crossed_50dma_recent": bool(up50.tail(DEEP_RECENT_CROSS_DAYS).any()),
    }


# ---------------------------------------------------------------------------
# Step 18-20: fundamentals (conditional -- never fabricated when data is missing)
# ---------------------------------------------------------------------------

def _qoq_growth_seq(values: list[float | None]) -> list[float | None]:
    seq: list[float | None] = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if prev is None or cur is None or prev == 0:
            seq.append(None)
        else:
            seq.append(round((cur - prev) / abs(prev), 4))
    return seq


def _is_strictly_improving(seq: list[float | None]) -> bool | None:
    values = [v for v in seq if v is not None]
    if len(values) < 3:
        return None
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def analyze_fundamentals(ticker: str | None, as_of_date) -> dict[str, Any]:
    """Only ever computes a score when >=4 consecutive quarters of non-null revenue+PAT exist,
    with a reporting-lag cutoff so a quarter's figures are never used before they'd realistically
    have been public (period_end_date + FUNDAMENTALS_REPORTING_LAG_DAYS <= as_of_date). Returns
    fundamental_score=None ("NOT AVAILABLE") otherwise -- never a fabricated number."""
    unavailable = {
        "fundamental_data_available": False,
        "fundamental_score": None,
        "fundamental_inflection": None,
        "earnings_acceleration": None,
        "false_turnaround_flags": {k: "UNKNOWN — FUNDAMENTAL REVIEW REQUIRED" for k in ALL_FALSE_TURNAROUND_FLAG_KEYS},
        "data_quality_flags": ["fundamental_verification_required"],
    }
    if not ticker:
        return unavailable

    history = get_fundamentals_history(ticker, period_type="quarterly")
    as_of = pd.Timestamp(as_of_date)
    cutoff = as_of - pd.Timedelta(days=FUNDAMENTALS_REPORTING_LAG_DAYS)
    usable = [
        row for row in history
        if row.get("period_end_date")
        and pd.Timestamp(row["period_end_date"]) <= cutoff
        and row.get("total_revenue") is not None
        and row.get("net_income") is not None
    ]
    usable.sort(key=lambda r: r["period_end_date"])
    if len(usable) < FUNDAMENTALS_MIN_QUARTERS:
        return unavailable

    last4 = usable[-4:]
    revenue = [r.get("total_revenue") for r in last4]
    pat = [r.get("net_income") for r in last4]
    ebitda = [r.get("ebitda") for r in last4]
    margin = [(e / r) if e is not None and r not in (None, 0) else None for e, r in zip(ebitda, revenue)]

    revenue_growth_seq = _qoq_growth_seq(revenue)
    pat_growth_seq = _qoq_growth_seq(pat)
    ebitda_growth_seq = _qoq_growth_seq(ebitda)

    revenue_improving = _is_strictly_improving(revenue_growth_seq)
    pat_improving = _is_strictly_improving(pat_growth_seq)
    ebitda_improving = _is_strictly_improving(ebitda_growth_seq)
    margin_improving = _is_strictly_improving(margin)
    earnings_accel = pat_improving

    fundamental_score = round(
        25 * int(bool(revenue_improving)) + 25 * int(bool(pat_improving))
        + 25 * int(bool(margin_improving)) + 25 * int(bool(earnings_accel)),
        2,
    )

    flags: dict[str, Any] = {k: "UNKNOWN — FUNDAMENTAL REVIEW REQUIRED" for k in _UNCOMPUTABLE_FLAG_KEYS}
    pat_valid = [v for v in pat_growth_seq if v is not None]
    flags["one_quarter_only_improvement"] = bool(
        pat_valid and pat_valid[-1] > 0 and not all(v > 0 for v in pat_valid[:-1])
    ) if len(pat_valid) >= 2 else "UNKNOWN — FUNDAMENTAL REVIEW REQUIRED"
    flags["pat_growth_without_revenue_growth"] = (
        bool(pat_improving) and not bool(revenue_improving)
        if pat_improving is not None and revenue_improving is not None else "UNKNOWN — FUNDAMENTAL REVIEW REQUIRED"
    )
    flags["pat_growth_without_ebitda_improvement"] = (
        bool(pat_improving) and not bool(ebitda_improving)
        if pat_improving is not None and ebitda_improving is not None else "UNKNOWN — FUNDAMENTAL REVIEW REQUIRED"
    )
    flags["margin_improvement_without_revenue_stabilization"] = (
        bool(margin_improving) and not bool(revenue_improving)
        if margin_improving is not None and revenue_improving is not None else "UNKNOWN — FUNDAMENTAL REVIEW REQUIRED"
    )
    annual_history = [r for r in history if r.get("period_type") == "annual" and r.get("operating_cash_flow") is not None]
    if len(annual_history) >= 2:
        annual_history.sort(key=lambda r: r["period_end_date"])
        ocf_recent, ocf_prior = annual_history[-1]["operating_cash_flow"], annual_history[-2]["operating_cash_flow"]
        pat_recent = annual_history[-1].get("net_income")
        flags["profit_without_ocf_improvement"] = bool(
            pat_recent is not None and pat_recent > 0 and ocf_recent is not None and ocf_prior is not None and ocf_recent < ocf_prior
        )
        flags["profit_without_ocf_improvement_basis"] = "annual (quarterly OCF is not reliably populated in this database)"
    else:
        flags["profit_without_ocf_improvement"] = "UNKNOWN — FUNDAMENTAL REVIEW REQUIRED"
    yoy_row = next(
        (r for r in usable if abs((pd.Timestamp(r["period_end_date"]) - pd.Timestamp(last4[-1]["period_end_date"])).days - 365) <= 20),
        None,
    )
    if yoy_row and yoy_row.get("net_income") not in (None, 0):
        flags["low_base_effect"] = bool(pat_growth_seq and pat_growth_seq[-1] is not None and pat_growth_seq[-1] > 1.0 and yoy_row["net_income"] < 0)
    else:
        flags["low_base_effect"] = "UNKNOWN — FUNDAMENTAL REVIEW REQUIRED"

    return {
        "fundamental_data_available": True,
        "fundamental_score": fundamental_score,
        "fundamental_inflection": "YES" if (revenue_improving or pat_improving) else "NO",
        "earnings_acceleration": "YES" if earnings_accel else ("NO" if earnings_accel is False else None),
        "false_turnaround_flags": flags,
        "data_quality_flags": [],
    }


# ---------------------------------------------------------------------------
# Step 16-17, 21: scoring + stage classification
# ---------------------------------------------------------------------------

def score_candidate(ind: dict[str, Any], fundamentals: dict[str, Any]) -> dict[str, Any]:
    r5, r1m, r3m, r6m = ind.get("return_5d"), ind.get("return_1m"), ind.get("return_3m"), ind.get("return_6m")

    price_reversal = _clip(50 + (25 if ind.get("_close_above_dma20") else -25) + _clip((r5 or 0) / 0.05, -1, 1) * 25)

    ma_flags = [ind.get("_close_above_dma20"), (ind.get("dma20_slope") or 0) > 0, ind.get("_dma20_above_dma50"), (ind.get("dma50_slope") or 0) > 0]
    ma_improvement = _clip(sum(1 for f in ma_flags if f) / len(ma_flags) * 100)

    momentum = _clip(50 + (r1m or 0) * 300 + (r3m or 0) * 100 + (r6m or 0) * 40)

    vr20, pud = ind.get("volume_ratio_20d"), ind.get("pct_up_volume_days")
    volume_accum = _clip(0.6 * _clip(50 + ((vr20 - 1) if vr20 is not None else 0) * 100) + 0.4 * _clip((pud if pud is not None else 0.5) * 100))

    rs1, rs3, rs6 = ind.get("rs_1m"), ind.get("rs_3m"), ind.get("rs_6m")
    relative_strength = _clip(50 + (rs1 or 0) * 220 + (rs3 or 0) * 140 + (rs6 or 0) * 60)

    perf = _clip(50 + (r1m or 0) * 200 + (r3m or 0) * 80 + (r6m or 0) * 35)
    dist_high = ind.get("distance_from_52w_high")
    proximity_52w = _clip((1 + dist_high) * 100) if dist_high is not None else 50.0
    close, dma200 = ind.get("close"), ind.get("dma200")
    proximity_200 = _clip(50 + ((close / dma200 - 1) if close is not None and dma200 else 0) * 200)
    ma_confirm_flags = [ind.get("_close_above_dma20"), ind.get("_close_above_dma50"), ind.get("_close_above_dma200"), ind.get("_dma20_above_dma50"), ind.get("_dma50_above_dma200")]
    ma_confirmation = _clip(sum(1 for f in ma_confirm_flags if f) / len(ma_confirm_flags) * 100)
    vol_expansion = _clip(50 + ((vr20 - 1) if vr20 is not None else 0) * 60)
    breakout_flag = 100.0 if (dist_high is not None and dist_high >= -0.02) else 0.0

    recognition_score = round(
        0.25 * perf + 0.20 * proximity_52w + 0.15 * proximity_200 + 0.15 * ma_confirmation
        + 0.10 * vol_expansion + 0.10 * relative_strength + 0.05 * breakout_flag,
        2,
    )
    recognition_class = "LOW" if recognition_score < 35 else ("HIGH" if recognition_score > 65 else "MODERATE")
    recognition_positioning = _clip(100 - recognition_score)

    technical_score = round(
        0.20 * price_reversal + 0.20 * ma_improvement + 0.15 * momentum + 0.15 * volume_accum
        + 0.15 * relative_strength + 0.15 * recognition_positioning,
        2,
    )

    fundamental_score = fundamentals.get("fundamental_score")
    early_opportunity_score = round(_clip(fundamental_score - recognition_score + 50), 2) if fundamental_score is not None else None

    stage = _classify_stage(technical_score, recognition_class, fundamentals)

    return {
        "technical_score": technical_score,
        "technical_score_components": {
            "price_reversal": round(price_reversal, 2), "ma_improvement": round(ma_improvement, 2),
            "momentum": round(momentum, 2), "volume_accum": round(volume_accum, 2),
            "relative_strength": round(relative_strength, 2), "recognition_positioning": round(recognition_positioning, 2),
        },
        "recognition_score": recognition_score,
        "recognition_class": recognition_class,
        "fundamental_score": fundamental_score,
        "early_opportunity_score": early_opportunity_score,
        "stage": stage,
        "fundamental_inflection": fundamentals.get("fundamental_inflection"),
        "earnings_acceleration": fundamentals.get("earnings_acceleration"),
        "false_turnaround_flags": fundamentals.get("false_turnaround_flags", {}),
        "turnaround_type": (ind.get("latest_crossover") or "n/a"),
    }


def _classify_stage(technical_score: float, recognition_class: str, fundamentals: dict[str, Any]) -> str:
    fa = fundamentals.get("fundamental_data_available")
    fscore = fundamentals.get("fundamental_score")
    flags = fundamentals.get("false_turnaround_flags", {})
    red_flags = sum(1 for v in flags.values() if v is True)

    if fa and fscore is not None and fscore < 25 and red_flags >= 2:
        return "FALSE / UNCONFIRMED"
    if technical_score >= 65 and fa and fscore is not None and fscore >= 50:
        return "CONFIRMED"
    if technical_score >= 60 and recognition_class in ("LOW", "MODERATE") and (not fa or (fscore is not None and fscore >= 40)):
        return "EARLY DISCOVERY"
    if technical_score >= 55 and recognition_class == "HIGH":
        return "MATURE / RECOGNIZED"
    return "DEVELOPING"


def _key_signal(ind: dict[str, Any], scores: dict[str, Any]) -> str:
    parts: list[str] = []
    if ind.get("latest_crossover"):
        parts.append(f"{ind['latest_crossover'].replace('_', ' ')} on {ind.get('crossover_date')}")
    vr20 = ind.get("volume_ratio_20d")
    if vr20 is not None and vr20 >= 1.5:
        parts.append(f"volume {vr20:.1f}x 20DMA avg")
    rs3 = ind.get("rs_3m")
    if rs3 is not None and abs(rs3) >= 0.02:
        parts.append(f"RS {rs3 * 100:+.1f}% vs Nifty 50 (3M)")
    if scores.get("earnings_acceleration") == "YES":
        parts.append("earnings accelerating")
    if not parts:
        parts.append(f"{scores.get('stage', 'candidate')} per {scores.get('turnaround_type') or 'technical'} setup")
    return "; ".join(parts[:3])


# ---------------------------------------------------------------------------
# Steps 12-14: scanner-mode rules
# ---------------------------------------------------------------------------

def _passes_early(ind: dict[str, Any]) -> bool:
    avg20, vol = ind.get("avg_volume_20d"), ind.get("volume")
    close, high52 = ind.get("close"), ind.get("high_52w")
    return all([
        ind.get("_close_above_dma20") is True,
        (ind.get("dma20_slope") if ind.get("dma20_slope") is not None else -1) > 0,
        ind.get("_close_above_dma50") is True,
        (ind.get("dma50_slope") if ind.get("dma50_slope") is not None else -1) > 0,
        (ind.get("_return_1d") if ind.get("_return_1d") is not None else -1) > 0,
        vol is not None and avg20 not in (None, 0) and vol > 1.20 * avg20,
        (ind.get("return_1m") if ind.get("return_1m") is not None else -1) > 0.03,
        (ind.get("return_6m") if ind.get("return_6m") is not None else 0) < 0.40,
        close is not None and high52 not in (None, 0) and close < 0.90 * high52,
    ])


def _passes_confirmed(ind: dict[str, Any]) -> bool:
    avg20, vol = ind.get("avg_volume_20d"), ind.get("volume")
    return all([
        ind.get("_close_above_dma20") is True,
        ind.get("_dma20_above_dma50") is True,
        ind.get("_dma50_above_dma200") is True,
        ind.get("_close_above_dma200") is True,
        (ind.get("_dma50_slope_20d") if ind.get("_dma50_slope_20d") is not None else -1) > 0,
        vol is not None and avg20 not in (None, 0) and vol > 1.50 * avg20,
        (ind.get("return_1m") if ind.get("return_1m") is not None else -1) > 0.05,
        (ind.get("return_3m") if ind.get("return_3m") is not None else -1) > 0.10,
    ])


def _passes_deep(ind: dict[str, Any]) -> bool:
    crossed_recently = bool(ind.get("_close_above_dma20")) or bool(ind.get("_close_crossed_20dma_recent")) or bool(ind.get("_close_crossed_50dma_recent"))
    return all([
        crossed_recently,
        (ind.get("dma20_slope") if ind.get("dma20_slope") is not None else 0) > -0.02,
        (ind.get("return_1m") if ind.get("return_1m") is not None else -1) > 0,
        (ind.get("volume_trend") if ind.get("volume_trend") is not None else 1.0) > 0.70,
    ])


_MODE_RULES = {"EARLY": _passes_early, "CONFIRMED": _passes_confirmed, "DEEP": _passes_deep}


def _passes_shortlist(mode: str, candidate: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    ts = candidate.get("technical_score") or 0
    if ts < thresholds.get("min_technical_score", 0):
        return False
    if mode == "EARLY" and candidate.get("recognition_class") not in thresholds.get("recognition_in", ("LOW", "MODERATE", "HIGH")):
        return False
    if mode == "CONFIRMED":
        if thresholds.get("require_ma_confirmation") and not (candidate.get("_dma20_above_dma50") and candidate.get("_dma50_above_dma200")):
            return False
        if thresholds.get("require_positive_rs_3m") and (candidate.get("rs_3m") or -1) <= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_scanner(
    *,
    mode: str,
    as_of_date: str,
    exchanges: list[str] | None = None,
    sector: str | None = None,
    industry: str | None = None,
    market_cap_min: float | None = None,
    market_cap_max: float | None = None,
    symbols: list[str] | None = None,
    active_only: bool = False,
    min_history_days: int = MIN_HISTORY_DAYS_DEFAULT,
    shortlist_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    scan_run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    as_of = pd.Timestamp(as_of_date).date()

    raw = get_scanner_price_frame(
        as_of_date=str(as_of), lookback_days=LOOKBACK_DAYS_DEFAULT,
        exchanges=exchanges, sector=sector, industry=industry,
        market_cap_min=market_cap_min, market_cap_max=market_cap_max,
        symbols=symbols, active_only=active_only,
    )
    universe_count = int(raw["ticker_id"].nunique()) if not raw.empty else 0
    clean, exclusions, quality_stats = prepare_universe_frame(raw)

    bench_df = get_scanner_benchmark_frame(as_of_date=str(as_of), lookback_days=LOOKBACK_DAYS_DEFAULT)
    benchmark_returns: dict[str, float | None] = {}
    if not bench_df.empty:
        bench_ind = add_indicators(bench_df)
        if not bench_ind.empty:
            benchmark_returns = {key: _safe_return(bench_ind["close"], periods) for key, periods in RETURN_PERIODS.items()}

    sector_index_cache: dict[str, dict[str, float | None] | None] = {}

    def _sector_returns_for(sector_name: str | None) -> dict[str, float | None] | None:
        idx_name = SECTOR_TO_INDEX_NAME.get(sector_name or "")
        if not idx_name:
            return None
        if idx_name not in sector_index_cache:
            idx_ticker = INDEX_TICKER_BY_ARCHIVE_NAME.get(idx_name)
            sdf = get_scanner_benchmark_frame(as_of_date=str(as_of), lookback_days=LOOKBACK_DAYS_DEFAULT, ticker=idx_ticker) if idx_ticker else pd.DataFrame()
            if sdf.empty:
                sector_index_cache[idx_name] = None
            else:
                sind = add_indicators(sdf)
                sector_index_cache[idx_name] = {key: _safe_return(sind["close"], periods) for key, periods in RETURN_PERIODS.items()} if not sind.empty else None
        return sector_index_cache[idx_name]

    rule_fn = _MODE_RULES[mode]
    thresholds = shortlist_thresholds or DEFAULT_SHORTLIST_THRESHOLDS[mode]

    candidates: list[dict[str, Any]] = []
    scanned = 0
    insufficient_history = 0

    if not clean.empty:
        for ticker_id, tdf in clean.groupby("ticker_id"):
            scanned += 1
            if len(tdf) < min_history_days:
                exclusions[ticker_id].append("insufficient_history")
                insufficient_history += 1
                continue
            symbol = tdf["symbol"].iloc[-1]
            ticker_str = tdf["ticker"].iloc[-1] if "ticker" in tdf.columns else None
            sector_name = tdf["sector"].iloc[-1]

            ind = compute_ticker_indicators(tdf, benchmark_returns, _sector_returns_for(sector_name))
            if ind is None:
                exclusions[ticker_id].append("insufficient_history")
                insufficient_history += 1
                continue
            if not rule_fn(ind):
                continue

            fundamentals = analyze_fundamentals(ticker_str, as_of)
            scores = score_candidate(ind, fundamentals)
            row = {
                "symbol": symbol,
                "company_name": tdf["company_name"].iloc[-1],
                "exchange": tdf["exchange"].iloc[-1],
                "isin": tdf["isin"].iloc[-1],
                "sector": sector_name,
                "industry": tdf["industry"].iloc[-1],
                "status": tdf["status"].iloc[-1],
                "as_of_date": str(as_of),
                "data_quality_flags": list(dict.fromkeys(exclusions.get(ticker_id, []) + fundamentals.get("data_quality_flags", []))),
                **{k: v for k, v in ind.items() if not k.startswith("_")},
                **{k: v for k, v in ind.items() if k.startswith("_")},  # keep internal flags for shortlist filtering
                **scores,
            }
            row["key_signal"] = _key_signal(ind, scores)
            row["adjusted_close"] = None  # not available anywhere in this database
            candidates.append(row)

    shortlist = [c for c in candidates if _passes_shortlist(mode, c, thresholds)]

    def sort_key(c: dict[str, Any]) -> tuple:
        stage_rank = STAGE_ORDER.get(c.get("stage"), 99)
        eo = c.get("early_opportunity_score")
        ts = c.get("technical_score") or 0.0
        return (stage_rank, -(eo if eo is not None else -1), -ts)

    candidates.sort(key=sort_key)
    shortlist.sort(key=sort_key)

    params = {
        "mode": mode, "as_of_date": str(as_of), "min_history_days": min_history_days,
        "lookback_days": LOOKBACK_DAYS_DEFAULT, "shortlist_thresholds": thresholds,
        "universe_filters": {
            "exchanges": exchanges, "sector": sector, "industry": industry,
            "market_cap_min": market_cap_min, "market_cap_max": market_cap_max,
            "symbols": symbols, "active_only": active_only,
        },
    }

    return {
        "scan_run_id": scan_run_id,
        "generated_at": generated_at,
        "as_of_date": str(as_of),
        "mode": mode,
        "params": params,
        "universe_count": universe_count,
        "scanned_count": scanned,
        "excluded_count": len(exclusions),
        "insufficient_history_count": insufficient_history,
        "invalid_ohlcv_rows_dropped": quality_stats["invalid_rows_dropped"],
        "duplicate_listings_dropped": quality_stats["duplicate_listings_dropped"],
        "benchmark_available": bool(benchmark_returns),
        "candidates": candidates,
        "shortlist": shortlist,
    }

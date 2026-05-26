from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from stock_advisor.analysis.chart_patterns import detect_chart_patterns
from stock_advisor.analysis.indicators import latest_indicators
from stock_advisor.data.market_data import get_basic_fundamentals, get_price_history
from stock_advisor.data.nse_indices import get_nse_index_price_history


BENCHMARK_TICKER = "^NSEI"

SECTOR_DEFINITIONS: dict[str, dict[str, Any]] = {
    "auto": {
        "name": "Nifty Auto",
        "index_ticker": "^CNXAUTO",
        "sector": "Auto",
        "stocks": (
            "MARUTI.NS",
            "M&M.NS",
            "TMPV.NS",
            "BAJAJ-AUTO.NS",
            "EICHERMOT.NS",
            "TVSMOTOR.NS",
            "HEROMOTOCO.NS",
            "ASHOKLEY.NS",
            "MOTHERSON.NS",
            "BOSCHLTD.NS",
        ),
    },
    "bank": {
        "name": "Nifty Bank",
        "index_ticker": "^NSEBANK",
        "sector": "Banking",
        "stocks": (
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "SBIN.NS",
            "AXISBANK.NS",
            "KOTAKBANK.NS",
            "INDUSINDBK.NS",
            "BANKBARODA.NS",
            "PNB.NS",
            "CANBK.NS",
            "IDFCFIRSTB.NS",
        ),
    },
    "fmcg": {
        "name": "Nifty FMCG",
        "index_ticker": "^CNXFMCG",
        "sector": "FMCG",
        "stocks": (
            "HINDUNILVR.NS",
            "ITC.NS",
            "NESTLEIND.NS",
            "BRITANNIA.NS",
            "DABUR.NS",
            "MARICO.NS",
            "GODREJCP.NS",
            "TATACONSUM.NS",
            "VBL.NS",
            "COLPAL.NS",
        ),
    },
    "it": {
        "name": "Nifty IT",
        "index_ticker": "^CNXIT",
        "sector": "Information Technology",
        "stocks": (
            "TCS.NS",
            "INFY.NS",
            "HCLTECH.NS",
            "WIPRO.NS",
            "TECHM.NS",
            "LTM.NS",
            "PERSISTENT.NS",
            "COFORGE.NS",
            "MPHASIS.NS",
            "LTTS.NS",
        ),
    },
    "metal": {
        "name": "Nifty Metals & Mining",
        "index_ticker": "^CNXMETAL",
        "sector": "Metals",
        "stocks": (
            "TATASTEEL.NS",
            "JSWSTEEL.NS",
            "HINDALCO.NS",
            "VEDL.NS",
            "JINDALSTEL.NS",
            "SAIL.NS",
            "NMDC.NS",
            "NATIONALUM.NS",
            "HINDZINC.NS",
            "COALINDIA.NS",
        ),
    },
    "pharma": {
        "name": "Nifty Pharma",
        "index_ticker": "^CNXPHARMA",
        "sector": "Pharma",
        "stocks": (
            "SUNPHARMA.NS",
            "CIPLA.NS",
            "DRREDDY.NS",
            "DIVISLAB.NS",
            "LUPIN.NS",
            "AUROPHARMA.NS",
            "MANKIND.NS",
            "ZYDUSLIFE.NS",
            "GLENMARK.NS",
            "BIOCON.NS",
        ),
    },
    "psu_bank": {
        "name": "Nifty PSU Bank",
        "index_ticker": "^CNXPSUBANK",
        "sector": "PSU Banking",
        "stocks": (
            "SBIN.NS",
            "BANKBARODA.NS",
            "PNB.NS",
            "CANBK.NS",
            "UNIONBANK.NS",
            "INDIANB.NS",
            "BANKINDIA.NS",
            "MAHABANK.NS",
            "CENTRALBK.NS",
            "UCOBANK.NS",
        ),
    },
    "realty": {
        "name": "Nifty Realty",
        "index_ticker": "^CNXREALTY",
        "sector": "Real Estate",
        "stocks": (
            "DLF.NS",
            "GODREJPROP.NS",
            "LODHA.NS",
            "OBEROIRLTY.NS",
            "PHOENIXLTD.NS",
            "PRESTIGE.NS",
            "BRIGADE.NS",
            "SOBHA.NS",
            "ANANTRAJ.NS",
            "SIGNATURE.NS",
        ),
    },
    "energy": {
        "name": "Nifty Energy",
        "index_ticker": "^CNXENERGY",
        "sector": "Energy",
        "stocks": (
            "RELIANCE.NS",
            "ONGC.NS",
            "NTPC.NS",
            "POWERGRID.NS",
            "COALINDIA.NS",
            "BPCL.NS",
            "IOC.NS",
            "GAIL.NS",
            "TATAPOWER.NS",
            "ADANIGREEN.NS",
        ),
    },
    "infra": {
        "name": "Nifty Infrastructure",
        "index_ticker": "^CNXINFRA",
        "sector": "Infrastructure",
        "stocks": (
            "LT.NS",
            "NCC.NS",
            "IRB.NS",
            "RVNL.NS",
            "IRCON.NS",
            "KPIL.NS",
            "KEC.NS",
            "PNCINFRA.NS",
            "KNRCON.NS",
            "HGINFRA.NS",
        ),
    },
    "media": {
        "name": "Nifty Media",
        "index_ticker": "^CNXMEDIA",
        "sector": "Media",
        "stocks": (
            "SUNTV.NS",
            "ZEEL.NS",
            "PVRINOX.NS",
            "NETWORK18.NS",
            "SAREGAMA.NS",
            "NAZARA.NS",
        ),
    },
    "consumption": {
        "name": "Nifty Consumption",
        "index_ticker": "CONSUMBEES.NS",
        "index_tickers": ("CONSUMBEES.NS",),
        "sector": "Consumption",
        "stocks": (
            "TITAN.NS",
            "TRENT.NS",
            "DMART.NS",
            "VBL.NS",
            "INDHOTEL.NS",
            "JUBLFOOD.NS",
            "KALYANKJIL.NS",
            "ETERNAL.NS",
            "NYKAA.NS",
            "BATAINDIA.NS",
        ),
    },
    "oil_gas": {
        "name": "Nifty Oil & Gas",
        "index_ticker": "OILIETF.NS",
        "index_tickers": ("OILIETF.NS",),
        "sector": "Oil & Gas",
        "stocks": (
            "RELIANCE.NS",
            "ONGC.NS",
            "OIL.NS",
            "GAIL.NS",
            "PETRONET.NS",
            "MGL.NS",
            "IGL.NS",
            "GUJGASLTD.NS",
            "ATGL.NS",
            "CASTROLIND.NS",
        ),
    },
}

SECTOR_ALIASES = {
    "nifty auto": "auto",
    "nifty bank": "bank",
    "banking": "bank",
    "nifty fmcg": "fmcg",
    "nifty it": "it",
    "technology": "it",
    "nifty metal": "metal",
    "metals": "metal",
    "nifty pharma": "pharma",
    "healthcare": "pharma",
    "nifty psu bank": "psu_bank",
    "nifty realty": "realty",
    "real estate": "realty",
    "nifty energy": "energy",
    "nifty infra": "infra",
    "nifty infrastructure": "infra",
    "nifty media": "media",
    "consumption": "consumption",
    "nifty consumption": "consumption",
    "oil and gas": "oil_gas",
    "oil & gas": "oil_gas",
}


def list_sector_definitions() -> dict[str, Any]:
    """Return configured sector indices and their stock universes."""
    return {
        sector_id: {
            "id": sector_id,
            "name": definition["name"],
            "index_ticker": definition["index_ticker"],
            "index_tickers": list(_index_tickers(definition)),
            "sector": definition["sector"],
            "stocks": list(definition["stocks"]),
        }
        for sector_id, definition in SECTOR_DEFINITIONS.items()
    }


def get_sector_rotation(
    *,
    period: str = "1y",
    interval: str = "1d",
    max_sectors: int | None = None,
    include_breadth: bool = True,
    max_breadth_stocks: int = 8,
    benchmark_ticker: str = BENCHMARK_TICKER,
) -> dict[str, Any]:
    """Rank configured sector indices by relative strength, trend, acceleration, and breadth."""
    benchmark_prices = get_price_history(benchmark_ticker, period=period, interval=interval)
    benchmark_metrics = _price_metrics(benchmark_prices)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for sector_id, definition in SECTOR_DEFINITIONS.items():
        prices = _get_index_price_history(definition, period=period, interval=interval)
        metrics = _price_metrics(prices)
        if not metrics:
            warnings.append(
                f"No index price history available for {definition['name']} ({', '.join(_index_tickers(definition))})."
            )
            continue

        breadth = (
            _sector_breadth(definition["stocks"], period=period, interval=interval, max_stocks=max_breadth_stocks)
            if include_breadth
            else _empty_breadth()
        )
        scored = _score_sector(sector_id, definition, metrics, benchmark_metrics, breadth)
        rows.append(scored)

    ranked = sorted(rows, key=lambda row: row["rotation_score"], reverse=True)
    if max_sectors is not None:
        ranked = ranked[: max(0, int(max_sectors))]

    return _json_safe(
        {
            "benchmark": {
                "ticker": benchmark_ticker,
                "name": "Nifty 50",
                "metrics": benchmark_metrics,
            },
            "period": period,
            "interval": interval,
            "sectors": ranked,
            "top_sector": ranked[0] if ranked else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "methodology": {
                "rotation_score": "35% relative strength, 20% trend, 15% momentum acceleration, 15% breadth, 10% risk-adjusted extension, 5% liquidity/data quality.",
                "stage": "Emerging, Leadership, Pullback in uptrend, Exhausted, Weak/Avoid, or Neutral based on trend and relative strength.",
            },
        }
    )


def rank_sector_stocks(
    sector: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    max_stocks: int = 10,
    include_fundamentals: bool = True,
) -> dict[str, Any]:
    """Rank stock candidates inside one sector by relative strength and setup quality."""
    sector_id, definition = _resolve_sector(sector)
    sector_prices = _get_index_price_history(definition, period=period, interval=interval)
    sector_metrics = _price_metrics(sector_prices)
    rows = []
    warnings = []

    for ticker in list(definition["stocks"])[: max(1, int(max_stocks))]:
        prices = get_price_history(ticker, period=period, interval=interval)
        metrics = _price_metrics(prices)
        if not metrics:
            warnings.append(f"No price history available for {ticker}.")
            continue
        indicators = latest_indicators(prices)
        patterns = detect_chart_patterns(prices)
        fundamentals = get_basic_fundamentals(ticker) if include_fundamentals else {}
        rows.append(_score_sector_stock(ticker, metrics, sector_metrics, indicators, patterns, fundamentals))

    ranked = sorted(rows, key=lambda row: row["stock_score"], reverse=True)
    return _json_safe(
        {
            "sector_id": sector_id,
            "sector_name": definition["name"],
            "sector_index_ticker": sector_metrics.get("selected_ticker") or definition["index_ticker"],
            "configured_index_tickers": list(_index_tickers(definition)),
            "sector_metrics": sector_metrics,
            "period": period,
            "interval": interval,
            "stocks": ranked,
            "top_stock": ranked[0] if ranked else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        }
    )


def discover_sector_opportunities(
    *,
    period: str = "1y",
    interval: str = "1d",
    top_sectors: int = 3,
    stocks_per_sector: int = 5,
    include_fundamentals: bool = True,
) -> dict[str, Any]:
    """Find leading/emerging sectors and rank the best stock setups inside them."""
    rotation = get_sector_rotation(
        period=period,
        interval=interval,
        max_sectors=max(1, int(top_sectors)),
        include_breadth=True,
        max_breadth_stocks=max(5, int(stocks_per_sector)),
    )
    opportunities = []
    for sector in rotation.get("sectors", [])[: max(1, int(top_sectors))]:
        ranked = rank_sector_stocks(
            sector["sector_id"],
            period=period,
            interval=interval,
            max_stocks=max(1, int(stocks_per_sector)),
            include_fundamentals=include_fundamentals,
        )
        opportunities.append(
            {
                "sector": sector,
                "stocks": ranked.get("stocks", []),
                "top_stock": ranked.get("top_stock"),
                "warnings": ranked.get("warnings", []),
            }
        )

    return _json_safe(
        {
            "period": period,
            "interval": interval,
            "top_sectors": rotation.get("sectors", []),
            "opportunities": opportunities,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": rotation.get("warnings", []),
        }
    )


def _resolve_sector(sector: str) -> tuple[str, dict[str, Any]]:
    key = (sector or "").strip().lower().replace("-", "_").replace(" ", "_")
    alias_key = (sector or "").strip().lower()
    sector_id = SECTOR_ALIASES.get(alias_key) or key
    if sector_id not in SECTOR_DEFINITIONS:
        expected = ", ".join(sorted(SECTOR_DEFINITIONS))
        raise ValueError(f"Unknown sector: {sector}. Expected one of: {expected}")
    return sector_id, SECTOR_DEFINITIONS[sector_id]


def _index_tickers(definition: dict[str, Any]) -> tuple[str, ...]:
    tickers = definition.get("index_tickers") or (definition["index_ticker"],)
    return tuple(str(ticker) for ticker in tickers if ticker)


def _get_index_price_history(definition: dict[str, Any], *, period: str, interval: str) -> pd.DataFrame:
    index_history = get_nse_index_price_history(str(definition.get("nse_index_name") or definition.get("name") or ""), period=period, interval=interval)
    if not index_history.empty:
        return index_history
    for ticker in _index_tickers(definition):
        prices = get_price_history(ticker, period=period, interval=interval)
        if not prices.empty:
            prices = prices.copy()
            prices.attrs.update(prices.attrs)
            prices.attrs["selected_ticker"] = ticker
            return prices
    return pd.DataFrame()


def _score_sector(
    sector_id: str,
    definition: dict[str, Any],
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
    breadth: dict[str, Any],
) -> dict[str, Any]:
    rel_20d = _number(metrics.get("return_20d")) - _number(benchmark.get("return_20d"))
    rel_60d = _number(metrics.get("return_60d")) - _number(benchmark.get("return_60d"))
    rel_120d = _number(metrics.get("return_120d")) - _number(benchmark.get("return_120d"))
    relative_score = _score_relative_strength(rel_20d, rel_60d, rel_120d)
    trend_score = _score_trend(metrics)
    acceleration_score = _score_acceleration(metrics, benchmark)
    breadth_score = _number(breadth.get("breadth_score"), 50)
    extension_score = _score_extension(metrics)
    quality_score = 70 if metrics.get("data_points", 0) >= 120 else 45
    rotation_score = round(
        0.35 * relative_score
        + 0.20 * trend_score
        + 0.15 * acceleration_score
        + 0.15 * breadth_score
        + 0.10 * extension_score
        + 0.05 * quality_score,
        2,
    )
    stage = _sector_stage(metrics, rel_20d, rel_60d, breadth_score)

    return {
        "sector_id": sector_id,
        "name": definition["name"],
        "index_ticker": metrics.get("selected_ticker") or definition["index_ticker"],
        "configured_index_tickers": list(_index_tickers(definition)),
        "sector": definition["sector"],
        "rotation_score": rotation_score,
        "stage": stage,
        "relative_strength": {
            "vs_benchmark_20d": round(rel_20d, 4),
            "vs_benchmark_60d": round(rel_60d, 4),
            "vs_benchmark_120d": round(rel_120d, 4),
            "score": relative_score,
        },
        "trend_score": trend_score,
        "acceleration_score": acceleration_score,
        "extension_score": extension_score,
        "breadth": breadth,
        "metrics": metrics,
        "candidate_count": len(definition["stocks"]),
        "candidate_stocks": list(definition["stocks"]),
        "notes": _sector_notes(stage, rel_20d, rel_60d, breadth_score, metrics),
    }


def _score_sector_stock(
    ticker: str,
    metrics: dict[str, Any],
    sector_metrics: dict[str, Any],
    indicators: dict[str, Any],
    patterns: dict[str, Any],
    fundamentals: dict[str, Any],
) -> dict[str, Any]:
    rel_20d = _number(metrics.get("return_20d")) - _number(sector_metrics.get("return_20d"))
    rel_60d = _number(metrics.get("return_60d")) - _number(sector_metrics.get("return_60d"))
    rs_score = _score_relative_strength(rel_20d, rel_60d, 0)
    trend_score = _score_trend(metrics)
    pattern_score = _number(patterns.get("pattern_score"), 50)
    volume_score = _score_volume(indicators)
    fundamental_score = _stock_fundamental_quality(fundamentals)
    risk_score = _stock_risk_quality(indicators, metrics)
    stock_score = round(
        0.30 * rs_score
        + 0.20 * trend_score
        + 0.15 * pattern_score
        + 0.15 * volume_score
        + 0.10 * fundamental_score
        + 0.10 * risk_score,
        2,
    )
    return {
        "ticker": ticker,
        "stock_score": stock_score,
        "stage": _stock_stage(metrics, rel_20d, trend_score, pattern_score),
        "relative_strength": {
            "vs_sector_20d": round(rel_20d, 4),
            "vs_sector_60d": round(rel_60d, 4),
            "score": rs_score,
        },
        "trend_score": trend_score,
        "pattern_score": pattern_score,
        "volume_score": volume_score,
        "fundamental_quality_score": fundamental_score,
        "risk_quality_score": risk_score,
        "metrics": metrics,
        "dominant_chart_pattern": (patterns.get("dominant_pattern") or {}).get("pattern"),
        "fundamentals": {
            "name": fundamentals.get("shortName"),
            "sector": fundamentals.get("sector"),
            "industry": fundamentals.get("industry"),
            "trailingPE": fundamentals.get("trailingPE"),
            "forwardPE": fundamentals.get("forwardPE"),
            "debtToEquity": fundamentals.get("debtToEquity"),
            "profitMargins": fundamentals.get("profitMargins"),
            "revenueGrowth": fundamentals.get("revenueGrowth"),
            "earningsGrowth": fundamentals.get("earningsGrowth"),
            "providers": fundamentals.get("_sources", []),
        },
        "reasons": _stock_reasons(metrics, rel_20d, trend_score, pattern_score, indicators),
    }


def _sector_breadth(
    tickers: tuple[str, ...],
    *,
    period: str,
    interval: str,
    max_stocks: int,
) -> dict[str, Any]:
    rows = []
    for ticker in list(tickers)[: max(1, int(max_stocks))]:
        metrics = _price_metrics(get_price_history(ticker, period=period, interval=interval))
        if metrics:
            rows.append(metrics)

    if not rows:
        return _empty_breadth()

    above_20 = _percent(row.get("above_sma_20") for row in rows)
    above_50 = _percent(row.get("above_sma_50") for row in rows)
    above_200 = _percent(row.get("above_sma_200") for row in rows)
    positive_20d = _percent(_number(row.get("return_20d")) > 0 for row in rows)
    avg_return_20d = sum(_number(row.get("return_20d")) for row in rows) / len(rows)
    breadth_score = round(0.25 * above_20 + 0.30 * above_50 + 0.25 * above_200 + 0.20 * positive_20d, 2)
    return {
        "sample_size": len(rows),
        "above_sma_20_percent": round(above_20, 2),
        "above_sma_50_percent": round(above_50, 2),
        "above_sma_200_percent": round(above_200, 2),
        "positive_20d_percent": round(positive_20d, 2),
        "avg_return_20d": round(avg_return_20d, 4),
        "breadth_score": breadth_score,
    }


def _empty_breadth() -> dict[str, Any]:
    return {
        "sample_size": 0,
        "above_sma_20_percent": None,
        "above_sma_50_percent": None,
        "above_sma_200_percent": None,
        "positive_20d_percent": None,
        "avg_return_20d": None,
        "breadth_score": 50,
    }


def _price_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty or "close" not in df.columns:
        return {}
    clean = df.dropna(subset=["close"]).copy()
    if clean.empty:
        return {}

    close = pd.to_numeric(clean["close"], errors="coerce").dropna()
    if close.empty:
        return {}

    indicators = latest_indicators(clean)
    latest_close = float(close.iloc[-1])
    high_52w = _number(indicators.get("high_52w"))
    low_52w = _number(indicators.get("low_52w"))
    latest_date = None
    if "date" in clean.columns:
        dates = pd.to_datetime(clean["date"], errors="coerce").dropna()
        if not dates.empty:
            latest_date = dates.iloc[-1].date().isoformat()
    return {
        "close": latest_close,
        "latest_date": latest_date,
        "data_points": int(len(close)),
        "return_1d": _return(close, 1),
        "return_5d": _return(close, 5),
        "return_20d": _return(close, 20),
        "return_60d": _return(close, 60),
        "return_120d": _return(close, 120),
        "return_252d": _return(close, 252),
        "sma_20": indicators.get("sma_20"),
        "sma_50": indicators.get("sma_50"),
        "sma_100": indicators.get("sma_100"),
        "sma_200": indicators.get("sma_200"),
        "ema_20": indicators.get("ema_20"),
        "ema_21": indicators.get("ema_21"),
        "ema_50": indicators.get("ema_50"),
        "ema_100": indicators.get("ema_100"),
        "ema_200": indicators.get("ema_200"),
        "above_sma_20": _above(latest_close, indicators.get("sma_20")),
        "above_sma_50": _above(latest_close, indicators.get("sma_50")),
        "above_sma_100": _above(latest_close, indicators.get("sma_100")),
        "above_sma_200": _above(latest_close, indicators.get("sma_200")),
        "above_ema_20": _above(latest_close, indicators.get("ema_20")),
        "above_ema_21": _above(latest_close, indicators.get("ema_21")),
        "above_ema_50": _above(latest_close, indicators.get("ema_50")),
        "above_ema_100": _above(latest_close, indicators.get("ema_100")),
        "above_ema_200": _above(latest_close, indicators.get("ema_200")),
        "rsi_14": indicators.get("rsi_14"),
        "adx_14": indicators.get("adx_14"),
        "atr_percent_14": indicators.get("atr_percent_14"),
        "volume_ratio": indicators.get("volume_ratio"),
        "high_52w": high_52w or None,
        "low_52w": low_52w or None,
        "distance_from_52w_high": indicators.get("distance_from_52w_high"),
        "distance_from_52w_low": indicators.get("distance_from_52w_low"),
        "max_drawdown": indicators.get("max_drawdown"),
        "volatility_20d": indicators.get("volatility_20d"),
        "liquidity_score": indicators.get("liquidity_score"),
        "provider": clean.attrs.get("provider"),
        "selected_ticker": clean.attrs.get("selected_ticker"),
    }


def _return(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    start = float(close.iloc[-periods - 1])
    end = float(close.iloc[-1])
    if start <= 0:
        return None
    return round(end / start - 1, 4)


def _above(close: float, average: Any) -> bool | None:
    avg = _number(average, None)
    if avg is None or avg <= 0:
        return None
    return close > avg


def _score_relative_strength(rel20: float, rel60: float, rel120: float) -> float:
    score = 50 + rel20 * 220 + rel60 * 140 + rel120 * 60
    return round(max(0, min(100, score)), 2)


def _score_trend(metrics: dict[str, Any]) -> float:
    score = 35
    weights = {
        "above_sma_20": 15,
        "above_sma_50": 20,
        "above_sma_100": 15,
        "above_sma_200": 20,
    }
    for key, weight in weights.items():
        if metrics.get(key) is True:
            score += weight
        elif metrics.get(key) is False:
            score -= weight * 0.35
    adx = _number(metrics.get("adx_14"), None)
    if adx is not None and adx >= 25:
        score += 8
    rsi = _number(metrics.get("rsi_14"), None)
    if rsi is not None:
        if 45 <= rsi <= 65:
            score += 5
        elif rsi > 75:
            score -= 8
        elif rsi < 30:
            score -= 4
    return round(max(0, min(100, score)), 2)


def _score_acceleration(metrics: dict[str, Any], benchmark: dict[str, Any]) -> float:
    ret5 = _number(metrics.get("return_5d"))
    ret20 = _number(metrics.get("return_20d"))
    bench5 = _number(benchmark.get("return_5d"))
    bench20 = _number(benchmark.get("return_20d"))
    acceleration = (ret5 - ret20 / 4) - (bench5 - bench20 / 4)
    score = 50 + acceleration * 500
    return round(max(0, min(100, score)), 2)


def _score_extension(metrics: dict[str, Any]) -> float:
    rsi = _number(metrics.get("rsi_14"), 50)
    distance_high = _number(metrics.get("distance_from_52w_high"))
    drawdown = abs(_number(metrics.get("max_drawdown")))
    score = 70
    if rsi > 75:
        score -= 18
    elif rsi > 68:
        score -= 8
    elif 40 <= rsi <= 65:
        score += 8
    if distance_high > -0.03:
        score -= 5
    if drawdown > 0.35:
        score -= 10
    return round(max(0, min(100, score)), 2)


def _score_volume(indicators: dict[str, Any]) -> float:
    ratio = _number(indicators.get("volume_ratio"), None)
    if ratio is None:
        return 50
    if ratio >= 1.8:
        return 86
    if ratio >= 1.25:
        return 72
    if ratio >= 0.9:
        return 58
    return 42


def _stock_fundamental_quality(fundamentals: dict[str, Any]) -> float:
    if not fundamentals:
        return 50
    score = 50
    pe = _number(fundamentals.get("forwardPE") or fundamentals.get("trailingPE"), None)
    debt = _number(fundamentals.get("debtToEquity"), None)
    margins = _number(fundamentals.get("profitMargins"), None)
    growth = _number(fundamentals.get("revenueGrowth"), None)
    earnings_growth = _number(fundamentals.get("earningsGrowth"), None)
    if pe is not None:
        if 0 < pe <= 35:
            score += 10
        elif pe > 70:
            score -= 10
    if debt is not None:
        if debt < 80:
            score += 8
        elif debt > 180:
            score -= 8
    if margins is not None and margins > 0.08:
        score += 8
    if growth is not None and growth > 0:
        score += 8
    if earnings_growth is not None:
        score += 6 if earnings_growth > 0 else -5
    return round(max(0, min(100, score)), 2)


def _stock_risk_quality(indicators: dict[str, Any], metrics: dict[str, Any]) -> float:
    score = 70
    volatility = _number(metrics.get("volatility_20d"), None)
    drawdown = abs(_number(metrics.get("max_drawdown")))
    atr_percent = _number(metrics.get("atr_percent_14"), None)
    liquidity = _number(indicators.get("liquidity_score"), None)
    if volatility is not None and volatility > 0.55:
        score -= 12
    if drawdown > 0.40:
        score -= 12
    if atr_percent is not None and atr_percent > 0.055:
        score -= 8
    if liquidity is not None and liquidity < 45:
        score -= 10
    return round(max(0, min(100, score)), 2)


def _sector_stage(metrics: dict[str, Any], rel20: float, rel60: float, breadth_score: float) -> str:
    rsi = _number(metrics.get("rsi_14"), 50)
    if rel20 > 0.015 and rel60 > 0.015 and metrics.get("above_sma_50") and metrics.get("above_sma_200"):
        return "Exhausted leadership" if rsi > 72 else "Leadership"
    if rel20 > 0.01 and rel60 <= 0.01 and metrics.get("above_sma_20") and breadth_score >= 50:
        return "Emerging"
    if rel60 > 0.01 and _number(metrics.get("return_5d")) < 0 and metrics.get("above_sma_200"):
        return "Pullback in uptrend"
    if rel20 < -0.015 and rel60 < -0.015 and not metrics.get("above_sma_50"):
        return "Weak/Avoid"
    return "Neutral"


def _stock_stage(metrics: dict[str, Any], rel20: float, trend_score: float, pattern_score: float) -> str:
    if rel20 > 0.02 and trend_score >= 70 and pattern_score >= 58:
        return "Sector leader"
    if rel20 > 0 and trend_score >= 58:
        return "Improving"
    if rel20 < -0.02 and trend_score < 45:
        return "Laggard"
    return "Watch"


def _sector_notes(stage: str, rel20: float, rel60: float, breadth_score: float, metrics: dict[str, Any]) -> list[str]:
    notes = [f"Stage: {stage}."]
    if rel20 > 0:
        notes.append("20-day relative strength is positive versus Nifty 50.")
    elif rel20 < 0:
        notes.append("20-day relative strength is negative versus Nifty 50.")
    if rel60 > 0:
        notes.append("60-day relative strength is positive versus Nifty 50.")
    if breadth_score >= 65:
        notes.append("Sector breadth is broad; multiple constituents are participating.")
    elif breadth_score < 45:
        notes.append("Sector breadth is weak; leadership may be narrow.")
    if metrics.get("above_sma_200"):
        notes.append("Sector index is above its 200-day average.")
    else:
        notes.append("Sector index is below or missing its 200-day average.")
    return notes


def _stock_reasons(
    metrics: dict[str, Any],
    rel20: float,
    trend_score: float,
    pattern_score: float,
    indicators: dict[str, Any],
) -> list[str]:
    reasons = []
    if rel20 > 0:
        reasons.append("Outperforming its sector over 20 days.")
    if metrics.get("above_sma_50"):
        reasons.append("Price is above the 50-day average.")
    if metrics.get("above_sma_200"):
        reasons.append("Price is above the 200-day average.")
    if pattern_score >= 60:
        reasons.append("Chart-pattern bias is constructive.")
    if _number(indicators.get("volume_ratio"), 0) > 1.2:
        reasons.append("Volume is above recent average.")
    if trend_score < 45:
        reasons.append("Trend is still weak despite sector context.")
    return reasons or ["Setup is neutral; use as a research candidate."]


def _percent(values: Any) -> float:
    rows = [value for value in values if value is not None]
    if not rows:
        return 0.0
    return 100 * sum(bool(value) for value in rows) / len(rows)


def _number(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None if default is None else float(default)
        return float(value)
    except (TypeError, ValueError):
        return None if default is None else float(default)


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
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, float):
        return None if not np.isfinite(value) else round(value, 6)
    return value

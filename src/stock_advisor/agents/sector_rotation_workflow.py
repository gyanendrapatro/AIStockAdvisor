from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from stock_advisor.analysis.sector_rotation import get_sector_rotation, rank_sector_stocks


def run_sector_rotation_workflow(
    *,
    period: str = "1y",
    interval: str = "1d",
    auto_period: bool = False,
    max_sectors: int = 10,
    max_breadth_stocks: int = 6,
    stocks_per_sector: int = 6,
    include_fundamentals: bool = True,
    selected_sector: str | None = None,
) -> dict[str, Any]:
    """Run a sector rotation workflow from the latest available market data."""
    started_at = datetime.now(timezone.utc).isoformat()
    requested_period = period
    auto_period = auto_period or str(period).strip().lower() == "auto"
    analysis_period = "1y" if auto_period else period
    rotation = get_sector_rotation(
        period=analysis_period,
        interval=interval,
        max_sectors=max_sectors,
        include_breadth=True,
        max_breadth_stocks=max_breadth_stocks,
    )
    sector_id = selected_sector or (rotation.get("top_sector") or {}).get("sector_id")
    ranked_stocks = (
        rank_sector_stocks(
            sector_id,
            period=analysis_period,
            interval=interval,
            max_stocks=stocks_per_sector,
            include_fundamentals=include_fundamentals,
        )
        if sector_id
        else {}
    )
    decision_summary = _decision_summary(rotation, ranked_stocks)

    return {
        "workflow": "sector_rotation_workflow",
        "mode": "latest_available_market_data",
        "cache_used": True,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "period": analysis_period,
            "requested_period": requested_period,
            "analysis_period": analysis_period,
            "auto_period": auto_period,
            "movement_windows": ["5D", "20D", "60D", "120D"],
            "interval": interval,
            "max_sectors": max_sectors,
            "max_breadth_stocks": max_breadth_stocks,
            "stocks_per_sector": stocks_per_sector,
            "include_fundamentals": include_fundamentals,
            "selected_sector": selected_sector,
        },
        "mcp_equivalent_tools": [
            "get_sector_rotation",
            "rank_sector_stocks",
        ],
        "steps": [
            "Loaded latest available benchmark and sector index/ETF price history from cache/provider fallback.",
            *(
                [
                    "Auto period selected: used a balanced 1y history while comparing 5D, 20D, 60D, and 120D movement windows."
                ]
                if auto_period
                else []
            ),
            "Calculated sector relative strength, trend, acceleration, breadth, and extension scores.",
            "Ranked stock candidates inside the selected/top sector using relative strength, trend, chart patterns, volume, fundamentals, and risk.",
        ],
        "decision_method": {
            "auto_period": auto_period,
            "analysis_period": analysis_period,
            "primary_signals": ["20D RS vs Nifty", "60D RS vs Nifty", "trend score", "acceleration score", "breadth score"],
            "short_term_confirmation": "5D return and 20D acceleration show whether buying interest is increasing now.",
            "summary": "Sector status is recalculated on each run from latest available local/provider data and classified from movement, relative strength, trend, acceleration, and breadth.",
        },
        "indicator_explanations": indicator_explanations(),
        "decision_summary": decision_summary,
        "rotation": rotation,
        "ranked_stocks": ranked_stocks,
        "warnings": [
            *rotation.get("warnings", []),
            *ranked_stocks.get("warnings", []),
        ],
    }


def indicator_explanations() -> dict[str, Any]:
    """Explain sector rotation filters, scores, and UI table columns."""
    return {
        "sector_filters": [
            {
                "name": "Sector period",
                "meaning": "Historical window used to calculate sector returns, moving averages, RSI, drawdown, and relative strength. Auto uses a balanced 1y history and compares 5D, 20D, 60D, and 120D movement windows.",
                "how_to_read": "Use Auto for normal sector-rotation decisions, 6mo for faster rotation, 1y for balanced context, and 2y for slower cycle confirmation.",
            },
            {
                "name": "Sector interval",
                "meaning": "Price candle interval used for all calculations.",
                "how_to_read": "Daily candles show current movement earlier; weekly candles smooth noise and are better for slower position-style sector trends.",
            },
            {
                "name": "Sectors",
                "meaning": "Maximum number of ranked sectors shown in the dashboard.",
                "how_to_read": "Higher values show more sectors but make charts denser.",
            },
            {
                "name": "Breadth sample",
                "meaning": "Number of constituent stocks sampled inside each sector to estimate participation.",
                "how_to_read": "Higher sample is more reliable but slower because it fetches more stock histories.",
            },
            {
                "name": "Stocks per sector",
                "meaning": "Number of candidate stocks ranked inside the selected/top sector.",
                "how_to_read": "Use a smaller value for speed and a larger value for wider opportunity discovery.",
            },
            {
                "name": "Stock fundamentals",
                "meaning": "Whether stock candidate scoring includes Yahoo Finance fundamentals.",
                "how_to_read": "Turn on for quality filtering; turn off for faster pure price-action scans.",
            },
        ],
        "sector_columns": [
            {"column": "rotation_score", "meaning": "Composite sector score from relative strength, trend, acceleration, breadth, extension risk, and data quality.", "good": "Higher is better; above 65 usually means leadership or strong emerging strength."},
            {"column": "stage", "meaning": "Lifecycle label: Leadership, Emerging, Pullback in uptrend, Exhausted leadership, Neutral, or Weak/Avoid.", "good": "Emerging and Pullback are often better watch zones; Leadership confirms momentum but may be crowded."},
            {"column": "movement_status", "meaning": "Simplified action bucket: currently_running, upcoming, watch, or avoid.", "good": "Target upcoming first for early rotation; use currently_running for confirmed leaders."},
            {"column": "movement_arrow", "meaning": "Directional arrow derived from the selected movement value: up, flat, or down.", "good": "Up arrows on 20D return and 20D RS show current price movement and benchmark outperformance."},
            {"column": "5D move / 20D move / 60D move", "meaning": "Arrow plus percentage return for roughly one week, one month, and one quarter.", "good": "A sector with positive 20D and 60D moves is more persistent than a one-week spike."},
            {"column": "20D RS / 60D RS", "meaning": "Arrow plus sector return minus Nifty 50 return over the same window.", "good": "Up arrows mean the sector is outperforming the benchmark."},
            {"column": "return_5d_pct", "meaning": "Sector return over roughly one trading week.", "good": "Positive 5D return helps confirm near-term money flow."},
            {"column": "return_20d_pct", "meaning": "Sector return over roughly one trading month.", "good": "Positive 20D return with positive relative strength indicates current movement."},
            {"column": "return_60d_pct", "meaning": "Sector return over roughly one quarter.", "good": "Positive 60D return shows the move has broader persistence."},
            {"column": "rs_20d_pct", "meaning": "20-day sector return minus Nifty 50 return.", "good": "Positive means the sector is beating Nifty over the last month."},
            {"column": "rs_60d_pct", "meaning": "60-day sector return minus Nifty 50 return.", "good": "Positive means the sector is beating Nifty over the last quarter."},
            {"column": "trend_score", "meaning": "Score from price versus 20/50/100/200 DMA, ADX, and RSI.", "good": "Higher means the sector trend is technically healthier."},
            {"column": "acceleration_score", "meaning": "Whether recent 5D strength is improving faster than 20D trend versus benchmark.", "good": "Higher values can identify upcoming sectors before full leadership appears."},
            {"column": "breadth_score", "meaning": "Participation score from sampled constituents above 20/50/200 DMA and positive 20D returns.", "good": "Higher means the move is broad, not driven by one or two stocks."},
            {"column": "above_20_pct", "meaning": "Percent of sampled stocks above 20-day moving average.", "good": "Shows short-term participation."},
            {"column": "above_50_pct", "meaning": "Percent of sampled stocks above 50-day moving average.", "good": "Shows medium-term participation."},
            {"column": "above_200_pct", "meaning": "Percent of sampled stocks above 200-day moving average.", "good": "Shows long-term trend health."},
        ],
        "stock_columns": [
            {"column": "stock_score", "meaning": "Composite stock score within selected sector.", "good": "Higher means stronger candidate relative to sector peers."},
            {"column": "stage", "meaning": "Stock state: Sector leader, Improving, Watch, or Laggard.", "good": "Sector leader and Improving are preferred for follow-up research."},
            {"column": "rs_20d_pct", "meaning": "20-day stock return minus selected sector return.", "good": "Positive means the stock is outperforming its own sector."},
            {"column": "rs_60d_pct", "meaning": "60-day stock return minus selected sector return.", "good": "Positive means sustained outperformance versus the sector."},
            {"column": "trend_score", "meaning": "Stock trend score from moving averages, ADX, and RSI.", "good": "Higher is better, but very overbought setups need caution."},
            {"column": "pattern_score", "meaning": "Chart-pattern bias score from detected OHLCV patterns.", "good": "Above 60 is constructive; below 40 is weak."},
            {"column": "volume_score", "meaning": "Volume confirmation score from current volume versus recent average.", "good": "Higher confirms participation with volume."},
            {"column": "risk_quality_score", "meaning": "Risk score based on volatility, drawdown, ATR, and liquidity.", "good": "Higher means cleaner risk profile."},
            {"column": "dominant_chart_pattern", "meaning": "Highest-confidence detected chart pattern.", "good": "Bullish continuation/reversal patterns support follow-up; bearish patterns require caution."},
        ],
        "chart_explanations": [
            {"chart": "Sector Rotation Score", "meaning": "Ranks sectors by overall rotation quality. Color shows movement status."},
            {"chart": "Sector Stage Mix", "meaning": "Shows how many sectors are in each lifecycle stage."},
            {"chart": "Relative Strength Map", "meaning": "X-axis is 20D relative strength, Y-axis is 60D relative strength. Upper-right is strongest."},
            {"chart": "Sector Returns", "meaning": "Compares 5D, 20D, and 60D returns to separate short-term spikes from persistent moves."},
            {"chart": "Sector Breadth Participation", "meaning": "Checks whether sector movement is broad across constituents."},
            {"chart": "Stock Targets", "meaning": "Ranks stocks inside selected sector by the stock score."},
            {"chart": "Stock Relative Strength vs Sector", "meaning": "Shows which stocks are outperforming their own sector."},
            {"chart": "Setup Quality Radar", "meaning": "Compares stock score, trend, pattern, volume, and risk quality in one view."},
        ],
    }


def _decision_summary(rotation: dict[str, Any], ranked_stocks: dict[str, Any]) -> dict[str, Any]:
    sectors = rotation.get("sectors", []) or []
    enriched = [_sector_card(row) for row in sectors]
    running = [row for row in enriched if row["movement_status"] == "currently_running"]
    upcoming = [row for row in enriched if row["movement_status"] == "upcoming"]
    avoid = [row for row in enriched if row["movement_status"] == "avoid"]
    target_next = upcoming[0] if upcoming else (running[0] if running else (enriched[0] if enriched else None))
    stock_targets = [_stock_card(row) for row in (ranked_stocks.get("stocks", []) or [])[:5]]
    top_stock = stock_targets[0] if stock_targets else None

    return {
        "currently_running_sectors": running[:5],
        "upcoming_sectors": upcoming[:5],
        "avoid_or_weak_sectors": avoid[:5],
        "target_next_sector": target_next,
        "target_stocks": stock_targets,
        "top_stock_target": top_stock,
        "headline": _headline(target_next, top_stock),
        "rules": {
            "currently_running": "Leadership sector with positive 20D and 60D relative strength, strong trend, and acceptable breadth.",
            "upcoming": "Emerging or pullback sector with positive recent relative strength or acceleration before full leadership confirmation.",
            "avoid": "Weak sector with negative relative strength and weak trend/breadth.",
        },
    }


def _sector_card(row: dict[str, Any]) -> dict[str, Any]:
    status = _sector_movement_status(row)
    metrics = row.get("metrics", {}) or {}
    relative = row.get("relative_strength", {}) or {}
    breadth = row.get("breadth", {}) or {}
    return {
        "sector_id": row.get("sector_id"),
        "name": row.get("name"),
        "stage": row.get("stage"),
        "movement_status": status,
        "rotation_score": row.get("rotation_score"),
        "return_5d": metrics.get("return_5d"),
        "return_20d": metrics.get("return_20d"),
        "return_60d": metrics.get("return_60d"),
        "return_5d_direction": _movement_direction(metrics.get("return_5d")),
        "return_20d_direction": _movement_direction(metrics.get("return_20d")),
        "return_60d_direction": _movement_direction(metrics.get("return_60d")),
        "rs_20d": relative.get("vs_benchmark_20d"),
        "rs_60d": relative.get("vs_benchmark_60d"),
        "rs_20d_direction": _movement_direction(relative.get("vs_benchmark_20d")),
        "rs_60d_direction": _movement_direction(relative.get("vs_benchmark_60d")),
        "trend_score": row.get("trend_score"),
        "acceleration_score": row.get("acceleration_score"),
        "breadth_score": breadth.get("breadth_score"),
        "root_causes": _sector_root_causes(row, status),
    }


def _stock_card(row: dict[str, Any]) -> dict[str, Any]:
    relative = row.get("relative_strength", {}) or {}
    return {
        "ticker": row.get("ticker"),
        "stage": row.get("stage"),
        "stock_score": row.get("stock_score"),
        "rs_20d": relative.get("vs_sector_20d"),
        "rs_60d": relative.get("vs_sector_60d"),
        "trend_score": row.get("trend_score"),
        "pattern_score": row.get("pattern_score"),
        "volume_score": row.get("volume_score"),
        "risk_quality_score": row.get("risk_quality_score"),
        "dominant_chart_pattern": row.get("dominant_chart_pattern"),
        "root_causes": row.get("reasons", [])[:5],
    }


def _sector_movement_status(row: dict[str, Any]) -> str:
    stage = str(row.get("stage", "")).lower()
    relative = row.get("relative_strength", {}) or {}
    rel20 = _num(relative.get("vs_benchmark_20d"))
    rel60 = _num(relative.get("vs_benchmark_60d"))
    trend = _num(row.get("trend_score"))
    acceleration = _num(row.get("acceleration_score"), 50)
    breadth = _num((row.get("breadth", {}) or {}).get("breadth_score"), 50)

    if "weak" in stage or (rel20 < -0.015 and rel60 < -0.015 and trend < 50):
        return "avoid"
    if "leadership" in stage and "exhausted" not in stage:
        return "currently_running"
    if rel20 > 0.015 and rel60 > 0.005 and trend >= 65 and breadth >= 50:
        return "currently_running"
    if "emerging" in stage or "pullback" in stage:
        return "upcoming"
    if rel20 > 0.005 and acceleration >= 55 and trend >= 45:
        return "upcoming"
    return "watch"


def _movement_direction(value: Any, threshold: float = 0.001) -> str:
    number = _num(value)
    if number > threshold:
        return "up"
    if number < -threshold:
        return "down"
    return "flat"


def _sector_root_causes(row: dict[str, Any], status: str) -> list[str]:
    metrics = row.get("metrics", {}) or {}
    relative = row.get("relative_strength", {}) or {}
    breadth = row.get("breadth", {}) or {}
    causes = [
        f"Rotation score {row.get('rotation_score')} with stage '{row.get('stage')}'.",
        f"20D relative strength vs Nifty: {_pct(relative.get('vs_benchmark_20d'))}.",
        f"60D relative strength vs Nifty: {_pct(relative.get('vs_benchmark_60d'))}.",
        f"Sector 20D / 60D returns: {_pct(metrics.get('return_20d'))} / {_pct(metrics.get('return_60d'))}.",
        f"Breadth score {breadth.get('breadth_score')} with {breadth.get('above_sma_50_percent')}% above 50-DMA.",
        f"Trend score {row.get('trend_score')} and acceleration score {row.get('acceleration_score')}.",
    ]
    if status == "currently_running":
        causes.append("Classified as currently running because relative strength and trend are already confirmed.")
    elif status == "upcoming":
        causes.append("Classified as upcoming because recent relative strength/acceleration is improving before full leadership confirmation.")
    elif status == "avoid":
        causes.append("Classified as avoid because trend and relative strength are weak.")
    return causes


def _headline(target_sector: dict[str, Any] | None, top_stock: dict[str, Any] | None) -> str:
    if not target_sector:
        return "No target sector could be selected from available data."
    sector = target_sector.get("name")
    status = target_sector.get("movement_status")
    if top_stock:
        return f"Target next: {sector} ({status}); first stock to inspect: {top_stock.get('ticker')}."
    return f"Target next: {sector} ({status}); no stock candidate was ranked."


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"

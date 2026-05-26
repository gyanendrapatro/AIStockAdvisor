from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from stock_advisor.analysis.pipeline import analyze_stock, sanitize_for_json


Analyzer = Callable[..., dict[str, Any]]

CASH_LIKE_SYMBOLS = {
    "LIQUIDCASE",
    "LIQUIDBEES",
    "LIQUIDETF",
    "LIQUIDIETF",
    "LIQUID",
}


def analyze_portfolio_holdings(
    holdings: list[dict[str, Any]],
    *,
    period: str | None = None,
    interval: str | None = None,
    include_news: bool = True,
    include_intelligence: bool = False,
    max_holdings: int | None = None,
    include_full_analysis: bool = False,
    analyzer: Analyzer = analyze_stock,
) -> dict[str, Any]:
    """Analyze broker holdings and return portfolio-aware action buckets."""
    selected_holdings = holdings[: max_holdings or len(holdings)]
    total_cost = sum(_holding_cost(row) for row in selected_holdings)
    total_value = sum(_holding_value(row) for row in selected_holdings)

    rows: list[dict[str, Any]] = []
    for holding in selected_holdings:
        rows.append(
            _analyze_holding(
                holding,
                total_value=total_value,
                period=period,
                interval=interval,
                include_news=include_news,
                include_intelligence=include_intelligence,
                include_full_analysis=include_full_analysis,
                analyzer=analyzer,
            )
        )

    action_counts = Counter(row.get("recommendation", {}).get("bucket", "UNKNOWN") for row in rows)
    stance_counts = Counter(row.get("recommendation", {}).get("stance", "review") for row in rows)
    ranked_by_score = sorted(rows, key=lambda row: _number(row.get("score")), reverse=True)
    ranked_by_weight = sorted(rows, key=lambda row: _number(row.get("weight_percent")), reverse=True)

    return sanitize_for_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "holding_count": len(selected_holdings),
            "total_cost_value": round(total_cost, 2),
            "total_current_value": round(total_value, 2),
            "total_unrealized_pnl": round(total_value - total_cost, 2) if total_value or total_cost else None,
            "total_unrealized_pnl_percent": round(((total_value - total_cost) / total_cost) * 100, 2)
            if total_cost
            else None,
            "action_counts": dict(action_counts),
            "stance_counts": dict(stance_counts),
            "top_add_candidates": [
                row for row in ranked_by_score if row.get("recommendation", {}).get("stance") == "add"
            ][:5],
            "largest_exposures": ranked_by_weight[:5],
            "holdings": rows,
            "warnings": [
                "Educational portfolio triage only; this is not financial advice.",
                "Recommendations depend on free-source market/fundamental/news coverage and should be checked against official exchange filings before trading.",
            ],
        }
    )


def _analyze_holding(
    holding: dict[str, Any],
    *,
    total_value: float,
    period: str | None,
    interval: str | None,
    include_news: bool,
    include_intelligence: bool,
    include_full_analysis: bool,
    analyzer: Analyzer,
) -> dict[str, Any]:
    symbol = _holding_symbol(holding)
    ticker = _holding_ticker(holding)
    quantity = _number(holding.get("totalQty") or holding.get("quantity"))
    cost_value = _holding_cost(holding)
    current_value = _holding_value(holding)
    weight_percent = round((current_value / total_value) * 100, 2) if total_value else None
    pnl = _number(holding.get("unrealized_pnl"))
    if not pnl and current_value and cost_value:
        pnl = current_value - cost_value
    pnl_percent = holding.get("unrealized_pnl_percent")
    if pnl_percent is None and cost_value:
        pnl_percent = (pnl / cost_value) * 100

    base = {
        "symbol": symbol,
        "ticker": ticker,
        "quantity": quantity,
        "cost_value": round(cost_value, 2),
        "current_value": round(current_value, 2),
        "weight_percent": weight_percent,
        "unrealized_pnl": round(pnl, 2),
        "unrealized_pnl_percent": round(float(pnl_percent), 2) if pnl_percent is not None else None,
    }

    if _is_cash_like(symbol):
        base.update(
            {
                "score": None,
                "signal": "Cash / Liquid",
                "recommendation": {
                    "bucket": "CASH_BUFFER",
                    "stance": "hold",
                    "action": "Hold as liquidity buffer for future rebalancing.",
                    "reasons": ["Detected as liquid/cash-like holding."],
                },
            }
        )
        return base

    if not ticker:
        base.update(
            {
                "score": None,
                "signal": "Review Manually",
                "recommendation": {
                    "bucket": "REVIEW_MANUALLY",
                    "stance": "review",
                    "action": "Review manually; no analysis ticker could be inferred.",
                    "reasons": ["Broker holding did not include a usable trading symbol."],
                },
            }
        )
        return base

    try:
        analysis = analyzer(
            ticker,
            period=period,
            interval=interval,
            include_news=include_news,
            include_intelligence=include_intelligence,
        )
    except Exception as exc:
        base.update(
            {
                "score": None,
                "signal": "Analysis Error",
                "recommendation": {
                    "bucket": "REVIEW_MANUALLY",
                    "stance": "review",
                    "action": "Review manually; analysis failed.",
                    "reasons": [str(exc)],
                },
            }
        )
        return base

    indicators = analysis.get("latest_indicators", {}) or {}
    fundamentals = analysis.get("fundamentals", {}) or {}
    score = analysis.get("final_score")
    recommendation = _recommend_holding(
        score=_number(score),
        technical_score=_number(analysis.get("technical_score")),
        fundamental_score=_number(analysis.get("fundamental_score")),
        risk_score=_number(analysis.get("risk_score")),
        momentum_score=_number(analysis.get("momentum_liquidity_score")),
        pnl_percent=float(pnl_percent) if pnl_percent is not None else None,
        weight_percent=weight_percent,
        close=_optional_number(indicators.get("close")),
        sma200=_optional_number(indicators.get("sma_200")),
        rsi=_optional_number(indicators.get("rsi_14")),
        signal=str(analysis.get("signal") or ""),
        risks=analysis.get("risks", []) or [],
    )

    base.update(
        {
            "score": score,
            "signal": analysis.get("signal"),
            "confidence": analysis.get("confidence"),
            "technical_score": analysis.get("technical_score"),
            "fundamental_score": analysis.get("fundamental_score"),
            "news_score": analysis.get("news_score"),
            "risk_score": analysis.get("risk_score"),
            "momentum_liquidity_score": analysis.get("momentum_liquidity_score"),
            "event_intelligence_score": analysis.get("event_intelligence_score"),
            "sector": fundamentals.get("sector"),
            "industry": fundamentals.get("industry"),
            "close": indicators.get("close"),
            "sma50": indicators.get("sma_50"),
            "sma200": indicators.get("sma_200"),
            "rsi": indicators.get("rsi_14"),
            "recommendation": recommendation,
            "reasons": analysis.get("reasons", [])[:5],
            "risks": analysis.get("risks", [])[:5],
            "warnings": (analysis.get("metadata", {}) or {}).get("warnings", [])[:5],
        }
    )
    if include_full_analysis:
        base["analysis"] = analysis
    return base


def _recommend_holding(
    *,
    score: float,
    technical_score: float,
    fundamental_score: float,
    risk_score: float,
    momentum_score: float,
    pnl_percent: float | None,
    weight_percent: float | None,
    close: float | None,
    sma200: float | None,
    rsi: float | None,
    signal: str,
    risks: list[str],
) -> dict[str, Any]:
    below_200 = close is not None and sma200 is not None and close < sma200
    overbought = rsi is not None and rsi >= 68
    large_profit = pnl_percent is not None and pnl_percent >= 100
    deep_loss = pnl_percent is not None and pnl_percent <= -30
    concentrated = weight_percent is not None and weight_percent >= 12
    weak_quality = fundamental_score < 55 or risk_score < 55

    reasons: list[str] = []
    if score:
        reasons.append(f"Model score is {round(score, 2)} with signal '{signal or 'unknown'}'.")
    if below_200:
        reasons.append("Price is below the 200-day average, so avoid aggressive averaging down.")
    if overbought:
        reasons.append("RSI is elevated, so chasing fresh buys has poor risk/reward.")
    if concentrated:
        reasons.append("Position is a large portfolio weight.")
    if risks:
        reasons.append(str(risks[0]))

    if score >= 75:
        if large_profit or overbought or concentrated:
            return {
                "bucket": "HOLD_TRIM",
                "stance": "trim",
                "action": "Hold the core position; consider partial profit booking or a trailing stop.",
                "reasons": reasons or ["High score, but position risk calls for profit protection."],
            }
        return {
            "bucket": "ADD_ON_DIPS",
            "stance": "add",
            "action": "Add on dips or staged pullbacks; avoid chasing sharp intraday moves.",
            "reasons": reasons or ["High score and setup quality support staged add-ons."],
        }

    if score >= 68:
        if deep_loss and below_200:
            return {
                "bucket": "HOLD_NO_ADD",
                "stance": "hold",
                "action": "Hold only; wait for confirmation above the 200-day average before adding.",
                "reasons": reasons or ["Recovery is not confirmed enough to average down."],
            }
        return {
            "bucket": "SMALL_ADD_OR_HOLD",
            "stance": "add",
            "action": "Hold; small add-ons are acceptable only if position sizing stays controlled.",
            "reasons": reasons or ["Score is constructive, but not strong enough for aggressive buying."],
        }

    if score >= 58:
        return {
            "bucket": "HOLD_NO_ADD" if deep_loss or weak_quality else "HOLD",
            "stance": "hold",
            "action": "Hold and monitor; avoid fresh add-ons until score and trend improve.",
            "reasons": reasons or ["Setup is mixed."],
        }

    if deep_loss or weak_quality or momentum_score < 35:
        return {
            "bucket": "REDUCE_ON_BOUNCE",
            "stance": "reduce",
            "action": "Avoid averaging down; reduce on rebounds unless new fundamental evidence improves.",
            "reasons": reasons or ["Weak score and portfolio loss need risk control."],
        }

    return {
        "bucket": "HOLD_REVIEW",
        "stance": "hold",
        "action": "Hold only as a review candidate.",
        "reasons": reasons or ["Score is not strong enough for add-ons."],
    }


def _holding_symbol(holding: dict[str, Any]) -> str:
    return str(holding.get("tradingSymbol") or holding.get("symbol") or "").strip().upper()


def _holding_ticker(holding: dict[str, Any]) -> str | None:
    ticker = str(holding.get("analysis_ticker") or "").strip().upper()
    if ticker:
        return ticker
    symbol = _holding_symbol(holding)
    if not symbol:
        return None
    exchange = str(holding.get("exchange") or "").strip().upper()
    if exchange == "BSE":
        return f"{symbol}.BO"
    return f"{symbol}.NS"


def _holding_cost(holding: dict[str, Any]) -> float:
    explicit = _number(holding.get("cost_value"))
    if explicit:
        return explicit
    return _number(holding.get("totalQty") or holding.get("quantity")) * _number(holding.get("avgCostPrice"))


def _holding_value(holding: dict[str, Any]) -> float:
    explicit = _number(holding.get("current_value"))
    if explicit:
        return explicit
    return _number(holding.get("totalQty") or holding.get("quantity")) * _number(holding.get("last_price"))


def _is_cash_like(symbol: str) -> bool:
    normalized = symbol.strip().upper()
    return normalized in CASH_LIKE_SYMBOLS or normalized.startswith("LIQUID")


def _number(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_number(value: Any) -> float | None:
    number = _number(value)
    return number if number else None

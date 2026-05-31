from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, TypedDict

from stock_advisor.agents.ai_analyst import fallback_commentary
from stock_advisor.agents.llm_client import load_llm_config, synthesize_with_llm
from stock_advisor.analysis.pipeline import research_stock, sanitize_for_json
from stock_advisor.data.exchange_filings import get_exchange_announcements
from stock_advisor.data.market_data import refresh_latest_exchange_eod_cache

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - exercised when optional dependency is absent
    END = "__end__"
    StateGraph = None


class StockResearchState(TypedDict, total=False):
    ticker: str
    period: str
    interval: str
    intelligence_days: int
    intelligence_strategic_days: int
    include_exchange_announcements: bool
    parse_exchange_pdfs: bool
    include_llm: bool
    llm_provider: str | None
    llm_model: str | None
    llm_base_url: str | None
    force_refresh_prices: bool
    warnings: list[str]
    graph_steps: list[str]
    workflow_backend: str
    latest_eod_refresh: dict[str, Any]
    analysis: dict[str, Any]
    exchange_announcements: dict[str, Any]
    deterministic: dict[str, Any]
    llm_result: dict[str, Any]
    llm_config: dict[str, Any]


def run_stock_research_agent(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    intelligence_days: int = 45,
    intelligence_strategic_days: int = 365,
    include_exchange_announcements: bool = True,
    parse_exchange_pdfs: bool = True,
    include_llm: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Run the app-internal LangGraph stock research agent used by UI and MCP.

    LangGraph is the primary workflow engine. If it is not installed, the same
    nodes run sequentially and the result is marked as degraded.
    """
    normalized = _normalize_ticker(ticker)
    state: StockResearchState = {
        "ticker": normalized,
        "period": period,
        "interval": interval,
        "intelligence_days": int(intelligence_days),
        "intelligence_strategic_days": int(intelligence_strategic_days),
        "include_exchange_announcements": bool(include_exchange_announcements),
        "parse_exchange_pdfs": bool(parse_exchange_pdfs),
        "include_llm": bool(include_llm),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "force_refresh_prices": bool(force_refresh_prices),
        "warnings": [],
        "graph_steps": [],
    }
    final_state = _run_langgraph_workflow(state) if StateGraph is not None else _run_sequential_workflow(state)
    return _finalize_state(final_state)


def _run_langgraph_workflow(state: StockResearchState) -> StockResearchState:
    graph = StateGraph(StockResearchState)
    graph.add_node("refresh_market_data", _node_refresh_market_data)
    graph.add_node("stock_research", _node_stock_research)
    graph.add_node("exchange_filings", _node_exchange_filings)
    graph.add_node("evidence_scoring", _node_evidence_scoring)
    graph.add_node("llm_synthesis", _node_llm_synthesis)
    graph.set_entry_point("refresh_market_data")
    graph.add_edge("refresh_market_data", "stock_research")
    graph.add_edge("stock_research", "exchange_filings")
    graph.add_edge("exchange_filings", "evidence_scoring")
    graph.add_edge("evidence_scoring", "llm_synthesis")
    graph.add_edge("llm_synthesis", END)
    compiled = graph.compile()
    result = compiled.invoke(state)
    result["workflow_backend"] = "langgraph"
    return result


def _run_sequential_workflow(state: StockResearchState) -> StockResearchState:
    current = state
    for node in (
        _node_refresh_market_data,
        _node_stock_research,
        _node_exchange_filings,
        _node_evidence_scoring,
        _node_llm_synthesis,
    ):
        current = node(current)
    current["workflow_backend"] = "python_orchestrator_fallback"
    current["warnings"] = [
        *current.get("warnings", []),
        "LangGraph is not installed. The same research nodes ran sequentially; install langgraph for the production graph runtime.",
    ]
    return current


def _node_refresh_market_data(state: StockResearchState) -> StockResearchState:
    warnings = list(state.get("warnings", []))
    if state.get("force_refresh_prices") and state.get("interval") == "1d":
        try:
            refresh_result = refresh_latest_exchange_eod_cache([state["ticker"]], interval=state["interval"], period="1d")
        except Exception as exc:  # noqa: BLE001
            refresh_result = {"error": str(exc)}
            warnings.append(f"Latest NSE/BSE EOD refresh failed before stock research: {exc}")
    else:
        refresh_result = {}
    return {
        **state,
        "latest_eod_refresh": refresh_result,
        "warnings": warnings,
        "graph_steps": [*state.get("graph_steps", []), "refresh_market_data"],
    }


def _node_stock_research(state: StockResearchState) -> StockResearchState:
    analysis = research_stock(
        state["ticker"],
        period=state.get("period"),
        interval=state.get("interval"),
        intelligence_days=int(state.get("intelligence_days", 45)),
        intelligence_strategic_days=int(state.get("intelligence_strategic_days", 365)),
        force_refresh_prices=bool(state.get("force_refresh_prices", True)),
    )
    return {**state, "analysis": analysis, "graph_steps": [*state.get("graph_steps", []), "stock_research"]}


def _node_exchange_filings(state: StockResearchState) -> StockResearchState:
    if state.get("include_exchange_announcements", True):
        exchange_announcements = get_exchange_announcements(
            state["ticker"],
            limit=12,
            days=max(90, int(state.get("intelligence_strategic_days", 365))),
            parse_pdfs=bool(state.get("parse_exchange_pdfs", True)),
            max_pdf_documents=2,
        )
    else:
        exchange_announcements = {}
    warnings = [*state.get("warnings", []), *(exchange_announcements or {}).get("warnings", [])[:4]]
    return {
        **state,
        "exchange_announcements": exchange_announcements,
        "warnings": warnings,
        "graph_steps": [*state.get("graph_steps", []), "exchange_filings"],
    }


def _node_evidence_scoring(state: StockResearchState) -> StockResearchState:
    deterministic = _deterministic_brief(state.get("analysis") or {}, state.get("exchange_announcements") or {})
    return {**state, "deterministic": deterministic, "graph_steps": [*state.get("graph_steps", []), "evidence_scoring"]}


def _node_llm_synthesis(state: StockResearchState) -> StockResearchState:
    llm_config = load_llm_config(
        provider=state.get("llm_provider"),
        model=state.get("llm_model") or None,
        base_url=state.get("llm_base_url") or None,
    )
    deterministic = state.get("deterministic") or {}
    analysis = state.get("analysis") or {}
    exchange_announcements = state.get("exchange_announcements") or {}
    llm_result = (
        synthesize_with_llm(
            system_prompt=_system_prompt(),
            user_prompt=_llm_prompt(analysis, exchange_announcements, deterministic),
            config=llm_config,
        )
        if state.get("include_llm")
        else {
            "used": False,
            "provider": llm_config.provider,
            "model": llm_config.model,
            "warning": "Deep agent synthesis unavailable because LLM synthesis is disabled for this run.",
        }
    )
    warnings = list(state.get("warnings", []))
    if llm_result.get("warning"):
        warnings.append(str(llm_result["warning"]))
    return {
        **state,
        "llm_result": llm_result,
        "llm_config": {
            "provider": llm_config.provider,
            "model": llm_config.model,
            "base_url": llm_config.base_url,
            "enabled": llm_config.enabled,
        },
        "warnings": warnings,
        "graph_steps": [*state.get("graph_steps", []), "llm_synthesis"],
    }


def _finalize_state(state: StockResearchState) -> dict[str, Any]:
    analysis = state.get("analysis") or {}
    deterministic = state.get("deterministic") or _deterministic_brief(analysis, state.get("exchange_announcements") or {})
    llm_result = state.get("llm_result") or {}
    llm_config = state.get("llm_config") or {}
    warnings = list(dict.fromkeys([*state.get("warnings", []), *analysis.get("metadata", {}).get("warnings", [])]))
    return sanitize_for_json(
        {
            "ticker": state.get("ticker"),
            "period": state.get("period"),
            "interval": state.get("interval"),
            "workflow": {
                "backend": state.get("workflow_backend"),
                "steps": state.get("graph_steps", []),
                "langgraph_available": StateGraph is not None,
            },
            "verdict": deterministic["verdict"],
            "executive_summary": llm_result.get("content") if llm_result.get("used") else deterministic["summary"],
            "deterministic_summary": deterministic["summary"],
            "watch_items": deterministic["watch_items"],
            "evidence_map": deterministic["evidence_map"],
            "analysis": analysis,
            "exchange_announcements": state.get("exchange_announcements") or {},
            "latest_eod_refresh": state.get("latest_eod_refresh") or {},
            "llm": {
                "requested": bool(state.get("include_llm")),
                "used": bool(llm_result.get("used")),
                "provider": llm_result.get("provider") or llm_config.get("provider"),
                "model": llm_result.get("model") or llm_config.get("model"),
                "base_url": llm_config.get("base_url") if state.get("include_llm") else None,
                "warning": llm_result.get("warning"),
            },
            "warnings": warnings,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "not_financial_advice": True,
        }
    )


def _deterministic_brief(analysis: dict[str, Any], announcements: dict[str, Any]) -> dict[str, Any]:
    score = _number(analysis.get("final_score"), 0)
    risk = _number(analysis.get("risk_score"), 50)
    confidence = _number(analysis.get("confidence"), 0)
    signal = str(analysis.get("signal") or "Watch")
    action = _action_label(score, risk, confidence)

    indicators = analysis.get("latest_indicators") or {}
    fundamentals = analysis.get("fundamentals") or {}
    intelligence = analysis.get("company_intelligence") or {}
    analyst = analysis.get("analyst_insights") or {}
    events = analysis.get("stock_events") or {}
    announcement_rows = (announcements or {}).get("announcements", [])

    latest_price = _number(indicators.get("close"))
    rsi = _number(indicators.get("rsi_14"))
    market_cap = _number(fundamentals.get("marketCap"))
    target_upside = _number(((analyst.get("consensus") or {}).get("target_upside_percent")))

    evidence_map = {
        "technical": {
            "signal": signal,
            "latest_price": latest_price,
            "rsi_14": rsi,
            "above_50_sma": indicators.get("above_sma_50"),
            "above_200_sma": indicators.get("above_sma_200"),
            "patterns": (analysis.get("chart_patterns") or {}).get("patterns", [])[:5],
        },
        "fundamental": {
            "market_cap": market_cap,
            "sector": fundamentals.get("sector"),
            "industry": fundamentals.get("industry"),
            "trailing_pe": fundamentals.get("trailingPE"),
            "forward_pe": fundamentals.get("forwardPE"),
            "revenue_growth": fundamentals.get("revenueGrowth"),
            "earnings_growth": fundamentals.get("earningsGrowth"),
            "promoter_holding": fundamentals.get("promoter_holding"),
            "promoter_pledge": fundamentals.get("promoter_pledge"),
        },
        "news_and_events": {
            "news_count": len(analysis.get("news") or []),
            "exchange_announcement_count": len(announcement_rows),
            "company_intelligence_categories": intelligence.get("material_event_counts", {}),
            "exchange_categories": (announcements or {}).get("category_counts", {}),
            "analyst_target_upside_percent": target_upside,
            "event_provider_count": len(events.get("providers") or []),
        },
    }

    watch_items = list(dict.fromkeys(
        [
            *_compact_list(analysis.get("risks"), limit=3),
            *_compact_list(intelligence.get("monitoring_focus"), limit=3),
            *_announcement_watch_items(announcement_rows),
        ]
    ))[:8]

    summary = fallback_commentary(analysis)
    if announcement_rows:
        latest = announcement_rows[0]
        summary += (
            f"\nLatest exchange filing: {latest.get('headline') or 'announcement'} "
            f"({latest.get('exchange')}, {latest.get('published_at')})."
        )
    summary += f"\nAgent verdict: {action}."

    return {
        "verdict": {
            "action": action,
            "base_signal": signal,
            "score": score,
            "risk_score": risk,
            "confidence": confidence,
        },
        "summary": summary,
        "watch_items": watch_items,
        "evidence_map": evidence_map,
    }


def _system_prompt() -> str:
    return (
        "You are an equity research assistant for educational stock analysis. "
        "Use only the supplied structured data. Separate evidence from inference. "
        "Do not invent filings, numbers, targets, or news. End with watch/hold/add/avoid reasoning, not a guaranteed trade call."
    )


def _llm_prompt(analysis: dict[str, Any], announcements: dict[str, Any], deterministic: dict[str, Any]) -> str:
    compact = {
        "ticker": analysis.get("ticker") or analysis.get("metadata", {}).get("ticker"),
        "scores": {
            "signal": analysis.get("signal"),
            "final_score": analysis.get("final_score"),
            "technical_score": analysis.get("technical_score"),
            "fundamental_score": analysis.get("fundamental_score"),
            "news_score": analysis.get("news_score"),
            "risk_score": analysis.get("risk_score"),
            "confidence": analysis.get("confidence"),
        },
        "reasons": _compact_list(analysis.get("reasons"), limit=6),
        "risks": _compact_list(analysis.get("risks"), limit=6),
        "latest_indicators": _select_keys(
            analysis.get("latest_indicators") or {},
            [
                "close",
                "latest_date",
                "rsi_14",
                "macd",
                "macd_signal",
                "adx_14",
                "sma_20",
                "sma_50",
                "sma_100",
                "sma_200",
                "volume_ratio_20d",
                "max_drawdown",
                "distance_52w_high_pct",
            ],
        ),
        "fundamentals": _select_keys(
            analysis.get("fundamentals") or {},
            [
                "shortName",
                "sector",
                "industry",
                "marketCap",
                "trailingPE",
                "forwardPE",
                "debtToEquity",
                "profitMargins",
                "revenueGrowth",
                "earningsGrowth",
                "returnOnEquity",
                "promoter_holding",
                "promoter_pledge",
                "fii_holding",
                "dii_holding",
            ],
        ),
        "news": _article_rows(analysis.get("news") or [], limit=5),
        "company_intelligence": _select_keys(
            analysis.get("company_intelligence") or {},
            [
                "business_areas",
                "material_event_counts",
                "positive_catalysts",
                "order_book_updates",
                "innovation_updates",
                "risk_flags",
                "sector_fit",
                "monitoring_focus",
            ],
        ),
        "analyst_insights": _select_keys(
            analysis.get("analyst_insights") or {},
            ["consensus", "recommendation_summary", "upgrades_downgrades"],
        ),
        "stock_events": _select_keys(
            analysis.get("stock_events") or {},
            ["calendar_events", "recent_dividends", "recent_splits", "earnings_dates"],
        ),
        "exchange_announcements": _announcement_rows((announcements or {}).get("announcements", []), limit=8),
        "deterministic_verdict": deterministic.get("verdict"),
    }
    return (
        "Create a concise stock research brief with these sections: Verdict, Why, Risks, Latest filings/news, "
        "What to monitor, and Data gaps. Use bullet points. Data:\n"
        f"{json.dumps(compact, indent=2, default=str)[:16000]}"
    )


def _action_label(score: float, risk: float, confidence: float) -> str:
    if score >= 75 and risk >= 55 and confidence >= 60:
        return "Add on confirmation"
    if score >= 65 and risk >= 45:
        return "Hold / buy watch"
    if score >= 50:
        return "Hold / monitor"
    if score >= 40:
        return "Avoid fresh add"
    return "Reduce / avoid"


def _announcement_watch_items(rows: list[dict[str, Any]]) -> list[str]:
    items = []
    for row in rows[:4]:
        categories = ", ".join(row.get("material_categories") or [])
        headline = row.get("headline")
        if headline:
            items.append(f"Verify exchange filing: {headline}" + (f" [{categories}]" if categories else ""))
    return items


def _article_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        _select_keys(row, ["title", "source", "provider", "time_published", "overall_sentiment_label", "url"])
        for row in rows[:limit]
    ]


def _announcement_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        _select_keys(
            row,
            [
                "exchange",
                "headline",
                "published_at",
                "category",
                "material_categories",
                "attachment_url",
                "attachment_text_excerpt",
            ],
        )
        for row in rows[:limit]
    ]


def _select_keys(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in keys if key in row and row.get(key) is not None}


def _compact_list(value: Any, *, limit: int) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list | tuple):
        return list(value[:limit])
    return [value]


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return round(float(value), 4)
    except (TypeError, ValueError):
        return default


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker or "").strip().upper()
    if value and "." not in value:
        return f"{value}.NS"
    return value

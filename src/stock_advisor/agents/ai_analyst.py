from __future__ import annotations


def generate_ai_commentary(analysis: dict) -> str:
    """Generate free local commentary from the structured analysis."""
    return fallback_commentary(analysis)


def fallback_commentary(analysis: dict) -> str:
    reasons = analysis.get("reasons", [])[:3]
    risks = analysis.get("risks", [])[:3]
    warnings = analysis.get("metadata", {}).get("warnings", [])[:2]
    news = analysis.get("news", [])[:2]
    watch_items = []
    if analysis.get("latest_indicators", {}).get("rsi_14") is not None:
        watch_items.append("RSI moving back toward a neutral range")
    if analysis.get("latest_indicators", {}).get("sma_20") is not None:
        watch_items.append("price holding above or below the 20-day average")
    if news:
        watch_items.append("fresh headline sentiment from Yahoo Finance")
    return (
        f"Verdict: {analysis.get('signal')} with score {analysis.get('final_score')}.\n"
        f"Reasons: {'; '.join(reasons) if reasons else 'No clear reasons available'}.\n"
        f"Risks: {'; '.join(risks) if risks else 'No major risk flags from available data'}.\n"
        f"Data notes: {'; '.join(warnings) if warnings else 'No data-provider warnings'}.\n"
        f"What to watch next: {'; '.join(watch_items[:3]) if watch_items else 'price trend, volume, and major news updates'}.\n"
        "Use this as a watchlist signal, not a trade instruction."
    )

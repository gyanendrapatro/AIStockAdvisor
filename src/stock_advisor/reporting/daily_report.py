from __future__ import annotations
from datetime import datetime
from pathlib import Path
from stock_advisor.analysis.pipeline import rank_watchlist
from stock_advisor.agents.ai_analyst import generate_ai_commentary
from stock_advisor.config.settings import settings


def generate_daily_report(group: str | None = None) -> str:
    rows = rank_watchlist(group)
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# Daily Stock Advisor Report - {today}", "", "> Educational use only. Not financial advice.", ""]
    lines.append("| Rank | Ticker | Signal | Score | Confidence |")
    lines.append("|---:|---|---|---:|---:|")
    for idx, row in enumerate(rows, start=1):
        lines.append(f"| {idx} | {row.get('ticker')} | {row.get('signal')} | {row.get('final_score')} | {row.get('confidence', '')}% |")
    lines.append("")
    lines.append("## Analyst Notes")
    for row in rows[:10]:
        lines.append(f"\n### {row.get('ticker')} — {row.get('signal')}")
        lines.append(generate_ai_commentary(row))
    report = "\n".join(lines)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    suffix = group or "all"
    path = settings.report_dir / f"daily_report_{suffix}_{today}.md"
    path.write_text(report, encoding="utf-8")
    return str(path)

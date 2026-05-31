from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastmcp import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _mcp_config() -> dict[str, Any]:
    env = {
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "SEC_USER_AGENT": os.getenv("SEC_USER_AGENT", "ai-stock-advisor-mcp/0.1 local-research@example.com"),
        "OWNERSHIP_DATA_PATH": os.getenv("OWNERSHIP_DATA_PATH", "ownership.yaml"),
    }
    for key in (
        "DHAN_ACCESS_TOKEN",
        "DHAN_CLIENT_ID",
        "OPENAI_API_KEY",
        "STOCK_ADVISOR_LLM_API_KEY",
        "STOCK_ADVISOR_LLM_PROVIDER",
        "STOCK_ADVISOR_LLM_MODEL",
        "STOCK_ADVISOR_LLM_BASE_URL",
        "STOCK_ADVISOR_LLM_TIMEOUT_SECONDS",
    ):
        value = os.getenv(key)
        if value:
            env[key] = value

    return {
        "mcpServers": {
            "ai-stock-advisor": {
                "command": str(PYTHON),
                "args": ["-m", "stock_advisor.mcp.server"],
                "env": env,
            }
        }
    }


async def _run(args: argparse.Namespace) -> None:
    async with Client(_mcp_config()) as client:
        result = (
            await client.call_tool(
                "run_stock_research_agent",
                {
                    "ticker": args.ticker,
                    "period": args.period,
                    "interval": args.interval,
                    "include_exchange_announcements": not args.no_exchange_announcements,
                    "parse_exchange_pdfs": not args.no_pdf_parse,
                    "include_llm": not args.no_llm,
                    "force_refresh_prices": not args.no_price_refresh,
                },
            )
        ).data
    if args.full:
        print(json.dumps(result, indent=2, default=str))
        return

    analysis = result.get("analysis") or {}
    exchange = result.get("exchange_announcements") or {}
    compact = {
        "ticker": result.get("ticker"),
        "workflow": result.get("workflow"),
        "verdict": result.get("verdict"),
        "summary": result.get("executive_summary"),
        "reasons": analysis.get("reasons", []),
        "risks": analysis.get("risks", []),
        "metadata": analysis.get("metadata", {}),
        "latest_indicators": analysis.get("latest_indicators", {}),
        "fundamentals": analysis.get("fundamentals", {}),
        "news": (analysis.get("news") or [])[:8],
        "company_intelligence": analysis.get("company_intelligence", {}),
        "analyst_insights": analysis.get("analyst_insights", {}),
        "stock_events": analysis.get("stock_events", {}),
        "exchange_announcements": {
            "providers": exchange.get("providers", []),
            "category_counts": exchange.get("category_counts", {}),
            "announcements": (exchange.get("announcements") or [])[:8],
        },
        "llm": result.get("llm", {}),
        "warnings": result.get("warnings", []),
    }
    print(json.dumps(compact, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stock research through the local AI Stock Advisor MCP server.")
    parser.add_argument("ticker")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-exchange-announcements", action="store_true")
    parser.add_argument("--no-pdf-parse", action="store_true")
    parser.add_argument("--no-price-refresh", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

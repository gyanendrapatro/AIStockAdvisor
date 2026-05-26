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
    for key in ("DHAN_ACCESS_TOKEN", "DHAN_CLIENT_ID", "STOOQ_API_KEY"):
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


async def _run(check_dhan: bool, analyze_portfolio: bool) -> None:
    async with Client(_mcp_config()) as client:
        tools = await client.list_tools()
        tool_names = sorted(tool.name for tool in tools)
        health = (await client.call_tool("health_check", {})).data

        output: dict[str, Any] = {
            "mcp_server": "ai-stock-advisor",
            "tool_count": len(tool_names),
            "has_analyze_dhan_portfolio": "analyze_dhan_portfolio" in tool_names,
            "providers": health.get("providers", {}),
        }

        if check_dhan:
            profile = (await client.call_tool("get_dhan_profile", {})).data
            output["dhan_profile"] = {
                "dhanClientId": profile.get("dhanClientId"),
                "tokenValidity": profile.get("tokenValidity"),
                "activeSegment": profile.get("activeSegment"),
                "ddpi": profile.get("ddpi"),
                "mtf": profile.get("mtf"),
                "dataPlan": profile.get("dataPlan"),
            }

        if analyze_portfolio:
            summary = (
                await client.call_tool(
                    "analyze_dhan_portfolio",
                    {
                        "include_news": False,
                        "include_intelligence": False,
                        "include_full_analysis": False,
                    },
                )
            ).data
            analysis = summary.get("portfolio_analysis", {})
            output["portfolio_analysis"] = {
                "holding_count": analysis.get("holding_count"),
                "total_current_value": analysis.get("total_current_value"),
                "total_unrealized_pnl": analysis.get("total_unrealized_pnl"),
                "action_counts": analysis.get("action_counts"),
            }

        print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the AI Stock Advisor MCP server through MCP calls.")
    parser.add_argument("--dhan", action="store_true", help="Also call the read-only Dhan profile tool.")
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Also call analyze_dhan_portfolio. Requires DHAN_ACCESS_TOKEN.",
    )
    args = parser.parse_args()
    asyncio.run(_run(check_dhan=args.dhan or args.portfolio, analyze_portfolio=args.portfolio))


if __name__ == "__main__":
    main()

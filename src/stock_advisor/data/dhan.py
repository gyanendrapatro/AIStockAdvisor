from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from stock_advisor.data.market_data import get_price_history

logger = logging.getLogger(__name__)

DHAN_BASE_URL = "https://api.dhan.co/v2"


def dhan_config_status() -> dict[str, Any]:
    """Return whether read-only Dhan Trading API access is configured."""
    token = _access_token()
    return {
        "configured": bool(token),
        "required": False,
        "cost": "free_trading_api",
        "mode": "read_only",
        "notes": "Requires DHAN_ACCESS_TOKEN. Dhan Web-generated tokens are valid for 24 hours.",
    }


def get_dhan_profile() -> dict[str, Any]:
    """Return Dhan profile/token status using the read-only profile endpoint."""
    return _request_json("/profile")


def get_dhan_holdings() -> list[dict[str, Any]]:
    """Return Dhan demat holdings."""
    payload = _request_json("/holdings")
    return payload if isinstance(payload, list) else []


def get_dhan_positions() -> list[dict[str, Any]]:
    """Return Dhan open/carry-forward positions."""
    payload = _request_json("/positions")
    return payload if isinstance(payload, list) else []


def get_dhan_fund_limits() -> dict[str, Any]:
    """Return Dhan trading account fund limits."""
    payload = _request_json("/fundlimit")
    return payload if isinstance(payload, dict) else {}


def get_dhan_portfolio_summary(include_market_values: bool = True) -> dict[str, Any]:
    """Summarize Dhan holdings and positions for portfolio-aware stock research."""
    holdings = get_dhan_holdings()
    positions = get_dhan_positions()

    enriched_holdings = [
        _enrich_holding_with_market_value(holding) if include_market_values else dict(holding)
        for holding in holdings
    ]
    holding_cost = sum(_number(row.get("cost_value")) for row in enriched_holdings)
    holding_value = sum(_number(row.get("current_value")) for row in enriched_holdings)
    unrealized = sum(_number(row.get("unrealized_pnl")) for row in enriched_holdings)
    position_pnl = sum(_number(row.get("realizedProfit")) + _number(row.get("unrealizedProfit")) for row in positions)

    sector_like_exposure: dict[str, float] = {}
    for row in enriched_holdings:
        exchange = str(row.get("exchange") or "UNKNOWN")
        sector_like_exposure[exchange] = sector_like_exposure.get(exchange, 0.0) + _number(row.get("current_value"))

    return {
        "provider": "dhan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holding_count": len(holdings),
        "position_count": len(positions),
        "holding_cost_value": round(holding_cost, 2),
        "holding_current_value": round(holding_value, 2) if include_market_values else None,
        "holding_unrealized_pnl": round(unrealized, 2) if include_market_values else None,
        "position_total_pnl": round(position_pnl, 2),
        "exchange_exposure": {key: round(value, 2) for key, value in sector_like_exposure.items()},
        "holdings": enriched_holdings,
        "positions": positions,
        "warnings": _portfolio_warnings(enriched_holdings, include_market_values),
    }


def _request_json(path: str) -> Any:
    token = _access_token()
    if not token:
        raise RuntimeError("Dhan is not configured. Set DHAN_ACCESS_TOKEN in .env or MCP env.")

    request = Request(
        f"{DHAN_BASE_URL}{path}",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": token,
        },
    )
    try:
        with urlopen(request, timeout=12) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Dhan API request failed for {path}: HTTP {exc.code} {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Dhan API request failed for {path}: {exc}") from exc

    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Dhan API returned invalid JSON for {path}") from exc


def _access_token() -> str:
    return os.getenv("DHAN_ACCESS_TOKEN", "").strip()


def _enrich_holding_with_market_value(holding: dict[str, Any]) -> dict[str, Any]:
    row = dict(holding)
    ticker = _holding_to_yfinance_ticker(row)
    row["analysis_ticker"] = ticker
    quantity = _number(row.get("totalQty"))
    average_cost = _number(row.get("avgCostPrice"))
    row["cost_value"] = round(quantity * average_cost, 2)

    if not ticker:
        row["market_data_warning"] = "Could not infer Yahoo Finance ticker from Dhan holding."
        return row

    try:
        prices = get_price_history(ticker, period="5d", interval="1d")
    except Exception as exc:
        logger.info("Failed to fetch market value for Dhan holding %s: %s", ticker, exc)
        row["market_data_warning"] = str(exc)
        return row

    if prices.empty or "close" not in prices.columns:
        row["market_data_warning"] = f"No market price returned for {ticker}."
        return row

    last_close = _number(prices.iloc[-1].get("close"))
    row["last_price"] = round(last_close, 4) if last_close else None
    row["current_value"] = round(quantity * last_close, 2)
    row["unrealized_pnl"] = round(row["current_value"] - row["cost_value"], 2)
    row["unrealized_pnl_percent"] = round((row["unrealized_pnl"] / row["cost_value"]) * 100, 2) if row["cost_value"] else None
    row["market_data_provider"] = prices.attrs.get("provider")
    return row


def _holding_to_yfinance_ticker(holding: dict[str, Any]) -> str | None:
    symbol = str(holding.get("tradingSymbol") or "").strip().upper()
    if not symbol:
        return None
    exchange = str(holding.get("exchange") or "").upper()
    if exchange == "BSE":
        return f"{symbol}.BO"
    return f"{symbol}.NS"


def _number(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _portfolio_warnings(holdings: list[dict[str, Any]], include_market_values: bool) -> list[str]:
    warnings = [
        "Dhan Trading API data is portfolio/account data only; it does not provide promoter holding, fundamentals, analyst suggestions, or latest company news.",
    ]
    if include_market_values:
        missing = [row.get("tradingSymbol") for row in holdings if row.get("market_data_warning")]
        if missing:
            warnings.append(f"Market value could not be calculated for: {', '.join(str(item) for item in missing[:10])}.")
    return warnings

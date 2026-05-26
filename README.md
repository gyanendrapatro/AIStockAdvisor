# Free AI Stock Advisor MCP

Personal daily stock suggestion system for Indian and US stocks with no paid API keys.

> Educational decision-support only. This project does not place trades and does not provide financial advice.

## What it does

- Tracks India and US watchlists
- Calculates a production-grade technical suite across trend, momentum, volatility, volume, support/resistance, and risk quality
- Detects sector rotation by ranking Nifty sector indices versus Nifty 50, then scores stock candidates inside leading sectors
- Pulls price history through Yahoo Finance, with a direct Yahoo Chart fallback and optional Stooq fallback
- Pulls basic fundamentals through Yahoo Finance and enriches US names with official SEC EDGAR facts
- Supports local ownership/governance fundamentals for Indian stocks, including promoter holding, pledge, FII/DII/MF/public holding, and quarterly trends
- Pulls free Yahoo Finance and GDELT headlines, then estimates sentiment locally
- Builds a broad NSE universe from free NSE Total Market constituents with sector/basic-industry metadata for ChartsMaze-style sector and industry analytics
- Runs a deeper company-intelligence check for business areas, sector fit, recent material events, expansion/capex themes, earnings/news evidence, and legal/regulatory/fundraising keywords
- Optionally imports read-only Dhan Trading API account data: profile, holdings, positions, fund limits, and portfolio exposure
- Generates Buy / Hold / Sell / Avoid style suggestions
- Generates deterministic local analyst notes without OpenAI or paid LLM APIs
- Produces daily Markdown reports
- Provides a Streamlit dashboard
- Exposes MCP tools for Codex, Gemini, Claude, Cursor, etc.

## Install

```bash
git clone git@github.com:gyanendrapatro/AIStockAdvisor.git
cd AIStockAdvisor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Use `requirements.txt` for app/runtime only, or `requirements-dev.txt` when you
also want to run tests. Because this repo uses a local `src/` layout, either
prefix commands with `PYTHONPATH=src` or export it once:

```bash
export PYTHONPATH=src
```

Optional `.env` values:

```bash
SEC_USER_AGENT=ai-stock-advisor-mcp/0.1 your-email@example.com
DHAN_ACCESS_TOKEN=
DHAN_CLIENT_ID=
OWNERSHIP_DATA_PATH=ownership.yaml
STOOQ_API_KEY=
```

`data/sectors/**` and the seed universe CSVs are committed so a fresh clone can
run sector, industry, RRG, and top-gainer views immediately. Runtime cache files
such as `data/advisor.sqlite` and `data/daily_refresh_report.json` are generated
locally and are intentionally not committed.

## Run the UI

Start the Streamlit dashboard:

```bash
cd AIStockAdvisor
source .venv/bin/activate
export PYTHONPATH=src
streamlit run src/stock_advisor/dashboard/app.py
```

Open the URL Streamlit prints, usually:

```text
http://localhost:8501
```

Main UI tabs:

- `Stock Scan`: analyze one stock or rank configured watchlists.
- `Sector Analytics`: ChartsMaze-style sector breadth from `data/sectors` plus fresh/cached prices.
- `RRG`: sector relative rotation graph versus Nifty 50.
- `Industry Analytics`: industry performance and stock drill-downs.
- `Market Indices`, `Market Breadth`, `Top Gainers`: broader market dashboards.
- `Universe`: audit stock membership and run the daily NSE/BSE refresh manually.

## Quick CLI Checks

```bash
PYTHONPATH=src .venv/bin/python -m stock_advisor.cli scan --group india --limit 5
PYTHONPATH=src .venv/bin/python -m stock_advisor.cli analyze ANANTRAJ.NS --period 1y --deep-research
PYTHONPATH=src .venv/bin/python -m stock_advisor.cli refresh-data --max-price-symbols 100
```

## Free-Only Data Model

This project intentionally does not use paid APIs.

- Market data: free Yahoo Finance data through `yfinance`
- Market-data fallback: free direct Yahoo Chart endpoint when `yfinance` cookie handling fails
- Optional market-data fallback: Stooq for supported symbols when `STOOQ_API_KEY` is configured
- Fundamentals: free Yahoo Finance fields through `yfinance`
- US filing facts: free SEC EDGAR CompanyFacts API
- Ownership/governance: optional local YAML/CSV file from official NSE/BSE filings, investor-relations pages, or personal exports
- News: free Yahoo Finance headlines through `yfinance`, mixed with free GDELT global-news results
- Company intelligence: free Yahoo/Google News RSS/GDELT evidence search, local categorization, and source-linked event summaries
- Sector rotation: free Yahoo/Nifty sector index or ETF-proxy price history, local relative-strength/breadth scoring, and local stock-candidate ranking
- Broad sector/industry universe: free NSE `NIFTY TOTAL MARKET` constituent snapshot saved to `data/nse_universe.csv`
- Full equity universes: NSE full equity master, BSE active equity master, and combined NSE+BSE master saved under `data/`
- Broker portfolio: optional Dhan Trading API read-only endpoints when `DHAN_ACCESS_TOKEN` is configured
- Sentiment: local keyword-based sentiment estimator
- Commentary: local deterministic analyst note generator

Tradeoff: Yahoo Finance data is free but unofficial. If Yahoo is down, rate-limited, or missing a field for a ticker, the app degrades gracefully and surfaces warnings in CLI, Streamlit, MCP responses, and reports.

Refresh the stock universe when you want current sector/basic-industry metadata:

```bash
PYTHONPATH=src .venv/bin/python scripts/refresh_universe.py
PYTHONPATH=src .venv/bin/python scripts/refresh_universe.py --universe full
PYTHONPATH=src .venv/bin/python scripts/refresh_universe.py --universe bse
PYTHONPATH=src .venv/bin/python scripts/refresh_universe.py --universe india
```

Run the full daily refresh after Indian market close to update public NSE files,
pull the latest NSE/BSE bhavcopy EOD rows into SQLite, and warm the historical
price cache used by the dashboard:

```bash
PYTHONPATH=src .venv/bin/python scripts/daily_refresh.py
PYTHONPATH=src .venv/bin/python -m stock_advisor.cli refresh-data
```

The daily job writes `data/daily_refresh_report.json`. For a faster debug run:

```bash
PYTHONPATH=src .venv/bin/python scripts/daily_refresh.py --max-price-symbols 100 --skip-full-nse-universe
```

The Streamlit Universe tab can audit `broad`, `full_nse`, `full_bse`, and
`all_india`. Sector and industry analytics default to full NSE because NSE
provides cleaner sector/basic-industry metadata. The combined NSE+BSE universe
keeps BSE-only rows too, but those can remain unclassified until a public sector
source is available for that ISIN. MCP tools expose the same path:

- `list_stock_universe`
- `refresh_stock_universe`
- `refresh_full_stock_universe`
- `refresh_bse_stock_universe`
- `refresh_india_stock_universe`
- `refresh_latest_exchange_eod_cache`
- `run_daily_market_data_refresh`
- `get_sector_analytics`
- `list_industry_definitions`
- `get_industry_analytics`
- `rank_industry_stocks`

For faster unit-style checks or the older hand-picked baskets, pass
`universe="local"`.

SEC EDGAR requests should include a descriptive user agent. Set this in `.env`:

```bash
SEC_USER_AGENT=ai-stock-advisor-mcp/0.1 your-email@example.com
```

Stooq is optional because its CSV endpoint now requires a free API key in some environments:

```bash
STOOQ_API_KEY=
```

Ownership/governance data is optional and local. Copy `ownership.example.yaml` to `ownership.yaml`, or point to a CSV/YAML file:

```bash
OWNERSHIP_DATA_PATH=ownership.yaml
```

Supported ownership fields:

- `promoter_holding`
- `promoter_pledge`
- `fii_holding`
- `dii_holding`
- `mf_holding`
- `public_holding`
- `shareholder_count`

The app computes quarter-over-quarter changes when at least two quarters are present. Screener.in can be used manually as a reference or export source, but the production app does not scrape Screener directly.

Dhan is optional and read-only in this project. It can import your holdings,
positions, profile, and fund limits from the free Trading API. It does not
provide promoter holding, company fundamentals, recommendations, or news. Dhan
Web-generated access tokens are valid for 24 hours:

```bash
DHAN_ACCESS_TOKEN=
DHAN_CLIENT_ID=
```

Use one of these standard credential paths:

1. Local project `.env`: copy `.env.example` to `.env` and fill
   `DHAN_ACCESS_TOKEN` and `DHAN_CLIENT_ID`. The MCP server loads this file at
   startup. `.env` is git-ignored.
2. MCP client environment: pass `DHAN_ACCESS_TOKEN` and `DHAN_CLIENT_ID` in the
   MCP server config `env` block or the client secret manager.
3. Shell environment for local verification:

```bash
export DHAN_ACCESS_TOKEN="fresh-token-from-dhan"
export DHAN_CLIENT_ID="your-dhan-client-id"
```

Do not pass Dhan tokens as MCP tool arguments and do not commit them to source
control. Rotate the token if it was pasted into chat or logs.

The MCP deliberately does not expose order placement, order modification, exit
position, or trade execution tools.

## MCP Server

The MCP server exposes the same analysis engine to agents such as Codex, Claude,
Cursor, Gemini-compatible clients, and any tool runner that supports MCP.

Start it manually:

```bash
cd AIStockAdvisor
source .venv/bin/activate
PYTHONPATH=src python -m stock_advisor.mcp.server
```

When configured in an MCP client, the client starts this command automatically
when an agent needs a tool call. You do not normally keep a separate terminal
open unless you are debugging.

Example MCP tools:

- `server_info`
- `health_check`
- `get_watchlist`
- `analyze_stock`
- `research_stock`
- `rank_watchlist`
- `compare_stocks`
- `get_stock_profile`
- `get_stock_fundamentals`
- `get_stock_news`
- `get_latest_technical_indicators`
- `get_chart_patterns`
- `get_analyst_insights`
- `get_stock_events`
- `discover_market_themes`
- `list_stock_universe`
- `refresh_stock_universe`
- `refresh_full_stock_universe`
- `refresh_bse_stock_universe`
- `refresh_india_stock_universe`
- `refresh_latest_exchange_eod_cache`
- `run_daily_market_data_refresh`
- `list_sector_definitions`
- `get_sector_rotation`
- `rank_sector_stocks`
- `discover_sector_opportunities`
- `run_sector_rotation_workflow`
- `get_sector_analytics`
- `list_industry_definitions`
- `get_industry_analytics`
- `rank_industry_stocks`
- `get_company_intelligence`
- `get_ownership_fundamentals`
- `get_dhan_profile`
- `get_dhan_holdings`
- `get_dhan_positions`
- `get_dhan_fund_limits`
- `get_dhan_portfolio_stocks`
- `get_dhan_portfolio_summary`
- `analyze_dhan_portfolio`
- `get_price_history`
- `get_historical_prices`
- `market_snapshot`
- `generate_daily_report`

Verify MCP wiring locally:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_mcp_setup.py
PYTHONPATH=src .venv/bin/python scripts/check_mcp_setup.py --dhan
PYTHONPATH=src .venv/bin/python scripts/check_mcp_setup.py --portfolio
```

The verifier starts the server through an MCP client and calls MCP tools. It
does not print your Dhan token.

## Configure MCP In Agents

For MCP clients that accept JSON server config, add this `mcpServers` block.
Update the paths if you cloned somewhere else:

```json
{
  "mcpServers": {
    "ai-stock-advisor": {
      "command": "/absolute/path/to/AIStockAdvisor/.venv/bin/python",
      "args": ["-m", "stock_advisor.mcp.server"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/AIStockAdvisor/src",
        "SEC_USER_AGENT": "ai-stock-advisor-mcp/0.1 your-email@example.com",
        "OWNERSHIP_DATA_PATH": "ownership.yaml",
        "DHAN_ACCESS_TOKEN": "",
        "DHAN_CLIENT_ID": ""
      }
    }
  }
}
```

For this local machine, the current repo path is:

```json
{
  "mcpServers": {
    "ai-stock-advisor": {
      "command": "/Users/gypatro/Downloads/ai-stock-advisor/.venv/bin/python",
      "args": ["-m", "stock_advisor.mcp.server"],
      "env": {
        "PYTHONPATH": "/Users/gypatro/Downloads/ai-stock-advisor/src",
        "SEC_USER_AGENT": "ai-stock-advisor-mcp/0.1 your-email@example.com",
        "OWNERSHIP_DATA_PATH": "ownership.yaml",
        "DHAN_ACCESS_TOKEN": "",
        "DHAN_CLIENT_ID": ""
      }
    }
  }
}
```

Where to put it depends on the client:

- Codex/agent clients: add the server through the app's MCP/server settings or the client config file.
- Claude Desktop: add it under `mcpServers` in the Claude Desktop config.
- Cursor: add it to the project/global MCP config.
- Other agents: use the same command, args, and env block wherever that client accepts MCP servers.

After configuration, ask the agent to call `server_info` or `health_check` from
`ai-stock-advisor`. If that works, the agent can call tools such as
`analyze_stock`, `get_sector_analytics`, `get_relative_rotation_graph`,
`get_top_gainers`, and `analyze_dhan_portfolio`.

Portfolio analysis should be done through MCP tool calls, not by importing the
Python modules directly from the agent. The intended flow is:

```text
agent -> MCP client -> ai-stock-advisor MCP server -> Dhan/read-only data + analysis tools
```

For a connected Dhan account, call `analyze_dhan_portfolio` from the MCP client.
That tool fetches holdings through Dhan, analyzes each equity holding, preserves
cash-like holdings such as `LIQUIDCASE`, and returns add/hold/reduce buckets.

`mcp-config.example.json` contains the same config shape and can be copied into
any MCP client that accepts JSON config.

Useful CLI examples:

```bash
PYTHONPATH=src python -m stock_advisor.cli analyze AAPL --period 3mo --no-news
PYTHONPATH=src python -m stock_advisor.cli analyze ANANTRAJ.NS --period 1y --deep-research
PYTHONPATH=src python -m stock_advisor.cli scan --group us --limit 5 --json
PYTHONPATH=src python -m stock_advisor.cli report --group india
```

Run the offline test suite:

```bash
PYTHONPATH=src pytest
```

## Watchlists

Edit `watchlists.yaml`.

```yaml
india:
  - RELIANCE.NS
  - TCS.NS
us:
  - AAPL
  - MSFT
```

## Suggested scoring

```text
35% technicals
20% fundamentals
20% news/sentiment
15% risk
10% momentum/liquidity
```

Ownership/governance data feeds into both the fundamental and risk scores. High or increasing promoter ownership, no pledge, and meaningful institutional ownership help the score. High promoter pledge or falling promoter holding adds risk.

## Technical Indicator Coverage

- Trend: SMA/EMA 20, 50, 100, 200; MACD; ADX/DMI; moving-average crossover signals; trend alignment
- Momentum: RSI, stochastic %K/%D, ROC, CCI, Williams %R
- Volatility: ATR, Bollinger Bands, Keltner Channels, Donchian Channels, annualized 20-day volatility
- Volume: volume ratio, dollar volume, OBV, MFI, VWAP, accumulation/distribution line, CMF
- Support/resistance: 20-day swing high/low, classic pivot levels, 52-week high/low distance
- Risk/quality: max drawdown, downside volatility, gap risk, zero-volume days, liquidity score

## Roadmap

- Broker portfolio import through Zerodha/Kite MCP or CSV
- Better free Indian fundamentals provider
- Backtesting module
- Telegram/email daily delivery
- Optional local-LLM commentary through Ollama, while keeping paid APIs out

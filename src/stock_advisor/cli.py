from __future__ import annotations
import argparse
import json
from rich.console import Console
from rich.table import Table
from stock_advisor.analysis.pipeline import analyze_stock, rank_watchlist
from stock_advisor.data.daily_refresh import run_daily_market_data_refresh
from stock_advisor.reporting.daily_report import generate_daily_report

console = Console()


def main():
    parser = argparse.ArgumentParser(description="AI Stock Advisor")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("ticker")
    p_analyze.add_argument("--period", default=None)
    p_analyze.add_argument("--interval", default=None)
    p_analyze.add_argument("--no-news", action="store_true")
    p_analyze.add_argument("--deep-research", action="store_true", help="Include broader company intelligence and event research")
    p_scan = sub.add_parser("scan")
    p_scan.add_argument("--group", choices=["india", "us"], default=None)
    p_scan.add_argument("--period", default=None)
    p_scan.add_argument("--interval", default=None)
    p_scan.add_argument("--limit", type=int, default=None)
    p_scan.add_argument("--no-news", action="store_true")
    p_scan.add_argument("--with-intelligence", action="store_true", help="Include broader company intelligence during scans; slower")
    p_scan.add_argument("--json", action="store_true")
    p_report = sub.add_parser("report")
    p_report.add_argument("--group", choices=["india", "us"], default=None)
    p_refresh = sub.add_parser("refresh-data", aliases=["daily-refresh"], help="Refresh public NSE/BSE files and local OHLCV cache")
    p_refresh.add_argument("--universe", choices=["broad", "full_nse", "full_bse", "all_india"], default="full_nse")
    p_refresh.add_argument("--period", default="2y")
    p_refresh.add_argument("--interval", choices=["1d", "1wk", "1mo"], default="1d")
    p_refresh.add_argument("--max-universe-symbols", type=int, default=None)
    p_refresh.add_argument("--max-price-symbols", type=int, default=None)
    p_refresh.add_argument("--chunk-size", type=int, default=80)
    p_refresh.add_argument("--retry-attempts", type=int, default=2)
    p_refresh.add_argument("--report-path", default=None)
    p_refresh.add_argument("--skip-universe-refresh", action="store_true")
    p_refresh.add_argument("--skip-broad-universe", action="store_true")
    p_refresh.add_argument("--skip-full-nse-universe", action="store_true")
    p_refresh.add_argument("--include-bse-universe", action="store_true")
    p_refresh.add_argument("--include-india-universe", action="store_true")
    p_refresh.add_argument("--skip-exchange-eod", action="store_true")
    p_refresh.add_argument("--skip-price-cache", action="store_true")
    p_refresh.add_argument("--no-force-refresh", action="store_true")
    args = parser.parse_args()

    if args.cmd == "analyze":
        console.print_json(
            json.dumps(
                analyze_stock(
                    args.ticker,
                    period=args.period,
                    interval=args.interval,
                    include_news=not args.no_news,
                    include_intelligence=args.deep_research,
                )
            )
        )
    elif args.cmd == "scan":
        rows = rank_watchlist(
            args.group,
            limit=args.limit,
            period=args.period,
            interval=args.interval,
            include_news=not args.no_news,
            include_intelligence=args.with_intelligence,
        )
        if args.json:
            console.print_json(json.dumps(rows))
            return
        table = Table(title="Stock Suggestions")
        for col in ["Ticker", "Signal", "Score", "Confidence", "Data", "Reasons"]:
            table.add_column(col)
        for r in rows:
            metadata = r.get("metadata", {})
            data_state = "ok" if metadata.get("has_price_history") else "limited"
            table.add_row(
                str(r.get("ticker")),
                str(r.get("signal")),
                str(r.get("final_score")),
                str(r.get("confidence", "")),
                data_state,
                "; ".join(r.get("reasons", [])[:2]),
            )
        console.print(table)
    elif args.cmd == "report":
        path = generate_daily_report(args.group)
        console.print(f"Report written: {path}")
    elif args.cmd in {"refresh-data", "daily-refresh"}:
        console.print_json(
            json.dumps(
                run_daily_market_data_refresh(
                    refresh_universes=not args.skip_universe_refresh,
                    refresh_broad_universe=not args.skip_broad_universe,
                    refresh_full_nse_universe=not args.skip_full_nse_universe,
                    refresh_bse_universe=args.include_bse_universe,
                    refresh_india_universe=args.include_india_universe,
                    warm_price_cache=not args.skip_price_cache,
                    refresh_exchange_eod=not args.skip_exchange_eod,
                    warm_universe=args.universe,
                    period=args.period,
                    interval=args.interval,
                    max_universe_symbols=args.max_universe_symbols,
                    max_price_symbols=args.max_price_symbols,
                    chunk_size=args.chunk_size,
                    retry_attempts=args.retry_attempts,
                    force_refresh_prices=not args.no_force_refresh,
                    report_path=args.report_path,
                ),
                default=str,
            )
        )

if __name__ == "__main__":
    main()

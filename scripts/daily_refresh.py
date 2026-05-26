from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_advisor.data.daily_refresh import run_daily_market_data_refresh  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily refresh for public NSE/BSE files and local OHLCV cache.")
    parser.add_argument("--universe", default="full_nse", choices=["broad", "full_nse", "full_bse", "all_india"])
    parser.add_argument("--period", default="2y", help="Historical period to warm for analytics.")
    parser.add_argument("--interval", default="1d", choices=["1d", "1wk", "1mo"])
    parser.add_argument("--max-universe-symbols", type=int, default=None, help="Cap full NSE universe refresh for debugging.")
    parser.add_argument("--max-price-symbols", type=int, default=None, help="Cap price-cache warming for debugging.")
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument("--report-path", default=str(ROOT / "data" / "daily_refresh_report.json"))
    parser.add_argument("--skip-universe-refresh", action="store_true")
    parser.add_argument("--skip-broad-universe", action="store_true")
    parser.add_argument("--skip-full-nse-universe", action="store_true")
    parser.add_argument("--include-bse-universe", action="store_true")
    parser.add_argument("--include-india-universe", action="store_true")
    parser.add_argument("--skip-exchange-eod", action="store_true")
    parser.add_argument("--skip-price-cache", action="store_true")
    parser.add_argument("--no-force-refresh", action="store_true", help="Use fresh cache when available instead of forcing provider refresh.")
    args = parser.parse_args()

    result = run_daily_market_data_refresh(
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
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

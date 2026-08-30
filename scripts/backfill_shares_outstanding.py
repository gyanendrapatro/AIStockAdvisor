from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_advisor.data.market_data import backfill_shares_outstanding  # noqa: E402
from stock_advisor.data.screener import DEFAULT_REQUEST_DELAY_SECONDS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill shares_outstanding, NSE shareholding-pattern XBRL first (official, free, "
            "bulk -- covers every NSE-listed ticker plus any BSE ticker whose company is also "
            "NSE-listed, via ISIN). screener.in is an opt-in fallback (--include-screener-fallback) "
            "for the residual BSE-only tail with no NSE listing at all -- off by default given its "
            "terms restrict bulk copying to personal, non-commercial use. Not part of any automatic "
            "refresh -- run this manually, then market cap recomputes itself daily for free from "
            "the already-cached close price (see recompute_market_cap_from_shares_outstanding, "
            "called automatically at the end of every price-cache refresh)."
        )
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process this many candidate tickers (for testing).")
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS, help="Seconds to wait between per-ticker requests."
    )
    parser.add_argument(
        "--tickers", nargs="*", default=None, help="Specific tickers instead of scanning for missing/upgradeable ones."
    )
    parser.add_argument(
        "--force-refresh-all",
        action="store_true",
        help="Re-check every active ticker regardless of current value/source, not just missing/screener-sourced ones.",
    )
    parser.add_argument(
        "--include-screener-fallback",
        action="store_true",
        help="Also run the screener.in-derived fallback for tickers NSE can't resolve. Off by default.",
    )
    args = parser.parse_args()

    def _on_progress(update: dict) -> None:
        completed = update.get("completed", 0)
        total = update.get("total", 0)
        if total and (completed % 25 == 0 or completed == total):
            print(f"[{update.get('phase')}] {completed}/{total}")

    result = backfill_shares_outstanding(
        args.tickers,
        force_refresh_all=args.force_refresh_all,
        include_screener_fallback=args.include_screener_fallback,
        delay_seconds=args.delay,
        limit=args.limit,
        progress_callback=_on_progress,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

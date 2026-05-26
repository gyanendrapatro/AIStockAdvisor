from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_advisor.data.universe import (  # noqa: E402
    DEFAULT_NSE_INDEX_NAME,
    refresh_bse_stock_universe,
    refresh_full_stock_universe,
    refresh_india_stock_universe,
    refresh_stock_universe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh stock universe CSVs from free public NSE/BSE data.")
    parser.add_argument(
        "--universe",
        choices=["broad", "full", "bse", "india"],
        default="broad",
        help="broad = NIFTY Total Market. full = NSE equity master + quote metadata. bse/india = Dhan public NSE+BSE instrument master.",
    )
    parser.add_argument("--index", default=DEFAULT_NSE_INDEX_NAME, help="NSE index universe to fetch, for example NIFTY TOTAL MARKET or NIFTY 500.")
    parser.add_argument("--output", default=str(ROOT / "data" / "nse_universe.csv"), help="CSV output path.")
    parser.add_argument("--max-symbols", type=int, default=None, help="For full universe refresh, cap the number of NSE symbols fetched.")
    parser.add_argument("--symbols", nargs="*", default=None, help="For full universe refresh, fetch only these NSE symbols.")
    args = parser.parse_args()
    if args.universe == "full":
        output = args.output
        if output == str(ROOT / "data" / "nse_universe.csv"):
            output = str(ROOT / "data" / "nse_full_universe.csv")
        result = refresh_full_stock_universe(path=output, max_symbols=args.max_symbols, symbols=args.symbols)
    elif args.universe == "bse":
        output = args.output
        if output == str(ROOT / "data" / "nse_universe.csv"):
            output = str(ROOT / "data" / "bse_full_universe.csv")
        result = refresh_bse_stock_universe(path=output)
    elif args.universe == "india":
        output = args.output
        if output == str(ROOT / "data" / "nse_universe.csv"):
            output = str(ROOT / "data" / "india_full_universe.csv")
        result = refresh_india_stock_universe(path=output)
    else:
        result = refresh_stock_universe(index_name=args.index, path=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

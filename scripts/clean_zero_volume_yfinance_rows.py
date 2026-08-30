from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_advisor.data.market_data import clean_zero_volume_yfinance_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time cleanup for price_history_cache rows written before _store_price_history "
            "started discarding Yahoo's zero-volume placeholder rows (yf.download regularly "
            "carries a thinly-traded security's last known close forward as a synthetic "
            "zero-volume row instead of returning nothing). Deletes those rows and recomputes "
            "price_cache_meta/live_metrics/price_data for every affected ticker from what's real. "
            "Defaults to a dry run -- pass --apply to actually delete."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the fake rows and update meta/live_metrics/price_data. Without this, only reports counts.",
    )
    args = parser.parse_args()

    result = clean_zero_volume_yfinance_rows(dry_run=not args.apply)
    print(json.dumps(result, indent=2))
    if result["dry_run"]:
        print("\nDry run only -- re-run with --apply to actually delete these rows.")


if __name__ == "__main__":
    main()

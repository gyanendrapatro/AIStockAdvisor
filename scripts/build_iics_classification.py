from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_advisor.data.iics_classification import (  # noqa: E402
    DEFAULT_REQUEST_DELAY_SECONDS,
    build_iics_classification_mapping,
    fetch_iics_taxonomy_pdf,
    parse_iics_taxonomy_pdf,
)

CSV_COLUMNS = [
    "ticker",
    "macro_sector_code",
    "macro_sector",
    "sector_code",
    "sector",
    "industry_code",
    "industry",
    "basic_industry_code",
    "basic_industry",
    "synced_at",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time backfill: build the full NSE+BSE company -> official IICS classification "
            "mapping (Macro-Economic Sector / Sector / Industry / Basic Industry) from BSE's "
            "published code-tree PDF plus screener.in's per-basic-industry listing pages "
            "(~197 requests total for the whole universe -- see iics_classification.py). Writes "
            "data/iics_classification.csv, which market_data.load_iics_classification_seed then "
            "replays into any advisor.sqlite (yours or a fresh clone's) with zero network calls -- "
            "this script is the only thing that ever talks to screener.in for this data, and it's "
            "meant to run manually, occasionally, not as part of any automatic refresh."
        )
    )
    parser.add_argument("--limit", type=int, default=None, help="Only walk this many basic-industry leaves (for testing).")
    parser.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS, help="Seconds to wait between per-basic-industry requests.")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "iics_classification.csv", help="Where to write the resulting CSV."
    )
    args = parser.parse_args()

    print("Downloading BSE's IICS taxonomy PDF...")
    pdf_bytes = fetch_iics_taxonomy_pdf()
    taxonomy_rows = parse_iics_taxonomy_pdf(pdf_bytes)
    print(f"Parsed {len(taxonomy_rows)} basic-industry leaves from the taxonomy PDF.")
    if args.limit is not None:
        taxonomy_rows = taxonomy_rows[: max(0, args.limit)]
        print(f"--limit applied: walking {len(taxonomy_rows)} leaves.")

    def _on_progress(update: dict) -> None:
        completed = update.get("completed", 0)
        total = update.get("total", 0)
        if total and (completed % 10 == 0 or completed == total):
            print(f"[{completed}/{total}] {update.get('basic_industry_code')}")

    result = build_iics_classification_mapping(taxonomy_rows, delay_seconds=args.delay, progress_callback=_on_progress)
    rows = result["rows"]
    synced_at = datetime.now(timezone.utc).isoformat()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "synced_at": synced_at})

    summary = {
        "basic_industries_walked": result["basic_industries_walked"],
        "basic_industries_failed": result["basic_industries_failed"],
        "incomplete_basic_industries": result["incomplete_basic_industries"],
        "tickers_resolved": result["tickers_resolved"],
        "output_path": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    print(
        "\nDone. This CSV is what gets committed to the repo and seeded into any advisor.sqlite "
        "(see market_data.load_iics_classification_seed) -- no further scraping needed for anyone "
        "who clones the repo."
    )


if __name__ == "__main__":
    main()

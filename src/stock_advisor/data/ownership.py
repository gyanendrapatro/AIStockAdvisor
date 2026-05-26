from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from stock_advisor.config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

OWNERSHIP_FIELDS = {
    "promoter_holding",
    "promoter_pledge",
    "fii_holding",
    "dii_holding",
    "mf_holding",
    "public_holding",
    "shareholder_count",
}


def get_ownership_fundamentals(ticker: str, path: str | Path | None = None) -> dict[str, Any]:
    """Return optional ownership/governance fundamentals from local YAML or CSV data."""
    rows = _load_ownership_rows(path)
    ticker_rows = [row for row in rows if _normalize_ticker(row.get("ticker")) == _normalize_ticker(ticker)]
    if not ticker_rows:
        return {}

    ticker_rows = sorted(ticker_rows, key=lambda row: str(row.get("quarter") or ""))
    latest = ticker_rows[-1]
    previous = ticker_rows[-2] if len(ticker_rows) > 1 else {}

    result: dict[str, Any] = {
        "_sources": ["local_ownership"],
        "ownership_quarter": latest.get("quarter"),
        "ownership_source": latest.get("source") or "local",
    }
    for field in OWNERSHIP_FIELDS:
        value = _to_number(latest.get(field))
        if value is not None:
            result[field] = value
        previous_value = _to_number(previous.get(field))
        if value is not None and previous_value is not None:
            result[f"{field}_qoq_change"] = round(value - previous_value, 4)

    return result


def ownership_data_file_exists(path: str | Path | None = None) -> bool:
    """Return whether the configured local ownership data file exists."""
    return _resolve_path(path).exists()


def _load_ownership_rows(path: str | Path | None = None) -> list[dict[str, Any]]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        return []
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml_rows(resolved)
    if resolved.suffix.lower() == ".csv":
        return _load_csv_rows(resolved)
    logger.warning("Unsupported ownership data file format: %s", resolved)
    return []


def _load_yaml_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning("Failed to read ownership data from %s: %s", path, exc)
        return []

    rows: list[dict[str, Any]] = []
    for ticker, value in payload.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append({"ticker": ticker, **item})
        elif isinstance(value, dict):
            history = value.get("history")
            if isinstance(history, list):
                for item in history:
                    if isinstance(item, dict):
                        rows.append({"ticker": ticker, **item})
            else:
                rows.append({"ticker": ticker, **value})
    return rows


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        logger.warning("Failed to read ownership CSV from %s: %s", path, exc)
        return []


def _resolve_path(path: str | Path | None) -> Path:
    configured = path or os.getenv("OWNERSHIP_DATA_PATH") or "ownership.yaml"
    resolved = Path(configured)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _normalize_ticker(value: Any) -> str:
    return str(value or "").strip().upper().removesuffix(".NS").removesuffix(".BO")


def _to_number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        text = str(value).strip().replace("%", "").replace(",", "")
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number

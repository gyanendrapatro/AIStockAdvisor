from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(value: str, default: str) -> Path:
    raw = Path(value or default)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw

class Settings(BaseModel):
    default_period: str = os.getenv("DEFAULT_PERIOD", "6mo")
    default_interval: str = os.getenv("DEFAULT_INTERVAL", "1d")
    report_dir: Path = _project_path(os.getenv("REPORT_DIR", "reports"), "reports")
    db_path: Path = _project_path(os.getenv("DB_PATH", "data/advisor.sqlite"), "data/advisor.sqlite")
    # Floor date for the price-history cache backfill (fill_price_cache_for_universe). Only the
    # sidebar's "Refresh price cache" button, the daily_refresh CLI, and the MCP warm_price_history_cache
    # tool ever fetch/write price history; all other code just reads whatever is cached from here on.
    price_history_start_date: str = os.getenv("PRICE_HISTORY_START_DATE", "2024-01-01")

settings = Settings()

def load_watchlists(path: str | Path | None = None) -> dict[str, list[str]]:
    p = Path(path) if path is not None else PROJECT_ROOT / "watchlists.yaml"
    if not p.exists():
        return {"india": [], "us": []}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    watchlists = {"india": [], "us": []}
    for key, values in data.items():
        if values is None:
            watchlists[str(key)] = []
            continue
        watchlists[str(key)] = list(dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip()))
    return watchlists

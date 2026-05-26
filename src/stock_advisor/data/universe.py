from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote
import csv
import json
import time

import pandas as pd
import requests

from stock_advisor.config.settings import PROJECT_ROOT


DEFAULT_NSE_INDEX_NAME = "NIFTY TOTAL MARKET"
DEFAULT_UNIVERSE_PATH = PROJECT_ROOT / "data" / "nse_universe.csv"
DEFAULT_FULL_NSE_UNIVERSE_PATH = PROJECT_ROOT / "data" / "nse_full_universe.csv"
DEFAULT_FULL_BSE_UNIVERSE_PATH = PROJECT_ROOT / "data" / "bse_full_universe.csv"
DEFAULT_ALL_INDIA_UNIVERSE_PATH = PROJECT_ROOT / "data" / "india_full_universe.csv"
DEFAULT_SECTOR_TAXONOMY_PATH = PROJECT_ROOT / "data" / "sectors"
NSE_INDEX_API = "https://www.nseindia.com/api/equity-stockIndices?index={index_name}"
NSE_EQUITY_MASTER_CSV = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_QUOTE_EQUITY_API = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"
NSE_HOME = "https://www.nseindia.com"
DHAN_SCRIP_MASTER_DETAILED_CSV = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
SECTOR_TAXONOMY_SOURCE = "local_sector_csv_taxonomy"
TAXONOMY_FOLDER_SECTOR_ALIASES = {
    "CapitalGoods": "Capital Goods",
}
TAXONOMY_PREFERRED_SECTOR_BY_INDUSTRY = {
    "breweries & distilleries": "FMCG",
    "medical equipment & supplies": "Healthcare",
    "other telecom services": "Telecommunication",
    "railways": "Capital Goods",
    "telecom - equipment & accessories": "Telecommunication",
    "tv broadcasting & software": "Media Entertainment & Publication",
}
CHARTMAZE_METAL_INDUSTRY_OVERRIDES = {
    # ChartsMaze shows Aeroflex under Metal Fabrication even though NSE tags it
    # as Iron & Steel Products.
    "AEROFLEX": "Metal Fabrication",
    "AEROENTER": "Metal Fabrication",
}
CHARTMAZE_BASIC_INDUSTRY_ALIASES = {
    "2 3 wheelers": "Auto Manufacturers",
    "abrasives and bearings": "Industrial Products & Manufacturing",
    "advertising and media agencies": "Advertising",
    "airline": "Airlines",
    "airport and airport services": "Airlines",
    "animal feed": "Agro Products",
    "asset management company": "Asset Management",
    "auto components and equipments": "Auto Ancilaries",
    "auto ancillaries": "Auto Ancilaries",
    "business process outsourcing (bpo) knowledge process outsourcing (kpo)": "Software Services",
    "carbon black": "Chemicals-Basic",
    "castings and forgings": "Industrial Products & Manufacturing",
    "cement and cement products": "Cement",
    "ceramics": "Construction Products Miscallaneous",
    "cigarettes and tobacco products": "Tobacco",
    "coal": "Coal Products",
    "commercial vehicles": "Auto Manufacturers",
    "commodity chemicals": "Chemicals-Basic",
    "compressors pumps and diesel engines": "Pumps",
    "computers software and consulting": "Software Services",
    "computers hardware and equipments": "Computer Hardware",
    "construction vehicles": "Auto Manufacturers",
    "consumer electronics": "EMS",
    "cycles": "Auto Manufacturers",
    "data processing services": "Software Services",
    "dealers���commercial vehicles tractors construction vehicles": "Auto Dealer",
    "depositories clearing houses and other intermediaries": "Investment Banking & Broking",
    "digital entertainment": "Media & Entertainment",
    "diversified": "Diversified Operations",
    "diversified fmcg": "Food Products",
    "dredging": "Civil Construction",
    "dyes and pigments": "Chemicals Specialty",
    "e learning": "Education Services",
    "e retail e commerce": "E- Commerce",
    "edible oil": "Edible Oils & Solvent Extraction",
    "education": "Education Services",
    "exchange and data platform": "Investment Banking & Broking",
    "explosives": "Chemicals Specialty",
    "film production distribution and exhibition": "Film Production & Distribution",
    "financial institution": "Financial Services-Specialty",
    "financial products distributor": "Financial Services-Specialty",
    "financial technology (fintech)": "Financial Services-Specialty",
    "forest products": "Wood Products",
    "furniture home furnishing": "Home Furnishing",
    "garments and apparels": "Apparels",
    "gas transmission marketing": "Gas Distribution",
    "gems jewellery and watches": "Jewellery",
    "glass consumer": "Glass",
    "glass industrial": "Glass",
    "granites and marbles": "Construction Products Miscallaneous",
    "hospital": "Hospitals",
    "hotels resorts": "Hotels",
    "hotels and resorts": "Hotels",
    "healthcare service provider": "Medical-Diversified",
    "heavy electrical equipment": "Electrical - Power Equipment",
    "housing finance company": "Housing Finance",
    "household products": "Personal Care",
    "houseware": "Household Appliances",
    "it enabled services": "Software Services",
    "industrial equipments": "Industrial Products & Manufacturing",
    "industrial gases": "Industrial Products & Manufacturing",
    "industrial products": "Industrial Products & Manufacturing",
    "insurance distributors": "Financial Services-Specialty",
    "integrated power utilities": "Power Generation & Distribution",
    "internet and catalogue retail": "E- Commerce",
    "investment company": "Finance & Investment",
    "jute and jute products": "Textiles",
    "lpg cng png lng supplier": "Gas Distribution",
    "leather and leather products": "Footwear",
    "logistics solution provider": "Logistics",
    "meat products including poultry": "Food Products",
    "microfinance institutions": "Financial Services-Specialty",
    "non banking financial company (nbfc)": "NBFC",
    "offshore support solution drilling": "Oil & Gas Drilling",
    "oil equipment and services": "Oil & Gas-Field Services",
    "oil storage and transportation": "Oil & Gas-Field Services",
    "other agricultural products": "Agro Products",
    "other bank": "Private Banks",
    "other capital market related services": "Investment Banking & Broking",
    "other construction materials": "Construction Products Miscallaneous",
    "other consumer services": "Diversified Commercial Services",
    "other electrical equipment": "Electrical Miscallaneous",
    "other financial services": "Financial Services-Specialty",
    "other food products": "Food Products",
    "other industrial products": "Industrial Products & Manufacturing",
    "other textile products": "Textiles",
    "packaged foods": "Food Products",
    "passenger cars and utility vehicles": "Auto Manufacturers",
    "pesticides and agrochemicals": "Agro chemicals",
    "pharmacy retail": "Medical-Diversified",
    "plastic products consumer": "Chemicals-Plastics",
    "plastic products industrial": "Chemicals-Plastics",
    "port and port services": "Logistics",
    "power transmission": "Power Generation & Distribution",
    "power distribution": "Power Generation & Distribution",
    "power generation": "Power Generation & Distribution",
    "power trading": "Power Generation & Distribution",
    "printing and publication": "Print Media",
    "printing inks": "Chemicals Specialty",
    "private sector bank": "Private Banks",
    "public sector bank": "PSU Banks",
    "railway wagons": "Railways",
    "ratings": "Credit Rating Agencies",
    "real estate related services": "Real Estate",
    "refineries and marketing": "Oil & Gas-Refining & Marketing",
    "residential commercial projects": "Real Estate",
    "residential commercial sez project": "Real Estate",
    "road assets toll annuity hybrid annuity": "Civil Construction",
    "road transport": "Logistics",
    "rubber": "Tyres & Rubber Products",
    "sanitary ware": "Construction Products Miscallaneous",
    "seafood": "Food Products",
    "ship building and allied services": "Ship building & Allied services",
    "shipping": "Logistics",
    "specialty chemicals": "Chemicals Specialty",
    "stockbroking and allied": "Investment Banking & Broking",
    "telecom cellular and fixed line services": "Telecom - Cellular & Fixed line services",
    "tour travel related services": "Tour & Travel services",
    "tractors": "Auto Manufacturers",
    "trading": "Diversified Commercial Services",
    "trading and distributors": "Diversified Commercial Services",
    "trading auto components": "Auto Ancilaries",
    "trading chemicals": "Chemicals Specialty",
    "trading coal": "Coal Products",
    "trading gas": "Gas Distribution",
    "trading textile products": "Textiles",
    "transport related services": "Logistics",
    "tv broadcasting and software production": "TV Broadcasting & Software",
    "web based media and service": "Web Services",
    "wellness": "Medical-Diversified",
}

UNIVERSE_COLUMNS = [
    "ticker",
    "symbol",
    "name",
    "isin",
    "sector",
    "industry",
    "basic_industry",
    "index_name",
    "source",
    "active",
    "series",
    "free_float_market_cap",
    "last_price",
    "year_high",
    "year_low",
    "near_52w_high_pct",
    "return_30d_pct",
    "return_365d_pct",
    "refreshed_at",
    "exchange",
    "security_id",
    "nse_ticker",
    "bse_ticker",
    "nse_security_id",
    "bse_security_id",
    "data_quality",
    "classification_source",
]


def load_stock_universe(
    *,
    universe: str = "broad",
    path: str | Path | None = None,
    refresh: bool = False,
    index_name: str = DEFAULT_NSE_INDEX_NAME,
    max_stocks: int | None = None,
) -> pd.DataFrame:
    """Load the local broad NSE stock universe, optionally refreshing it first."""
    normalized = _normalize_universe_name(universe)
    if normalized == "local":
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)

    universe_path = Path(path) if path is not None else _default_universe_path(normalized)
    if normalized == "full_nse":
        if refresh:
            try:
                refresh_full_stock_universe(path=universe_path, max_symbols=max_stocks)
            except Exception:
                if not universe_path.exists():
                    return pd.DataFrame(columns=UNIVERSE_COLUMNS)
        elif not universe_path.exists():
            return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    elif normalized == "full_bse":
        if refresh:
            try:
                refresh_bse_stock_universe(path=universe_path)
            except Exception:
                if not universe_path.exists():
                    return pd.DataFrame(columns=UNIVERSE_COLUMNS)
        elif not universe_path.exists():
            return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    elif normalized == "all_india":
        if refresh:
            try:
                refresh_india_stock_universe(path=universe_path)
            except Exception:
                if not universe_path.exists():
                    return pd.DataFrame(columns=UNIVERSE_COLUMNS)
        elif not universe_path.exists():
            return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    elif refresh or not universe_path.exists():
        try:
            refresh_stock_universe(index_name=index_name, path=universe_path)
        except Exception:
            if not universe_path.exists():
                return pd.DataFrame(columns=UNIVERSE_COLUMNS)

    df = pd.read_csv(universe_path)
    for column in UNIVERSE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[UNIVERSE_COLUMNS].copy()
    df["active"] = df["active"].map(_to_bool)
    for column in ["free_float_market_cap", "last_price", "year_high", "year_low", "near_52w_high_pct", "return_30d_pct", "return_365d_pct"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["active"] == True].copy()  # noqa: E712
    df = _ensure_exchange_metadata(df)
    df = _apply_chartmaze_like_classification(df)
    df = _apply_sector_csv_taxonomy(df, restrict_to_taxonomy=normalized == "full_nse")
    if normalized == "full_nse":
        df = _apply_nse_total_market_cap_overlay(df)
    if max_stocks is not None:
        df = df.head(max(0, int(max_stocks))).copy()
    return df


def list_stock_universe(
    *,
    universe: str = "broad",
    sector: str | None = None,
    industry: str | None = None,
    limit: int | None = 500,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return stock universe metadata for UI/MCP discovery."""
    df = load_stock_universe(universe=universe, refresh=refresh)
    if sector:
        key = _slug(sector)
        df = df[df["sector"].map(_slug) == key]
    if industry:
        key = _slug(industry)
        df = df[(df["industry"].map(_slug) == key) | (df["basic_industry"].map(_slug) == key)]
    summary_df = df.copy()
    stocks_df = df.head(max(0, int(limit))).copy() if limit is not None else df
    sectors = (
        summary_df.groupby("sector", dropna=False)
        .agg(stock_count=("ticker", "count"), industry_count=("basic_industry", "nunique"))
        .reset_index()
        .sort_values(["stock_count", "sector"], ascending=[False, True])
        .to_dict(orient="records")
        if not summary_df.empty
        else []
    )
    industries = (
        summary_df.groupby(["sector", "basic_industry"], dropna=False)
        .agg(stock_count=("ticker", "count"))
        .reset_index()
        .sort_values(["stock_count", "basic_industry"], ascending=[False, True])
        .to_dict(orient="records")
        if not summary_df.empty
        else []
    )
    return {
        "universe": _normalize_universe_name(universe),
        "count": int(len(summary_df)),
        "sectors": sectors,
        "industries": industries,
        "stocks": stocks_df.to_dict(orient="records"),
        "source_path": str(_default_universe_path(_normalize_universe_name(universe))),
        "refreshed_at": _latest_universe_refreshed_at(summary_df),
    }


def list_sector_constituents(
    *,
    universe: str = "broad",
    sector: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return the exact stock constituents grouped by sector and basic industry."""
    df = load_stock_universe(universe=universe, refresh=refresh)
    if sector:
        key = _slug(sector)
        df = df[df["sector"].map(_slug) == key].copy()

    sector_rows = []
    for sector_name, sector_df in df.groupby("sector", dropna=False):
        sector_label = str(sector_name or "Unclassified")
        industries = []
        for industry_name, industry_df in sector_df.groupby("basic_industry", dropna=False):
            stocks = _stock_constituent_rows(industry_df)
            industries.append(
                {
                    "basic_industry": str(industry_name or "Unclassified"),
                    "stock_count": int(len(industry_df)),
                    "stocks": stocks,
                }
            )
        industries = sorted(industries, key=lambda row: (row["stock_count"], row["basic_industry"]), reverse=True)
        sector_rows.append(
            {
                "sector": sector_label,
                "stock_count": int(len(sector_df)),
                "industry_count": int(sector_df["basic_industry"].nunique(dropna=True)),
                "industries": industries,
                "stocks": _stock_constituent_rows(sector_df),
            }
        )

    sector_rows = sorted(sector_rows, key=lambda row: (row["stock_count"], row["sector"]), reverse=True)
    return {
        "universe": _normalize_universe_name(universe),
        "count": int(len(df)),
        "sector_count": len(sector_rows),
        "sectors": sector_rows,
        "source_path": str(_default_universe_path(_normalize_universe_name(universe))),
        "refreshed_at": _latest_universe_refreshed_at(df),
    }


def _latest_universe_refreshed_at(df: pd.DataFrame) -> str | None:
    if df.empty or "refreshed_at" not in df.columns:
        return None
    values = []
    for value in df["refreshed_at"].dropna().tolist():
        text = str(value).strip()
        if text and text.lower() != "nan":
            values.append(text)
    return max(values) if values else None


def _stock_constituent_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "ticker",
        "symbol",
        "name",
        "isin",
        "sector",
        "basic_industry",
        "free_float_market_cap",
        "last_price",
        "year_high",
        "year_low",
        "near_52w_high_pct",
        "return_30d_pct",
        "return_365d_pct",
        "exchange",
        "security_id",
        "nse_ticker",
        "bse_ticker",
        "data_quality",
        "classification_source",
    ]
    available = [column for column in columns if column in df.columns]
    rows = df.sort_values(["free_float_market_cap", "symbol"], ascending=[False, True], na_position="last")
    return rows[available].to_dict(orient="records")


def list_universe_industry_definitions(*, universe: str = "broad", refresh: bool = False) -> dict[str, Any]:
    """Return broad universe industries in the same shape as local industry definitions."""
    df = load_stock_universe(universe=universe, refresh=refresh)
    if df.empty:
        return {}
    definitions: dict[str, Any] = {}
    grouped = df.groupby(["sector", "basic_industry"], dropna=False)
    for (sector, basic_industry), rows in grouped:
        industry_name = str(basic_industry or "Unclassified")
        industry_id = _slug(industry_name)
        definitions[industry_id] = {
            "id": industry_id,
            "name": industry_name,
            "sector": str(sector or "Unclassified"),
            "stocks": list(rows["ticker"]),
            "stock_count": int(len(rows)),
            "source": "nse_broad_universe",
        }
    return dict(sorted(definitions.items(), key=lambda item: item[1]["name"]))


def refresh_stock_universe(
    *,
    index_name: str = DEFAULT_NSE_INDEX_NAME,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch the free NSE index constituent snapshot and persist it as CSV."""
    universe_path = Path(path) if path is not None else DEFAULT_UNIVERSE_PATH
    payload = _fetch_nse_index_json(index_name)
    rows = _nse_payload_to_universe_rows(payload, index_name=index_name)
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    with universe_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "index_name": index_name,
        "path": str(universe_path),
        "count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def refresh_full_stock_universe(
    *,
    path: str | Path | None = None,
    max_symbols: int | None = None,
    symbols: list[str] | None = None,
    request_delay: float = 0.15,
) -> dict[str, Any]:
    """Fetch a full NSE equity universe using public NSE equity master + quote metadata.

    This is slower than the NIFTY Total Market snapshot because NSE does not expose
    sector/basic-industry metadata in one bulk endpoint. We fetch the public equity
    master, then enrich each symbol through quote-equity industryInfo.
    """
    universe_path = Path(path) if path is not None else DEFAULT_FULL_NSE_UNIVERSE_PATH
    master_rows = _fetch_nse_equity_master_rows()
    if symbols:
        selected = {str(symbol).strip().upper().replace(".NS", "") for symbol in symbols}
        master_rows = [row for row in master_rows if str(row.get("SYMBOL") or "").strip().upper() in selected]
    if max_symbols is not None:
        master_rows = master_rows[: max(0, int(max_symbols))]

    refreshed_at = datetime.now(timezone.utc).isoformat()
    session = _nse_session()
    rows = []
    failures = []
    for index, master_row in enumerate(master_rows):
        symbol = str(master_row.get("SYMBOL") or "").strip().upper()
        if not symbol:
            continue
        try:
            quote_payload = _fetch_nse_quote_json(session, symbol)
            rows.append(_nse_quote_to_universe_row(quote_payload, master_row=master_row, refreshed_at=refreshed_at))
        except Exception as exc:  # noqa: BLE001
            failures.append({"symbol": symbol, "error": str(exc)})
            rows.append(_nse_master_to_unclassified_row(master_row, refreshed_at=refreshed_at))
        if request_delay and index < len(master_rows) - 1:
            time.sleep(max(0.0, float(request_delay)))

    universe_path.parent.mkdir(parents=True, exist_ok=True)
    with universe_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "index_name": "NSE FULL EQUITY",
        "path": str(universe_path),
        "count": len(rows),
        "failure_count": len(failures),
        "failures": failures[:25],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def refresh_bse_stock_universe(
    *,
    path: str | Path | None = None,
    scrip_master_url: str = DHAN_SCRIP_MASTER_DETAILED_CSV,
) -> dict[str, Any]:
    """Fetch active BSE equity shares from Dhan's free public instrument master."""
    universe_path = Path(path) if path is not None else DEFAULT_FULL_BSE_UNIVERSE_PATH
    refreshed_at = datetime.now(timezone.utc).isoformat()
    master_rows = _fetch_dhan_scrip_master_rows(scrip_master_url)
    rows = _dhan_scrip_master_to_universe_rows(master_rows, exchange="BSE", refreshed_at=refreshed_at)
    rows = _apply_nse_classification_lookup(rows)
    _write_universe_rows(universe_path, rows)
    return {
        "index_name": "BSE FULL EQUITY",
        "path": str(universe_path),
        "count": len(rows),
        "source": "dhan_public_scrip_master",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def refresh_india_stock_universe(
    *,
    path: str | Path | None = None,
    scrip_master_url: str = DHAN_SCRIP_MASTER_DETAILED_CSV,
) -> dict[str, Any]:
    """Fetch a combined NSE+BSE active equity universe from a free public instrument master.

    NSE quote metadata remains the classification source when available. BSE-only
    rows are kept, but marked as unclassified until an official sector source is
    available for that ISIN.
    """
    universe_path = Path(path) if path is not None else DEFAULT_ALL_INDIA_UNIVERSE_PATH
    refreshed_at = datetime.now(timezone.utc).isoformat()
    master_rows = _fetch_dhan_scrip_master_rows(scrip_master_url)
    rows = _dhan_scrip_master_to_universe_rows(master_rows, exchange=None, refreshed_at=refreshed_at)
    rows = _apply_nse_classification_lookup(rows)
    combined = _merge_exchange_rows(rows)
    _write_universe_rows(universe_path, combined)
    return {
        "index_name": "INDIA NSE+BSE EQUITY",
        "path": str(universe_path),
        "count": len(combined),
        "nse_count": sum(1 for row in combined if "NSE" in str(row.get("exchange") or "")),
        "bse_count": sum(1 for row in combined if "BSE" in str(row.get("exchange") or "")),
        "bse_only_count": sum(1 for row in combined if str(row.get("exchange") or "") == "BSE"),
        "source": "dhan_public_scrip_master+nse_full_universe_classification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_universe_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _fetch_dhan_scrip_master_rows(url: str = DHAN_SCRIP_MASTER_DETAILED_CSV) -> list[dict[str, Any]]:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), low_memory=False)
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame.to_dict(orient="records")


def _dhan_scrip_master_to_universe_rows(
    master_rows: list[dict[str, Any]],
    *,
    exchange: str | None,
    refreshed_at: str,
) -> list[dict[str, Any]]:
    selected_exchanges = {exchange.upper()} if exchange else {"NSE", "BSE"}
    rows = []
    for item in master_rows:
        exch = str(item.get("EXCH_ID") or "").strip().upper()
        if exch not in selected_exchanges:
            continue
        if str(item.get("SEGMENT") or "").strip().upper() != "E":
            continue
        if str(item.get("INSTRUMENT") or "").strip().upper() != "EQUITY":
            continue
        if str(item.get("INSTRUMENT_TYPE") or "").strip().upper() != "ES":
            continue
        security_id = _clean_security_id(item.get("SECURITY_ID"))
        raw_symbol = str(item.get("UNDERLYING_SYMBOL") or item.get("SYMBOL_NAME") or "").strip().upper()
        symbol = raw_symbol.replace(" ", "")
        if not symbol or not security_id:
            continue
        ticker = f"{symbol}.NS" if exch == "NSE" else f"{security_id}.BO"
        rows.append(
            {
                "ticker": ticker,
                "symbol": symbol,
                "name": str(item.get("SYMBOL_NAME") or item.get("DISPLAY_NAME") or symbol).strip(),
                "isin": _clean_isin(item.get("ISIN")),
                "sector": "Unclassified",
                "industry": "Unclassified",
                "basic_industry": "Unclassified",
                "index_name": "INDIA NSE+BSE EQUITY" if exchange is None else f"{exch} FULL EQUITY",
                "source": "dhan_public_scrip_master",
                "active": str(item.get("BUY_SELL_INDICATOR") or "A").strip().upper() != "S",
                "series": str(item.get("SERIES") or "").strip().upper(),
                "free_float_market_cap": None,
                "last_price": None,
                "year_high": None,
                "year_low": None,
                "near_52w_high_pct": None,
                "return_30d_pct": None,
                "return_365d_pct": None,
                "refreshed_at": refreshed_at,
                "exchange": exch,
                "security_id": security_id,
                "nse_ticker": ticker if exch == "NSE" else None,
                "bse_ticker": ticker if exch == "BSE" else None,
                "nse_security_id": security_id if exch == "NSE" else None,
                "bse_security_id": security_id if exch == "BSE" else None,
                "data_quality": "classified_by_nse_isin" if exch == "NSE" else "bse_public_master_unclassified",
            }
        )
    return rows


def _apply_nse_classification_lookup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _nse_classification_lookup()
    if not lookup:
        return rows
    out = []
    for row in rows:
        enriched = dict(row)
        match = lookup.get(f"isin:{row.get('isin')}") or lookup.get(f"symbol:{row.get('symbol')}")
        if match:
            for column in ["sector", "industry", "basic_industry"]:
                enriched[column] = match.get(column) or enriched[column]
            if match.get("name") and not enriched.get("name"):
                enriched["name"] = match["name"]
            enriched["data_quality"] = "classified_by_nse_isin"
        out.append(enriched)
    return out


def _nse_classification_lookup() -> dict[str, dict[str, Any]]:
    if not DEFAULT_FULL_NSE_UNIVERSE_PATH.exists():
        return {}
    try:
        frame = pd.read_csv(DEFAULT_FULL_NSE_UNIVERSE_PATH)
    except Exception:
        return {}
    if frame.empty:
        return {}
    frame = _apply_chartmaze_like_classification(frame)
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        payload = {
            "name": row.get("name"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "basic_industry": row.get("basic_industry"),
        }
        isin = _clean_isin(row.get("isin"))
        symbol = str(row.get("symbol") or "").strip().upper()
        if isin:
            lookup[f"isin:{isin}"] = payload
        if symbol:
            lookup[f"symbol:{symbol}"] = payload
    return lookup


def _merge_exchange_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: 0 if item.get("exchange") == "NSE" else 1):
        key = _combined_universe_key(row)
        existing = combined.get(key)
        if existing is None:
            combined[key] = dict(row)
            continue
        if row.get("exchange") == "BSE":
            existing["exchange"] = _join_exchanges(existing.get("exchange"), "BSE")
            existing["bse_ticker"] = row.get("bse_ticker") or row.get("ticker")
            existing["bse_security_id"] = row.get("bse_security_id") or row.get("security_id")
        elif row.get("exchange") == "NSE":
            replacement = dict(row)
            replacement["exchange"] = _join_exchanges("NSE", existing.get("exchange"))
            replacement["bse_ticker"] = existing.get("bse_ticker")
            replacement["bse_security_id"] = existing.get("bse_security_id")
            combined[key] = replacement
    return sorted(combined.values(), key=lambda item: (str(item.get("exchange") or ""), str(item.get("symbol") or "")))


def _combined_universe_key(row: dict[str, Any]) -> str:
    isin = _clean_isin(row.get("isin"))
    if isin:
        return f"isin:{isin}"
    return f"{row.get('exchange')}:{row.get('security_id') or row.get('symbol')}"


def _join_exchanges(*values: Any) -> str:
    exchanges = []
    for value in values:
        for part in str(value or "").replace("/", "+").split("+"):
            part = part.strip().upper()
            if part and part not in exchanges:
                exchanges.append(part)
    return "+".join(exchanges)


def _clean_security_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _clean_isin(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().upper()
    return "" if text in {"", "NA", "N.A.", "NAN", "NONE"} else text


def _fetch_nse_equity_master_rows() -> list[dict[str, Any]]:
    response = requests.get(NSE_EQUITY_MASTER_CSV, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    if "SYMBOL" not in frame.columns:
        raise ValueError("NSE equity master did not include SYMBOL column.")
    if "SERIES" in frame.columns:
        frame = frame[frame["SERIES"].astype(str).str.upper().isin({"EQ", "BE", "SM", "ST"})].copy()
    return frame.to_dict(orient="records")


def _nse_session() -> requests.Session:
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": NSE_HOME,
    }
    session.headers.update(headers)
    session.get(NSE_HOME, timeout=15)
    return session


def _fetch_nse_index_json(index_name: str) -> dict[str, Any]:
    session = _nse_session()
    url = NSE_INDEX_API.format(index_name=quote(index_name))
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return json.loads(response.text)


def _fetch_nse_quote_json(session: requests.Session, symbol: str) -> dict[str, Any]:
    url = NSE_QUOTE_EQUITY_API.format(symbol=quote(symbol))
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return json.loads(response.text)


def _nse_quote_to_universe_row(payload: dict[str, Any], *, master_row: dict[str, Any], refreshed_at: str) -> dict[str, Any]:
    info = payload.get("info") or {}
    metadata = payload.get("metadata") or {}
    industry_info = payload.get("industryInfo") or {}
    price_info = payload.get("priceInfo") or {}
    week_high_low = price_info.get("weekHighLow") or {}
    symbol = str(info.get("symbol") or metadata.get("symbol") or master_row.get("SYMBOL") or "").strip().upper()
    basic_industry = str(industry_info.get("basicIndustry") or metadata.get("industry") or info.get("industry") or "Unclassified").strip()
    sector = _sector_from_industry_info(industry_info.get("sector"), basic_industry)
    industry = str(industry_info.get("industry") or basic_industry).strip()
    year_high = _safe_float(week_high_low.get("max"))
    last_price = _safe_float(price_info.get("lastPrice"))
    return {
        "ticker": f"{symbol}.NS",
        "symbol": symbol,
        "name": str(info.get("companyName") or master_row.get("NAME OF COMPANY") or symbol).strip(),
        "isin": str(info.get("isin") or metadata.get("isin") or master_row.get("ISIN NUMBER") or "").strip(),
        "sector": sector or "Unclassified",
        "industry": industry or basic_industry,
        "basic_industry": basic_industry or "Unclassified",
        "index_name": "NSE FULL EQUITY",
        "source": "nse_equity_master_quote",
        "active": not bool(info.get("isDelisted") or info.get("isSuspended") or info.get("isETFSec")),
        "series": str(metadata.get("series") or master_row.get("SERIES") or "").strip().upper(),
        "free_float_market_cap": None,
        "last_price": last_price,
        "year_high": year_high,
        "year_low": _safe_float(week_high_low.get("min")),
        "near_52w_high_pct": None if not last_price or not year_high else round(100 * (year_high - last_price) / year_high, 4),
        "return_30d_pct": None,
        "return_365d_pct": None,
        "refreshed_at": refreshed_at,
    }


def _nse_master_to_unclassified_row(master_row: dict[str, Any], *, refreshed_at: str) -> dict[str, Any]:
    symbol = str(master_row.get("SYMBOL") or "").strip().upper()
    return {
        "ticker": f"{symbol}.NS",
        "symbol": symbol,
        "name": str(master_row.get("NAME OF COMPANY") or symbol).strip(),
        "isin": str(master_row.get("ISIN NUMBER") or "").strip(),
        "sector": "Unclassified",
        "industry": "Unclassified",
        "basic_industry": "Unclassified",
        "index_name": "NSE FULL EQUITY",
        "source": "nse_equity_master_unclassified",
        "active": True,
        "series": str(master_row.get("SERIES") or "").strip().upper(),
        "free_float_market_cap": None,
        "last_price": None,
        "year_high": None,
        "year_low": None,
        "near_52w_high_pct": None,
        "return_30d_pct": None,
        "return_365d_pct": None,
        "refreshed_at": refreshed_at,
    }


def _sector_from_industry_info(raw_sector: Any, basic_industry: str) -> str:
    inferred = infer_sector_from_basic_industry(basic_industry)
    if inferred in {"Forest Materials", "Metals & Mining"}:
        return inferred
    return _normalize_sector_label(raw_sector or inferred or "Unclassified")


def _normalize_sector_label(sector: Any) -> str:
    value = str(sector or "Unclassified").strip()
    key = " ".join(value.lower().replace("&", "and").replace(",", " ").split())
    aliases = {
        "automobile": "Auto",
        "automobile and auto components": "Auto",
        "fast moving consumer goods": "FMCG",
        "fmcg": "FMCG",
        "oil gas and consumable fuels": "Oil, Gas & Consumable fuels",
        "oil gas & consumable fuels": "Oil, Gas & Consumable fuels",
        "services": "Services",
        "construction": "Construction",
        "construction materials": "Construction",
        "healthcare services": "Healthcare",
    }
    return aliases.get(key, value)


def _nse_payload_to_universe_rows(payload: dict[str, Any], *, index_name: str) -> list[dict[str, Any]]:
    refreshed_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in payload.get("data", []):
        meta = item.get("meta") or {}
        symbol = str(meta.get("symbol") or item.get("symbol") or "").strip().upper()
        series = str(item.get("series") or "").strip().upper()
        if not symbol or symbol == index_name or meta.get("isETFSec") is True:
            continue
        basic_industry = str(meta.get("industry") or "Unclassified").strip()
        rows.append(
            {
                "ticker": f"{symbol}.NS",
                "symbol": symbol,
                "name": str(meta.get("companyName") or symbol).strip(),
                "isin": str(meta.get("isin") or "").strip(),
                "sector": infer_sector_from_basic_industry(basic_industry),
                "industry": basic_industry,
                "basic_industry": basic_industry,
                "index_name": str(payload.get("name") or index_name),
                "source": "nse_equity_stock_indices",
                "active": not bool(meta.get("isDelisted") or meta.get("isSuspended")) and series in {"EQ", "BE", "SM", "ST", ""},
                "series": series,
                "free_float_market_cap": item.get("ffmc"),
                "last_price": item.get("lastPrice"),
                "year_high": item.get("yearHigh"),
                "year_low": item.get("yearLow"),
                "near_52w_high_pct": item.get("nearWKH"),
                "return_30d_pct": item.get("perChange30d"),
                "return_365d_pct": item.get("perChange365d"),
                "refreshed_at": refreshed_at,
            }
        )
    return rows


def infer_sector_from_basic_industry(basic_industry: str) -> str:
    """Map NSE basic industries into ChartsMaze-like sector buckets."""
    value = str(basic_industry or "").lower()
    exact = {
        "private sector bank": "Financial Services",
        "public sector bank": "Financial Services",
        "other bank": "Financial Services",
        "non banking financial company (nbfc)": "Financial Services",
        "housing finance company": "Financial Services",
        "life insurance": "Financial Services",
        "general insurance": "Financial Services",
        "asset management company": "Financial Services",
        "stockbroking & allied": "Financial Services",
        "exchange and data platform": "Financial Services",
        "depositories clearing houses and other intermediaries": "Financial Services",
        "financial institution": "Financial Services",
        "financial technology (fintech)": "Financial Services",
        "financial products distributor": "Financial Services",
        "microfinance institutions": "Financial Services",
        "other financial services": "Financial Services",
        "paper & paper products": "Forest Materials",
        "plywood boards/ laminates": "Forest Materials",
        "wood products": "Forest Materials",
    }
    if value in exact:
        return exact[value]

    keyword_map = [
        ("Healthcare", ("pharma", "hospital", "healthcare", "medical", "biotechnology", "pharmacy", "wellness")),
        ("Oil, Gas & Consumable fuels", ("refineries", "oil ", "oil-", "gas ", "gas/", "coal", "lpg", "cng", "png", "lng", "lubricants")),
        ("FMCG", ("personal care", "packaged foods", "food", "beverages", "cigarettes", "tobacco", "tea", "coffee", "edible oil", "dairy", "sugar", "breweries", "distilleries", "animal feed", "household products")),
        ("Capital Goods", ("electrical equipment", "industrial products", "compressors", "pumps", "diesel engines", "aerospace", "defense", "cables", "abrasives", "bearings", "castings", "forgings", "construction vehicles", "railway wagons", "industrial gases", "explosives")),
        ("Metals & Mining", ("iron", "steel", "aluminium", "zinc", "copper", "metal", "mineral", "mining", "ferro", "manganese", "electrodes", "refractories")),
        ("Media Entertainment & Publication", ("media", "entertainment", "broadcasting", "film", "digital entertainment")),
        ("Information Technology", ("computer", "software", "it enabled", "bpo", "kpo", "e-learning")),
        ("Telecommunication", ("telecom",)),
        ("Construction", ("civil construction", "cement", "construction products", "ceramics", "sanitary", "glass")),
        ("Textiles", ("textile", "garments", "apparels")),
        ("Chemicals", ("chemical", "pesticides", "agrochemicals", "fertilizers", "paints", "dyes", "pigments", "petrochemicals", "plastic products", "carbon black")),
        ("Services", ("logistics", "port", "shipping", "transport", "commercial services", "water supply", "trading", "distributors")),
        ("Auto", ("auto components", "wheelers", "passenger cars", "commercial vehicles", "tyres", "rubber", "tractors")),
        ("Consumer Services", ("hotel", "restaurant", "retail", "e-commerce", "catalogue", "tour", "travel", "airline", "airport", "education")),
        ("Realty", ("residential commercial", "real estate",)),
        ("Consumer Durables", ("household appliances", "consumer electronics", "gems", "jewellery", "watches", "furniture", "footwear", "houseware", "stationary")),
        ("Power", ("power generation", "integrated power", "power distribution", "power - transmission", "power trading")),
        ("Forest Materials", ("paper", "forest", "plywood", "wood products", "laminates", "jute")),
        ("Diversified", ("diversified", "holding company", "investment company")),
    ]
    for sector, keywords in keyword_map:
        if any(keyword in value for keyword in keywords):
            return sector
    return "Others"


def _apply_chartmaze_like_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize public NSE classifications into the app's ChartsMaze-like buckets."""
    if df.empty:
        return df
    out = df.copy()
    for index, row in out.iterrows():
        sector, industry = _chartmaze_sector_industry(
            row.get("sector"),
            row.get("basic_industry") or row.get("industry"),
            row.get("symbol"),
        )
        out.at[index, "sector"] = sector
        out.at[index, "industry"] = industry
        out.at[index, "basic_industry"] = industry
    return out


def load_sector_csv_taxonomy(path: str | Path | None = None) -> pd.DataFrame:
    """Load sector/industry membership from data/sectors folder CSVs.

    Folder name is treated as the sector, file name as the industry, and the
    Stock Name column as the NSE symbol. Numeric CSV columns are ignored here;
    live analytics recompute RS, returns, and moving averages from price data.
    """
    root = Path(path) if path is not None else DEFAULT_SECTOR_TAXONOMY_PATH
    columns = [
        "symbol",
        "ticker",
        "sector",
        "industry",
        "basic_industry",
        "taxonomy_file",
        "classification_source",
    ]
    if not root.exists():
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for csv_path in sorted(root.glob("*/*.csv")):
        sector = _taxonomy_sector_name(csv_path.parent.name)
        industry = _taxonomy_industry_name(csv_path)
        try:
            file_df = pd.read_csv(csv_path)
        except Exception:
            continue
        if file_df.empty:
            continue
        stock_column = "Stock Name" if "Stock Name" in file_df.columns else str(file_df.columns[0])
        for raw_symbol in file_df[stock_column].dropna().tolist():
            symbol = _taxonomy_symbol(raw_symbol)
            if not symbol:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "ticker": f"{symbol}.NS",
                    "sector": sector,
                    "industry": industry,
                    "basic_industry": industry,
                    "taxonomy_file": str(csv_path),
                    "classification_source": SECTOR_TAXONOMY_SOURCE,
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    taxonomy_df = pd.DataFrame(rows, columns=columns)
    return _dedupe_sector_taxonomy(taxonomy_df)


def _apply_sector_csv_taxonomy(df: pd.DataFrame, *, restrict_to_taxonomy: bool = False) -> pd.DataFrame:
    taxonomy_df = load_sector_csv_taxonomy()
    if taxonomy_df.empty:
        return df

    taxonomy = {row["symbol"]: row for row in taxonomy_df.to_dict(orient="records")}
    out = df.copy()
    if "classification_source" not in out.columns:
        out["classification_source"] = None
    if "symbol" not in out.columns:
        out["symbol"] = out.get("ticker", "").map(lambda value: _taxonomy_symbol(str(value).replace(".NS", "")))

    matched_symbols: set[str] = set()
    keep_mask = []
    for index, row in out.iterrows():
        symbol = _taxonomy_symbol(row.get("symbol") or str(row.get("ticker") or "").replace(".NS", ""))
        taxonomy_row = taxonomy.get(symbol)
        keep_mask.append(taxonomy_row is not None)
        if not taxonomy_row:
            continue
        matched_symbols.add(symbol)
        out.at[index, "symbol"] = symbol
        out.at[index, "ticker"] = row.get("ticker") or taxonomy_row["ticker"]
        out.at[index, "sector"] = taxonomy_row["sector"]
        out.at[index, "industry"] = taxonomy_row["industry"]
        out.at[index, "basic_industry"] = taxonomy_row["basic_industry"]
        out.at[index, "classification_source"] = SECTOR_TAXONOMY_SOURCE

    if restrict_to_taxonomy:
        out = out[pd.Series(keep_mask, index=out.index)].copy()
        missing_rows = [
            _taxonomy_universe_row(taxonomy_row)
            for symbol, taxonomy_row in taxonomy.items()
            if symbol not in matched_symbols
        ]
        if missing_rows:
            out = pd.concat([out, pd.DataFrame(missing_rows)], ignore_index=True)

    for column in UNIVERSE_COLUMNS:
        if column not in out.columns:
            out[column] = None
    return out[UNIVERSE_COLUMNS].reset_index(drop=True)


def _apply_nse_total_market_cap_overlay(df: pd.DataFrame) -> pd.DataFrame:
    """Fill market-cap fields from the public NSE Total Market snapshot where available."""
    if df.empty or not DEFAULT_UNIVERSE_PATH.exists():
        return df
    try:
        broad_df = pd.read_csv(DEFAULT_UNIVERSE_PATH, usecols=["symbol", "free_float_market_cap"])
    except Exception:
        return df
    if broad_df.empty or "free_float_market_cap" not in broad_df.columns:
        return df

    broad_df["symbol"] = broad_df["symbol"].map(_taxonomy_symbol)
    broad_df["free_float_market_cap"] = pd.to_numeric(broad_df["free_float_market_cap"], errors="coerce")
    cap_by_symbol = (
        broad_df.dropna(subset=["symbol", "free_float_market_cap"])
        .drop_duplicates("symbol", keep="first")
        .set_index("symbol")["free_float_market_cap"]
        .to_dict()
    )
    if not cap_by_symbol:
        return df

    out = df.copy()
    out["symbol"] = out["symbol"].map(_taxonomy_symbol)
    out["free_float_market_cap"] = pd.to_numeric(out["free_float_market_cap"], errors="coerce")
    missing_cap = out["free_float_market_cap"].isna()
    out.loc[missing_cap, "free_float_market_cap"] = out.loc[missing_cap, "symbol"].map(cap_by_symbol)
    return out[UNIVERSE_COLUMNS].reset_index(drop=True)


def _dedupe_sector_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    selected: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        symbol = row["symbol"]
        existing = selected.get(symbol)
        if existing is None or _taxonomy_preference_score(row) > _taxonomy_preference_score(existing):
            selected[symbol] = row
    rows = sorted(selected.values(), key=lambda item: (item["sector"], item["industry"], item["symbol"]))
    return pd.DataFrame(rows, columns=list(df.columns))


def _taxonomy_preference_score(row: dict[str, Any]) -> int:
    industry_key = str(row.get("industry") or "").strip().lower()
    sector = str(row.get("sector") or "").strip()
    preferred_sector = TAXONOMY_PREFERRED_SECTOR_BY_INDUSTRY.get(industry_key)
    return 100 if preferred_sector == sector else 0


def _taxonomy_universe_row(taxonomy_row: dict[str, Any]) -> dict[str, Any]:
    symbol = taxonomy_row["symbol"]
    row = {column: None for column in UNIVERSE_COLUMNS}
    row.update(
        {
            "ticker": taxonomy_row["ticker"],
            "symbol": symbol,
            "name": symbol,
            "sector": taxonomy_row["sector"],
            "industry": taxonomy_row["industry"],
            "basic_industry": taxonomy_row["basic_industry"],
            "index_name": "LOCAL SECTOR TAXONOMY",
            "source": SECTOR_TAXONOMY_SOURCE,
            "active": True,
            "series": "EQ",
            "exchange": "NSE",
            "nse_ticker": taxonomy_row["ticker"],
            "data_quality": "taxonomy_only",
            "classification_source": SECTOR_TAXONOMY_SOURCE,
        }
    )
    return row


def _taxonomy_sector_name(folder_name: str) -> str:
    return TAXONOMY_FOLDER_SECTOR_ALIASES.get(folder_name, _normalize_sector_label(folder_name))


def _taxonomy_industry_name(csv_path: Path) -> str:
    label = csv_path.stem
    prefix = "Stocks Data_"
    if label.startswith(prefix):
        label = label[len(prefix) :]
    if label.endswith(")") and " (" in label:
        base, suffix = label.rsplit(" (", 1)
        if suffix[:-1].isdigit():
            label = base
    return " ".join(label.replace("_", "/").split())


def _taxonomy_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol or symbol == "NAN":
        return ""
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
    return symbol


def _chartmaze_sector_industry(raw_sector: Any, raw_basic_industry: Any, symbol: Any = None) -> tuple[str, str]:
    basic_industry = _clean_label(raw_basic_industry, default="Unclassified")
    key = _classification_key(basic_industry)
    basic_industry = CHARTMAZE_BASIC_INDUSTRY_ALIASES.get(key, basic_industry)
    key = _classification_key(basic_industry)
    if _is_metal_basic_industry(key):
        override = CHARTMAZE_METAL_INDUSTRY_OVERRIDES.get(str(symbol or "").strip().upper())
        if override:
            return "Metals & Mining", override
        return "Metals & Mining", _normalize_metal_industry(key, basic_industry)
    if _is_forest_basic_industry(key):
        return "Forest Materials", _normalize_forest_industry(key, basic_industry)

    inferred = infer_sector_from_basic_industry(basic_industry)
    sector = _normalize_sector_label(raw_sector or inferred or "Unclassified")
    return sector, basic_industry


def _is_metal_basic_industry(key: str) -> bool:
    exact = {
        "aluminium",
        "copper",
        "diversified metals",
        "electrodes and refractories",
        "ferro and silica manganese",
        "industrial minerals",
        "iron and steel",
        "iron and steel products",
        "pig iron",
        "precious metals",
        "sponge iron",
        "trading metals",
        "trading minerals",
        "zinc",
    }
    if key in exact:
        return True
    keywords = ("aluminium", "copper", "ferro", "iron", "metal", "mineral", "mining", "refractor", "steel", "zinc")
    return any(keyword in key for keyword in keywords)


def _normalize_metal_industry(key: str, fallback: str) -> str:
    if key in {"iron and steel", "iron and steel products", "pig iron", "sponge iron"}:
        return "Iron & Steel"
    if key in {"ferro and silica manganese", "industrial minerals", "trading minerals"} or "mining" in key or "mineral" in key:
        return "Mining/Minerals"
    if key == "trading metals":
        return "Trading - Metals"
    if key in {
        "aluminium",
        "copper",
        "diversified metals",
        "electrodes and refractories",
        "precious metals",
        "zinc",
    }:
        return "Metal Fabrication"
    if "product" in key or "fabrication" in key or "pipe" in key or "wire" in key:
        return "Metal Fabrication"
    if "iron" in key or "steel" in key:
        return "Iron & Steel"
    return fallback


def _is_forest_basic_industry(key: str) -> bool:
    return any(keyword in key for keyword in ("forest", "jute", "laminate", "paper", "plywood", "wood"))


def _normalize_forest_industry(key: str, fallback: str) -> str:
    if "paper" in key:
        return "Paper & Paper Products"
    if "plywood" in key or "wood" in key or "laminate" in key:
        return "Wood Products"
    if "jute" in key:
        return "Jute & Jute Products"
    return fallback


def _classification_key(value: Any) -> str:
    label = _clean_label(value, default="")
    return " ".join(label.lower().replace("&", "and").replace("/", " ").replace("-", " ").split())


def _clean_label(value: Any, *, default: str) -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    label = str(value).strip()
    return label or default


def _normalize_universe_name(universe: str) -> str:
    key = str(universe or "local").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"full", "full_nse", "nse_full", "nse_equity", "all_nse", "all_nse_equity", "nse_all_equity"}:
        return "full_nse"
    if key in {"full_bse", "bse_full", "bse_equity", "all_bse", "all_bse_equity", "bse_all_equity"}:
        return "full_bse"
    if key in {"india", "all_india", "nse_bse", "bse_nse", "india_equity", "all_india_equity", "nse_bse_equity"}:
        return "all_india"
    if key in {"broad", "nse", "nse_total_market", "nifty_total_market", "total_market"}:
        return "broad"
    return "local"


def _default_universe_path(universe: str) -> Path:
    if universe == "full_nse":
        return DEFAULT_FULL_NSE_UNIVERSE_PATH
    if universe == "full_bse":
        return DEFAULT_FULL_BSE_UNIVERSE_PATH
    if universe == "all_india":
        return DEFAULT_ALL_INDIA_UNIVERSE_PATH
    return DEFAULT_UNIVERSE_PATH


def _ensure_exchange_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in ["exchange", "nse_ticker", "bse_ticker"]:
        out[column] = out[column].astype("object")
    ticker = out["ticker"].fillna("").astype(str)
    missing_exchange = out["exchange"].fillna("").astype(str).str.strip() == ""
    out.loc[missing_exchange & ticker.str.endswith(".NS"), "exchange"] = "NSE"
    out.loc[missing_exchange & ticker.str.endswith(".BO"), "exchange"] = "BSE"
    out.loc[(out["nse_ticker"].fillna("").astype(str).str.strip() == "") & ticker.str.endswith(".NS"), "nse_ticker"] = ticker
    out.loc[(out["bse_ticker"].fillna("").astype(str).str.strip() == "") & ticker.str.endswith(".BO"), "bse_ticker"] = ticker
    return out


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(value: Any) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(value or ""))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unclassified"

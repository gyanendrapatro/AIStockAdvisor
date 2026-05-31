from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from stock_advisor.analysis.chart_patterns import detect_chart_patterns
from stock_advisor.analysis.indicators import add_indicators, latest_indicators
from stock_advisor.analysis.sector_rotation import (
    BENCHMARK_TICKER,
    SECTOR_DEFINITIONS,
    _get_index_price_history,
    _json_safe,
    _number,
    _price_metrics,
    _score_acceleration,
    _score_relative_strength,
    _score_sector_stock,
    _score_trend,
)
from stock_advisor.data.market_data import get_basic_fundamentals, get_price_histories, get_price_history
from stock_advisor.data.nse_indices import get_nse_index_price_history
from stock_advisor.data.universe import list_universe_industry_definitions, load_stock_universe


MIN_SECTOR_STOCKS_FOR_RANKING = 5
NON_RANKING_SECTORS = {"Others", "Unclassified"}
SECTOR_ANALYTICS_VERSION = "sector_analytics_industry_detail_v5"
RS_RATING_MIN = 1
RS_RATING_MAX = 99
RS_RATING_WEIGHTS = (
    ("return_20d", 0.15),
    ("return_60d", 0.25),
    ("return_120d", 0.25),
    ("return_252d", 0.35),
)
RRG_RATIO_LOOKBACK = 22
RRG_MOMENTUM_LOOKBACK = 8
RRG_SMOOTH_PERIOD = 9
RRG_MIN_HISTORY_POINTS = RRG_RATIO_LOOKBACK + RRG_MOMENTUM_LOOKBACK + RRG_SMOOTH_PERIOD


INDUSTRY_DEFINITIONS: dict[str, dict[str, Any]] = {
    "auto_oem": {
        "name": "Auto OEM",
        "sector": "Auto",
        "stocks": ("MARUTI.NS", "M&M.NS", "TMPV.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS"),
    },
    "auto_ancillary": {
        "name": "Auto Ancillary",
        "sector": "Auto",
        "stocks": ("MOTHERSON.NS", "BOSCHLTD.NS", "BALKRISIND.NS", "MRF.NS", "APOLLOTYRE.NS"),
    },
    "private_banks": {
        "name": "Private Banks",
        "sector": "Banking",
        "stocks": ("HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "IDFCFIRSTB.NS"),
    },
    "psu_banks": {
        "name": "PSU Banks",
        "sector": "Banking",
        "stocks": ("SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "INDIANB.NS"),
    },
    "it_services": {
        "name": "IT Services",
        "sector": "Information Technology",
        "stocks": ("TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTM.NS", "PERSISTENT.NS", "COFORGE.NS"),
    },
    "pharma": {
        "name": "Pharma",
        "sector": "Healthcare",
        "stocks": ("SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS", "ZYDUSLIFE.NS"),
    },
    "metals_steel": {
        "name": "Metals & Steel",
        "sector": "Metals",
        "stocks": ("TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS"),
    },
    "real_estate": {
        "name": "Real Estate Developers",
        "sector": "Real Estate",
        "stocks": ("DLF.NS", "GODREJPROP.NS", "LODHA.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS", "BRIGADE.NS", "ANANTRAJ.NS"),
    },
    "infra_epc": {
        "name": "Infrastructure & EPC",
        "sector": "Infrastructure",
        "stocks": ("LT.NS", "NCC.NS", "IRB.NS", "RVNL.NS", "IRCON.NS", "KPIL.NS", "KEC.NS", "PNCINFRA.NS"),
    },
    "power_energy": {
        "name": "Power & Energy",
        "sector": "Energy",
        "stocks": ("NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "JSWENERGY.NS", "NHPC.NS", "SJVN.NS"),
    },
    "oil_gas": {
        "name": "Oil & Gas",
        "sector": "Energy",
        "stocks": ("RELIANCE.NS", "ONGC.NS", "OIL.NS", "GAIL.NS", "PETRONET.NS", "MGL.NS", "IGL.NS", "BPCL.NS"),
    },
    "fmcg": {
        "name": "FMCG",
        "sector": "Consumption",
        "stocks": ("HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "TATACONSUM.NS"),
    },
    "retail_consumption": {
        "name": "Retail & Consumption",
        "sector": "Consumption",
        "stocks": ("TITAN.NS", "TRENT.NS", "DMART.NS", "VBL.NS", "INDHOTEL.NS", "JUBLFOOD.NS", "KALYANKJIL.NS", "ETERNAL.NS"),
    },
    "media_entertainment": {
        "name": "Media & Entertainment",
        "sector": "Media",
        "stocks": ("SUNTV.NS", "ZEEL.NS", "PVRINOX.NS", "NETWORK18.NS", "SAREGAMA.NS", "NAZARA.NS"),
    },
}

RETURN_WINDOWS = {
    "1d": "return_1d",
    "5d": "return_5d",
    "1w": "return_5d",
    "20d": "return_20d",
    "1m": "return_20d",
    "60d": "return_60d",
    "3m": "return_60d",
    "120d": "return_120d",
    "6m": "return_120d",
    "1y": "return_252d",
}

SECTOR_ANALYTICS_MODES = {
    "moving_average": "Moving Average",
    "relative_strength": "Relative Strength",
    "near_52w_high": "Near 52w High",
}

RRG_ADDITIONAL_INDEX_DEFINITIONS: dict[str, dict[str, Any]] = {
    "nifty_500": {
        "name": "Nifty 500",
        "index_ticker": "^CRSLDX",
        "sector": "Broad Market",
        "stocks": (
            "RELIANCE.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "INFY.NS",
            "TCS.NS",
            "LT.NS",
            "ITC.NS",
            "BHARTIARTL.NS",
            "SBIN.NS",
            "AXISBANK.NS",
            "M&M.NS",
            "SUNPHARMA.NS",
        ),
    },
    "nifty_100": {
        "name": "Nifty 100",
        "index_ticker": "^CNX100",
        "sector": "Broad Market",
        "stocks": (
            "RELIANCE.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "INFY.NS",
            "TCS.NS",
            "LT.NS",
            "ITC.NS",
            "BHARTIARTL.NS",
            "SBIN.NS",
            "KOTAKBANK.NS",
        ),
    },
    "midcap_150": {
        "name": "Nifty Midcap 150",
        "index_ticker": "MID150BEES.NS",
        "sector": "Broad Market",
        "stocks": (
            "MAXHEALTH.NS",
            "DIXON.NS",
            "PERSISTENT.NS",
            "COFORGE.NS",
            "POLYCAB.NS",
            "CUMMINSIND.NS",
            "BSE.NS",
            "SUPREMEIND.NS",
            "INDHOTEL.NS",
            "ASHOKLEY.NS",
        ),
    },
    "smallcap_250": {
        "name": "Nifty Smallcap 250",
        "index_ticker": "SMALLCAP.NS",
        "sector": "Broad Market",
        "stocks": (
            "CDSL.NS",
            "CAMS.NS",
            "ANGELONE.NS",
            "KFINTECH.NS",
            "IRCON.NS",
            "PNCINFRA.NS",
            "KNRCON.NS",
            "SAREGAMA.NS",
            "NAZARA.NS",
            "AARTIIND.NS",
        ),
    },
    "midsmall_400": {
        "name": "Nifty Midsmallcap 400",
        "index_ticker": "NIFTYMIDSML400_PROXY",
        "preferred_price_method": "equal_weight_proxy",
        "sector": "Broad Market",
        "stocks": (
            "MAXHEALTH.NS",
            "DIXON.NS",
            "PERSISTENT.NS",
            "COFORGE.NS",
            "BSE.NS",
            "CDSL.NS",
            "CAMS.NS",
            "IRCON.NS",
            "KFINTECH.NS",
            "ANGELONE.NS",
        ),
    },
    "private_bank": {
        "name": "Nifty Private Bank",
        "index_ticker": "BANKBEES.NS",
        "sector": "Banking",
        "stocks": ("HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "IDFCFIRSTB.NS"),
    },
    "financial_services": {
        "name": "Nifty Financial Services",
        "index_ticker": "^CNXFIN",
        "index_tickers": ("^CNXFIN", "FINIETF.NS", "BANKBEES.NS"),
        "sector": "Financial Services",
        "stocks": (
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "SBIN.NS",
            "BAJFINANCE.NS",
            "BAJAJFINSV.NS",
            "JIOFIN.NS",
            "HDFCLIFE.NS",
            "SBILIFE.NS",
            "ICICIPRULI.NS",
            "CHOLAFIN.NS",
        ),
    },
    "healthcare": {
        "name": "Nifty Healthcare",
        "index_ticker": "HEALTHY.NS",
        "sector": "Healthcare",
        "stocks": (
            "SUNPHARMA.NS",
            "CIPLA.NS",
            "DRREDDY.NS",
            "DIVISLAB.NS",
            "APOLLOHOSP.NS",
            "MAXHEALTH.NS",
            "FORTIS.NS",
            "LALPATHLAB.NS",
            "METROPOLIS.NS",
            "BIOCON.NS",
        ),
    },
    "consumer_durables": {
        "name": "Nifty Consumer Durables",
        "index_ticker": "CONSUMIETF.NS",
        "sector": "Consumer Durables",
        "stocks": (
            "TITAN.NS",
            "DIXON.NS",
            "KALYANKJIL.NS",
            "VOLTAS.NS",
            "BLUESTARCO.NS",
            "HAVELLS.NS",
            "CROMPTON.NS",
            "BATAINDIA.NS",
            "WHIRLPOOL.NS",
            "KAJARIACER.NS",
        ),
    },
    "commodities": {
        "name": "Nifty Commodities",
        "index_ticker": "COMMOIETF.NS",
        "sector": "Commodities",
        "stocks": (
            "RELIANCE.NS",
            "TATASTEEL.NS",
            "JSWSTEEL.NS",
            "HINDALCO.NS",
            "VEDL.NS",
            "COALINDIA.NS",
            "UPL.NS",
            "TATACHEM.NS",
            "ONGC.NS",
            "OIL.NS",
        ),
    },
    "mnc": {
        "name": "Nifty MNC",
        "index_ticker": "MNC.NS",
        "sector": "MNC",
        "stocks": (
            "HINDUNILVR.NS",
            "NESTLEIND.NS",
            "BRITANNIA.NS",
            "MARUTI.NS",
            "BOSCHLTD.NS",
            "ABB.NS",
            "SIEMENS.NS",
            "COLPAL.NS",
            "WHIRLPOOL.NS",
            "GILLETTE.NS",
        ),
    },
    "manufacturing": {
        "name": "Nifty Manufacturing",
        "index_ticker": "MAKEINDIA.NS",
        "sector": "Manufacturing",
        "stocks": (
            "RELIANCE.NS",
            "LT.NS",
            "TATASTEEL.NS",
            "JSWSTEEL.NS",
            "M&M.NS",
            "TMPV.NS",
            "MARUTI.NS",
            "DIXON.NS",
            "POLYCAB.NS",
            "CUMMINSIND.NS",
        ),
    },
    "chemicals": {
        "name": "Nifty Chemicals",
        "index_ticker": "CHEMICAL.NS",
        "index_tickers": ("CHEMICAL.NS", "CHEMICALS.NS"),
        "sector": "Chemicals",
        "stocks": (
            "PIDILITIND.NS",
            "SRF.NS",
            "PIIND.NS",
            "UPL.NS",
            "AARTIIND.NS",
            "DEEPAKNTR.NS",
            "TATACHEM.NS",
            "NAVINFLUOR.NS",
            "ATUL.NS",
            "CLEAN.NS",
        ),
    },
    "capital_markets": {
        "name": "Nifty Capital Markets",
        "index_ticker": "MOCAPITAL.NS",
        "index_tickers": ("MOCAPITAL.NS", "GROWWCAPM.NS"),
        "sector": "Capital Markets",
        "stocks": ("BSE.NS", "CDSL.NS", "CAMS.NS", "MCX.NS", "ANGELONE.NS", "IEX.NS", "KFINTECH.NS", "360ONE.NS", "MOTILALOFS.NS"),
    },
    "housing": {
        "name": "Nifty Housing",
        "index_ticker": "NIFTYHOUSING_PROXY",
        "preferred_price_method": "equal_weight_proxy",
        "sector": "Housing",
        "stocks": (
            "DLF.NS",
            "LODHA.NS",
            "GODREJPROP.NS",
            "BRIGADE.NS",
            "PRESTIGE.NS",
            "LICHSGFIN.NS",
            "CANFINHOME.NS",
            "PNBHOUSING.NS",
            "AAVAS.NS",
            "HOMEFIRST.NS",
        ),
    },
    "ev": {
        "name": "Nifty EV",
        "index_ticker": "EVINDIA.NS",
        "sector": "EV",
        "stocks": ("M&M.NS", "MARUTI.NS", "TVSMOTOR.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "EXIDEIND.NS", "ARE&M.NS", "MOTHERSON.NS", "BOSCHLTD.NS", "SONACOMS.NS"),
    },
    "mobility": {
        "name": "Nifty Mobility",
        "index_ticker": "MOBILITY.NS",
        "preferred_price_method": "equal_weight_proxy",
        "sector": "Mobility",
        "stocks": ("MARUTI.NS", "M&M.NS", "ASHOKLEY.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "BOSCHLTD.NS", "IRCTC.NS", "MOTHERSON.NS"),
    },
    "defence": {
        "name": "Nifty Defence",
        "index_ticker": "DEFENCE.NS",
        "preferred_price_method": "equal_weight_proxy",
        "sector": "Defence",
        "stocks": ("HAL.NS", "BEL.NS", "BDL.NS", "COCHINSHIP.NS", "MAZDOCK.NS", "GRSE.NS", "SOLARINDS.NS", "DATAPATTNS.NS", "BEML.NS", "MTARTECH.NS"),
    },
    "digital": {
        "name": "Nifty Digital",
        "index_ticker": "DIGITAL.NS",
        "preferred_price_method": "equal_weight_proxy",
        "sector": "Digital",
        "stocks": ("NYKAA.NS", "PAYTM.NS", "POLICYBZR.NS", "NAUKRI.NS", "AFFLE.NS", "ROUTE.NS", "MAPMYINDIA.NS", "TANLA.NS", "INDIAMART.NS", "TATAELXSI.NS"),
    },
    "tourism": {
        "name": "Nifty Tourism",
        "index_ticker": "TOURISM.NS",
        "preferred_price_method": "equal_weight_proxy",
        "sector": "Tourism",
        "stocks": ("INDHOTEL.NS", "EIHOTEL.NS", "LEMONTREE.NS", "CHALET.NS", "IRCTC.NS", "DEVYANI.NS", "JUBLFOOD.NS", "SAPPHIRE.NS", "EASEMYTRIP.NS", "THOMASCOOK.NS"),
    },
    "cpse": {
        "name": "Nifty CPSE",
        "index_ticker": "CPSEETF.NS",
        "sector": "CPSE",
        "stocks": ("NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "BEL.NS", "HAL.NS", "NHPC.NS", "SJVN.NS", "OIL.NS", "CONCOR.NS"),
    },
    "pse": {
        "name": "Nifty PSE",
        "index_ticker": "CPSEETF.NS",
        "sector": "PSE",
        "stocks": ("SBIN.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "BEL.NS", "HAL.NS", "BHEL.NS", "SAIL.NS", "GAIL.NS"),
    },
}

STOCK_INDUSTRY_OVERRIDES: dict[str, str] = {
    "MARUTI.NS": "Passenger Vehicles",
    "M&M.NS": "SUVs & Tractors",
    "TMPV.NS": "Passenger Vehicles & EV",
    "BAJAJ-AUTO.NS": "Two Wheelers",
    "EICHERMOT.NS": "Two Wheelers",
    "TVSMOTOR.NS": "Two Wheelers",
    "HEROMOTOCO.NS": "Two Wheelers",
    "ASHOKLEY.NS": "Commercial Vehicles",
    "MOTHERSON.NS": "Auto Components",
    "BOSCHLTD.NS": "Auto Components",
    "HDFCBANK.NS": "Private Banks",
    "ICICIBANK.NS": "Private Banks",
    "AXISBANK.NS": "Private Banks",
    "KOTAKBANK.NS": "Private Banks",
    "INDUSINDBK.NS": "Private Banks",
    "IDFCFIRSTB.NS": "Private Banks",
    "SBIN.NS": "PSU Banks",
    "BANKBARODA.NS": "PSU Banks",
    "PNB.NS": "PSU Banks",
    "CANBK.NS": "PSU Banks",
    "UNIONBANK.NS": "PSU Banks",
    "INDIANB.NS": "PSU Banks",
    "BANKINDIA.NS": "PSU Banks",
    "MAHABANK.NS": "PSU Banks",
    "CENTRALBK.NS": "PSU Banks",
    "UCOBANK.NS": "PSU Banks",
    "HINDUNILVR.NS": "Personal Care & Household",
    "ITC.NS": "Cigarettes & Staples",
    "NESTLEIND.NS": "Packaged Foods",
    "BRITANNIA.NS": "Packaged Foods",
    "DABUR.NS": "Personal Care & Household",
    "MARICO.NS": "Personal Care & Household",
    "GODREJCP.NS": "Personal Care & Household",
    "TATACONSUM.NS": "Beverages & Foods",
    "VBL.NS": "Beverages",
    "COLPAL.NS": "Personal Care & Household",
    "TCS.NS": "IT Services",
    "INFY.NS": "IT Services",
    "HCLTECH.NS": "IT Services",
    "WIPRO.NS": "IT Services",
    "TECHM.NS": "IT Services",
    "LTM.NS": "IT Services",
    "PERSISTENT.NS": "Digital Engineering",
    "COFORGE.NS": "Digital Engineering",
    "MPHASIS.NS": "IT Services",
    "LTTS.NS": "Engineering R&D",
    "TATASTEEL.NS": "Steel",
    "JSWSTEEL.NS": "Steel",
    "HINDALCO.NS": "Non-Ferrous Metals",
    "VEDL.NS": "Non-Ferrous Metals",
    "JINDALSTEL.NS": "Steel",
    "SAIL.NS": "Steel",
    "NMDC.NS": "Mining",
    "NATIONALUM.NS": "Non-Ferrous Metals",
    "HINDZINC.NS": "Non-Ferrous Metals",
    "COALINDIA.NS": "Mining",
    "SUNPHARMA.NS": "Pharmaceuticals",
    "CIPLA.NS": "Pharmaceuticals",
    "DRREDDY.NS": "Pharmaceuticals",
    "DIVISLAB.NS": "Life Sciences & Ingredients",
    "LUPIN.NS": "Pharmaceuticals",
    "AUROPHARMA.NS": "Pharmaceuticals",
    "MANKIND.NS": "Pharmaceuticals",
    "ZYDUSLIFE.NS": "Pharmaceuticals",
    "GLENMARK.NS": "Pharmaceuticals",
    "BIOCON.NS": "Biotechnology",
    "DLF.NS": "Residential Developers",
    "GODREJPROP.NS": "Residential Developers",
    "LODHA.NS": "Residential Developers",
    "OBEROIRLTY.NS": "Residential Developers",
    "PHOENIXLTD.NS": "Commercial Real Estate",
    "PRESTIGE.NS": "Residential Developers",
    "BRIGADE.NS": "Residential Developers",
    "SOBHA.NS": "Residential Developers",
    "ANANTRAJ.NS": "Real Estate & Data Centers",
    "SIGNATURE.NS": "Residential Developers",
    "RELIANCE.NS": "Integrated Oil, Gas & Retail",
    "ONGC.NS": "Upstream Oil & Gas",
    "NTPC.NS": "Power Utilities",
    "POWERGRID.NS": "Power Transmission",
    "BPCL.NS": "Refining & Marketing",
    "IOC.NS": "Refining & Marketing",
    "GAIL.NS": "Gas Utilities",
    "TATAPOWER.NS": "Power Utilities",
    "ADANIGREEN.NS": "Renewable Energy",
    "LT.NS": "EPC & Construction",
    "NCC.NS": "EPC & Construction",
    "IRB.NS": "Roads & Toll",
    "RVNL.NS": "Rail Infrastructure",
    "IRCON.NS": "Rail Infrastructure",
    "KPIL.NS": "Power & Infra EPC",
    "KEC.NS": "Power & Infra EPC",
    "PNCINFRA.NS": "Roads & Construction",
    "KNRCON.NS": "Roads & Construction",
    "HGINFRA.NS": "Roads & Construction",
    "SUNTV.NS": "Broadcasting",
    "ZEEL.NS": "Broadcasting",
    "PVRINOX.NS": "Cinema Exhibition",
    "NETWORK18.NS": "Broadcasting",
    "SAREGAMA.NS": "Music & IP",
    "NAZARA.NS": "Gaming",
    "TITAN.NS": "Jewellery & Lifestyle",
    "TRENT.NS": "Retail",
    "DMART.NS": "Retail",
    "INDHOTEL.NS": "Hotels",
    "JUBLFOOD.NS": "Restaurants",
    "KALYANKJIL.NS": "Jewellery & Lifestyle",
    "ETERNAL.NS": "Digital Consumer",
    "NYKAA.NS": "Beauty Retail",
    "BATAINDIA.NS": "Footwear",
    "OIL.NS": "Upstream Oil & Gas",
    "PETRONET.NS": "Gas Utilities",
    "MGL.NS": "City Gas Distribution",
    "IGL.NS": "City Gas Distribution",
    "GUJGASLTD.NS": "City Gas Distribution",
    "ATGL.NS": "City Gas Distribution",
    "CASTROLIND.NS": "Lubricants",
}


def list_industry_definitions(*, universe: str = "local", refresh_universe: bool = False) -> dict[str, Any]:
    """Return configured industry groups and stock universes."""
    if _normalize_universe(universe) != "local":
        definitions = list_universe_industry_definitions(universe=universe, refresh=refresh_universe)
        if definitions:
            return definitions
    return {
        industry_id: {
            "id": industry_id,
            "name": definition["name"],
            "sector": definition["sector"],
            "stocks": list(definition["stocks"]),
        }
        for industry_id, definition in INDUSTRY_DEFINITIONS.items()
    }


def list_rrg_index_definitions() -> dict[str, Any]:
    """Return the dedicated RRG index universe, including proxy-backed Nifty themes."""
    return {
        index_id: {
            "id": index_id,
            "name": definition["name"],
            "index_ticker": definition["index_ticker"],
            "index_tickers": list(definition.get("index_tickers") or (definition["index_ticker"],)),
            "sector": definition["sector"],
            "stocks": list(definition.get("stocks", ())),
            "price_method": "nse_index_or_equal_weight_proxy_fallback",
        }
        for index_id, definition in _rrg_index_definitions().items()
    }


def get_market_indices(
    *,
    period: str = "1y",
    interval: str = "1d",
    max_indices: int | None = None,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Return broad/sector index performance, trend, and relative strength."""
    benchmark_prices = get_price_history(BENCHMARK_TICKER, period=period, interval=interval, force_refresh=force_refresh_prices)
    benchmark_metrics = _price_metrics(benchmark_prices)
    rows = [
        _index_row(
            "nifty_50",
            "Nifty 50",
            BENCHMARK_TICKER,
            "Broad Market",
            benchmark_metrics,
            benchmark_metrics,
        )
    ]
    warnings = []

    for sector_id, definition in _rrg_index_definitions().items():
        prices = _rrg_price_history(definition, period=period, interval=interval, force_refresh_prices=force_refresh_prices)
        metrics = _price_metrics(prices)
        if not metrics:
            warnings.append(f"No index data available for {definition['name']}.")
            continue
        rows.append(
            _index_row(
                sector_id,
                definition["name"],
                metrics.get("selected_ticker") or definition["index_ticker"],
                definition["sector"],
                metrics,
                benchmark_metrics,
            )
        )

    ranked = sorted(rows, key=lambda row: _number(row.get("return_20d"), -999), reverse=True)
    if max_indices is not None:
        ranked = ranked[: max(0, int(max_indices))]
    return _json_safe(
        {
            "period": period,
            "interval": interval,
            "benchmark": {"ticker": BENCHMARK_TICKER, "metrics": benchmark_metrics},
            "force_refresh_prices": force_refresh_prices,
            "indices": ranked,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "methodology": "Latest available free provider OHLCV data. Rows are ranked by 20D return by default and include 1D/5D/20D/60D/120D returns, RS vs Nifty, trend, RSI, ADX, and moving-average state.",
        }
    )


def get_industry_analytics(
    *,
    period: str = "1y",
    interval: str = "1d",
    min_stocks: int = 3,
    weighting: str = "equal",
    include_fundamentals: bool = False,
    universe: str = "local",
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
    max_universe_stocks: int | None = None,
) -> dict[str, Any]:
    """Rank industries across multiple timeframes for top-down stock research."""
    if _normalize_universe(universe) != "local":
        return _get_broad_industry_analytics(
            period=period,
            interval=interval,
            min_stocks=min_stocks,
            weighting=weighting,
            universe=universe,
            refresh_universe=refresh_universe,
            force_refresh_prices=force_refresh_prices,
            max_universe_stocks=max_universe_stocks,
        )

    benchmark_metrics = _price_metrics(get_price_history(BENCHMARK_TICKER, period=period, interval=interval, force_refresh=force_refresh_prices))
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for industry_id, definition in INDUSTRY_DEFINITIONS.items():
        stock_rows = _industry_stock_metrics(
            definition["stocks"],
            period=period,
            interval=interval,
            weighting=weighting,
            include_fundamentals=include_fundamentals,
            force_refresh_prices=force_refresh_prices,
        )
        if len(stock_rows) < max(1, int(min_stocks)):
            warnings.append(f"Skipped {definition['name']} because only {len(stock_rows)} stocks had price data.")
            continue
        row = _industry_row(industry_id, definition, stock_rows, benchmark_metrics, weighting)
        rows.append(row)

    _apply_ranks(rows, "return_1d", "rank_1d")
    _apply_ranks(rows, "return_5d", "rank_1w")
    _apply_ranks(rows, "return_20d", "rank_1m")
    _apply_ranks(rows, "return_60d", "rank_3m")
    total = len(rows)
    for row in rows:
        row["composite_score"] = _industry_score(row, total)
        row["stage"] = _industry_stage(row, total)
        row["root_causes"] = _industry_root_causes(row)

    ranked = sorted(rows, key=lambda item: item["composite_score"], reverse=True)
    return _json_safe(
        {
            "period": period,
            "interval": interval,
            "weighting": weighting,
            "force_refresh_prices": force_refresh_prices,
            "industries": ranked,
            "top_industry": ranked[0] if ranked else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "methodology": "Industries are local NSE stock groups. Equal weighting averages constituent returns; market-cap weighting uses available free market-cap fundamentals when enabled. Ranks cover 1D, 1W, 1M, and 3M movement.",
        }
    )


def get_sector_analytics(
    *,
    mode: str = "near_52w_high",
    period: str = "2y",
    interval: str = "1d",
    ma_period: int = 200,
    ma_type: str | None = None,
    rs_cutoff: float = 80,
    near_high_pct: float = 5,
    selected_sector: str | None = None,
    max_stocks: int = 12,
    universe: str = "local",
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
    max_universe_stocks: int | None = None,
) -> dict[str, Any]:
    """Return ChartsMaze-style sector breadth and drill-down using local free data."""
    normalized_mode = _normalize_sector_analytics_mode(mode)
    ma_filter = _normalize_ma_filter(ma_period=ma_period, ma_type=ma_type)
    rs_cutoff = max(0.0, min(100.0, float(rs_cutoff)))
    near_high_pct = max(0.0, float(near_high_pct))
    if _normalize_universe(universe) != "local":
        return _get_broad_sector_analytics(
            mode=normalized_mode,
            period=period,
            interval=interval,
            ma_filter=ma_filter,
            rs_cutoff=rs_cutoff,
            near_high_pct=near_high_pct,
            selected_sector=selected_sector,
            max_stocks=max_stocks,
            universe=universe,
            refresh_universe=refresh_universe,
            force_refresh_prices=force_refresh_prices,
            max_universe_stocks=max_universe_stocks,
        )

    benchmark_metrics = _price_metrics(get_price_history(BENCHMARK_TICKER, period=period, interval=interval, force_refresh=force_refresh_prices))
    sectors = []
    stock_rows_by_sector: dict[str, list[dict[str, Any]]] = {}
    warnings = []

    for sector_id, definition in SECTOR_DEFINITIONS.items():
        stock_rows = _sector_analytics_stock_rows(
            sector_id,
            definition,
            benchmark_metrics=benchmark_metrics,
            period=period,
            interval=interval,
            mode=normalized_mode,
            ma_filter=ma_filter,
            rs_cutoff=rs_cutoff,
            near_high_pct=near_high_pct,
            force_refresh_prices=force_refresh_prices,
        )
        if not stock_rows:
            warnings.append(f"No stock data available for {definition['name']}.")
            continue
        stock_rows_by_sector[sector_id] = stock_rows
        sectors.append(_sector_analytics_row(sector_id, definition, stock_rows, normalized_mode, ma_filter, near_high_pct))

    sectors = _rank_sector_analytics_rows(sectors)
    selected_sector_id = _resolve_sector_analytics_selection(selected_sector, sectors)
    selected_row = next((row for row in sectors if row["sector_id"] == selected_sector_id), sectors[0] if sectors else None)
    all_stock_rows = [row for sector_rows in stock_rows_by_sector.values() for row in sector_rows]
    drilldowns = _sector_analytics_drilldowns_by_sector(
        stock_rows_by_sector,
        sectors,
        normalized_mode,
        ma_filter,
        max_stocks=max_stocks,
    )
    industries = drilldowns["industries_by_sector"].get(selected_sector_id or "", [])
    stocks = drilldowns["stocks_by_sector"].get(selected_sector_id or "", [])
    constituent_stocks = drilldowns["constituent_stocks_by_sector"].get(selected_sector_id or "", [])

    return _json_safe(
        {
            "mode": normalized_mode,
            "mode_label": SECTOR_ANALYTICS_MODES[normalized_mode],
            "metric_label": _sector_analytics_metric_label(normalized_mode, ma_filter, rs_cutoff, near_high_pct),
            "filter": {
                "ma_period": ma_filter["period"],
                "ma_type": ma_filter["label"],
                "rs_cutoff": rs_cutoff,
                "near_high_pct": near_high_pct,
            },
            "period": period,
            "interval": interval,
            "calculation_version": SECTOR_ANALYTICS_VERSION,
            "force_refresh_prices": force_refresh_prices,
            "price_history_end_date": _latest_stock_date(all_stock_rows),
            "min_sector_stocks_for_ranking": MIN_SECTOR_STOCKS_FOR_RANKING,
            "sectors": sectors,
            "top_sector": sectors[0] if sectors else None,
            "selected_sector": selected_row,
            "selected_sector_id": selected_sector_id,
            "industries": industries,
            "stocks": stocks,
            "constituent_stocks": constituent_stocks,
            "industries_by_sector": drilldowns["industries_by_sector"],
            "stocks_by_sector": drilldowns["stocks_by_sector"],
            "constituent_stocks_by_sector": drilldowns["constituent_stocks_by_sector"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "methodology": _sector_analytics_methodology(normalized_mode, ma_filter, rs_cutoff, near_high_pct),
            "column_explanations": _sector_analytics_column_explanations(),
        }
    )


def _get_broad_sector_analytics(
    *,
    mode: str,
    period: str,
    interval: str,
    ma_filter: dict[str, Any],
    rs_cutoff: float,
    near_high_pct: float,
    selected_sector: str | None,
    max_stocks: int,
    universe: str,
    refresh_universe: bool,
    force_refresh_prices: bool,
    max_universe_stocks: int | None,
) -> dict[str, Any]:
    normalized_universe = _normalize_universe(universe)
    universe_df = load_stock_universe(universe=universe, refresh=refresh_universe, max_stocks=max_universe_stocks)
    if universe_df.empty:
        result = get_sector_analytics(
            mode=mode,
            period=period,
            interval=interval,
            ma_period=ma_filter["period"],
            ma_type=ma_filter["label"],
            rs_cutoff=rs_cutoff,
            near_high_pct=near_high_pct,
            selected_sector=selected_sector,
            max_stocks=max_stocks,
            universe="local",
            force_refresh_prices=force_refresh_prices,
        )
        result["warnings"] = ["Broad NSE universe is unavailable; fell back to configured local baskets.", *result.get("warnings", [])]
        result["universe"] = "local_fallback"
        return result

    benchmark_metrics = _price_metrics(get_price_history(BENCHMARK_TICKER, period=period, interval=interval, force_refresh=force_refresh_prices))
    stock_rows = _broad_sector_stock_rows(
        universe_df,
        benchmark_metrics=benchmark_metrics,
        period=period,
        interval=interval,
        mode=mode,
        ma_filter=ma_filter,
        rs_cutoff=rs_cutoff,
        near_high_pct=near_high_pct,
        force_refresh_prices=force_refresh_prices,
    )
    sectors: list[dict[str, Any]] = []
    stock_rows_by_sector: dict[str, list[dict[str, Any]]] = {}
    for sector_name, sector_rows in _group_rows(stock_rows, "sector").items():
        sector_id = _industry_slug(sector_name)
        definition = {"name": sector_name, "sector": sector_name}
        stock_rows_by_sector[sector_id] = sector_rows
        sectors.append(_sector_analytics_row(sector_id, definition, sector_rows, mode, ma_filter, near_high_pct))

    sectors = _rank_sector_analytics_rows(sectors)
    selected_sector_id = _resolve_sector_analytics_selection(selected_sector, sectors)
    selected_row = next((row for row in sectors if row["sector_id"] == selected_sector_id), sectors[0] if sectors else None)
    drilldowns = _sector_analytics_drilldowns_by_sector(
        stock_rows_by_sector,
        sectors,
        mode,
        ma_filter,
        max_stocks=max_stocks,
    )
    industries = drilldowns["industries_by_sector"].get(selected_sector_id or "", [])
    stocks = drilldowns["stocks_by_sector"].get(selected_sector_id or "", [])
    constituent_stocks = drilldowns["constituent_stocks_by_sector"].get(selected_sector_id or "", [])
    return _json_safe(
        {
            "mode": mode,
            "mode_label": SECTOR_ANALYTICS_MODES[mode],
            "metric_label": _sector_analytics_metric_label(mode, ma_filter, rs_cutoff, near_high_pct),
            "filter": {
                "ma_period": ma_filter["period"],
                "ma_type": ma_filter["label"],
                "rs_cutoff": rs_cutoff,
                "near_high_pct": near_high_pct,
            },
            "period": period,
            "interval": interval,
            "calculation_version": SECTOR_ANALYTICS_VERSION,
            "force_refresh_prices": force_refresh_prices,
            "universe": normalized_universe,
            "universe_source": (
                "Local data/sectors CSV taxonomy for sector and industry membership + free Yahoo history when required"
                if normalized_universe == "full_nse"
                else "NSE NIFTY TOTAL MARKET snapshot + free Yahoo history when required"
            ),
            "universe_refreshed_at": _latest_universe_refresh(universe_df),
            "price_history_end_date": _latest_stock_date(stock_rows),
            "min_sector_stocks_for_ranking": MIN_SECTOR_STOCKS_FOR_RANKING,
            "universe_stock_count": int(len(universe_df)),
            "sectors": sectors,
            "top_sector": sectors[0] if sectors else None,
            "selected_sector": selected_row,
            "selected_sector_id": selected_sector_id,
            "industries": industries,
            "stocks": stocks,
            "constituent_stocks": constituent_stocks,
            "industries_by_sector": drilldowns["industries_by_sector"],
            "stocks_by_sector": drilldowns["stocks_by_sector"],
            "constituent_stocks_by_sector": drilldowns["constituent_stocks_by_sector"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": [] if stock_rows else ["No stock rows could be calculated from the broad NSE universe."],
            "methodology": _sector_analytics_methodology(mode, ma_filter, rs_cutoff, near_high_pct)
            + " Broad/full mode groups NSE stocks by NSE basic-industry metadata instead of small configured Nifty baskets. "
            + f"Sectors with fewer than {MIN_SECTOR_STOCKS_FOR_RANKING} stocks are shown for transparency but are not used as top-sector recommendations.",
            "column_explanations": _sector_analytics_column_explanations(),
        }
    )


def _get_broad_industry_analytics(
    *,
    period: str,
    interval: str,
    min_stocks: int,
    weighting: str,
    universe: str,
    refresh_universe: bool,
    force_refresh_prices: bool,
    max_universe_stocks: int | None,
) -> dict[str, Any]:
    universe_df = load_stock_universe(universe=universe, refresh=refresh_universe, max_stocks=max_universe_stocks)
    if universe_df.empty:
        result = get_industry_analytics(
            period=period,
            interval=interval,
            min_stocks=min_stocks,
            weighting=weighting,
            universe="local",
            force_refresh_prices=force_refresh_prices,
        )
        result["warnings"] = ["Broad NSE universe is unavailable; fell back to configured local industry groups.", *result.get("warnings", [])]
        result["universe"] = "local_fallback"
        return result

    benchmark_metrics = _price_metrics(get_price_history(BENCHMARK_TICKER, period=period, interval=interval, force_refresh=force_refresh_prices))
    price_map = get_price_histories(list(universe_df["ticker"]), period=period, interval=interval, force_refresh=force_refresh_prices)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for industry_name, industry_df in universe_df.groupby("basic_industry", dropna=False):
        stock_rows = []
        for record in industry_df.to_dict(orient="records"):
            metrics = _price_metrics(price_map.get(record["ticker"]))
            if not metrics:
                continue
            stock_rows.append(
                {
                    "ticker": record["ticker"],
                    "market_cap": _number(record.get("free_float_market_cap"), None),
                    **metrics,
                }
            )
        if len(stock_rows) < max(1, int(min_stocks)):
            continue
        sector_name = str(industry_df["sector"].iloc[0] or "Unclassified")
        definition = {
            "name": str(industry_name or "Unclassified"),
            "sector": sector_name,
            "stocks": tuple(industry_df["ticker"]),
        }
        rows.append(_industry_row(_industry_slug(definition["name"]), definition, stock_rows, benchmark_metrics, weighting))

    if not rows:
        warnings.append("No broad industry groups had enough price data for the selected filters.")
    _apply_ranks(rows, "return_1d", "rank_1d")
    _apply_ranks(rows, "return_5d", "rank_1w")
    _apply_ranks(rows, "return_20d", "rank_1m")
    _apply_ranks(rows, "return_60d", "rank_3m")
    total = len(rows)
    for row in rows:
        row["composite_score"] = _industry_score(row, total)
        row["stage"] = _industry_stage(row, total)
        row["root_causes"] = _industry_root_causes(row)

    ranked = sorted(rows, key=lambda item: item["composite_score"], reverse=True)
    return _json_safe(
        {
            "period": period,
            "interval": interval,
            "weighting": weighting,
            "force_refresh_prices": force_refresh_prices,
            "universe": "broad",
            "universe_source": "NSE NIFTY TOTAL MARKET metadata + batched free Yahoo history",
            "universe_stock_count": int(len(universe_df)),
            "industries": ranked,
            "top_industry": ranked[0] if ranked else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "methodology": "Industries are grouped from the broad NSE total-market universe using NSE basic-industry metadata. Equal weighting averages constituent returns; market-cap weighting uses NSE free-float market cap when available.",
        }
    )


def rank_industry_stocks(
    industry: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    max_stocks: int = 10,
    include_fundamentals: bool = True,
    universe: str = "local",
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Rank stocks inside one industry by relative strength and setup quality."""
    if _normalize_universe(universe) != "local":
        return _rank_broad_industry_stocks(
            industry,
            period=period,
            interval=interval,
            max_stocks=max_stocks,
            include_fundamentals=include_fundamentals,
            universe=universe,
            refresh_universe=refresh_universe,
            force_refresh_prices=force_refresh_prices,
        )

    industry_id, definition = _resolve_industry(industry)
    stock_metrics = _industry_stock_metrics(
        definition["stocks"],
        period=period,
        interval=interval,
        weighting="equal",
        include_fundamentals=False,
        force_refresh_prices=force_refresh_prices,
    )
    industry_metrics = _aggregate_metrics(stock_metrics, "equal")
    ranked = []
    warnings = []
    for ticker in list(definition["stocks"])[: max(1, int(max_stocks))]:
        prices = get_price_history(ticker, period=period, interval=interval, force_refresh=force_refresh_prices)
        metrics = _price_metrics(prices)
        if not metrics:
            warnings.append(f"No price history available for {ticker}.")
            continue
        indicators = latest_indicators(prices)
        patterns = detect_chart_patterns(prices)
        fundamentals = get_basic_fundamentals(ticker) if include_fundamentals else {}
        ranked.append(_score_sector_stock(ticker, metrics, industry_metrics, indicators, patterns, fundamentals))

    ranked = sorted(ranked, key=lambda row: row["stock_score"], reverse=True)
    return _json_safe(
        {
            "industry_id": industry_id,
            "industry_name": definition["name"],
            "sector": definition["sector"],
            "period": period,
            "interval": interval,
            "force_refresh_prices": force_refresh_prices,
            "stocks": ranked,
            "top_stock": ranked[0] if ranked else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        }
    )


def _rank_broad_industry_stocks(
    industry: str,
    *,
    period: str,
    interval: str,
    max_stocks: int,
    include_fundamentals: bool,
    universe: str,
    refresh_universe: bool,
    force_refresh_prices: bool,
) -> dict[str, Any]:
    universe_df = load_stock_universe(universe=universe, refresh=refresh_universe)
    industry_id, industry_df = _resolve_broad_industry(industry, universe_df)
    tickers = list(industry_df["ticker"])
    price_map = get_price_histories(tickers, period=period, interval=interval, force_refresh=force_refresh_prices)
    stock_metrics = []
    for ticker in tickers:
        metrics = _price_metrics(price_map.get(ticker))
        if metrics:
            stock_metrics.append({"ticker": ticker, **metrics})
    industry_metrics = _aggregate_metrics(stock_metrics, "equal")
    ranked = []
    warnings = []
    for row in stock_metrics:
        ticker = row["ticker"]
        indicators = latest_indicators(price_map.get(ticker))
        patterns = detect_chart_patterns(price_map.get(ticker))
        fundamentals = get_basic_fundamentals(ticker) if include_fundamentals else {}
        ranked.append(_score_sector_stock(ticker, row, industry_metrics, indicators, patterns, fundamentals))
    if not ranked:
        warnings.append(f"No price history available for broad industry {industry}.")
    ranked = sorted(ranked, key=lambda row: row["stock_score"], reverse=True)[: max(1, int(max_stocks))]
    return _json_safe(
        {
            "industry_id": industry_id,
            "industry_name": str(industry_df["basic_industry"].iloc[0]),
            "sector": str(industry_df["sector"].iloc[0]),
            "period": period,
            "interval": interval,
            "force_refresh_prices": force_refresh_prices,
            "universe": "broad",
            "stocks": ranked,
            "top_stock": ranked[0] if ranked else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        }
    )


def get_market_breadth(
    *,
    period: str = "1y",
    interval: str = "1d",
    max_stocks: int | None = None,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Measure market health using stocks above key averages and positive return windows."""
    rows = _universe_stock_rows(period=period, interval=interval, max_stocks=max_stocks, force_refresh_prices=force_refresh_prices)
    summary = _breadth_summary(rows)
    groups = []
    by_sector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sector[row["sector"]].append(row)
    for sector, sector_rows in sorted(by_sector.items()):
        group = _breadth_summary(sector_rows)
        group["sector"] = sector
        groups.append(group)

    return _json_safe(
        {
            "period": period,
            "interval": interval,
            "force_refresh_prices": force_refresh_prices,
            "summary": summary,
            "sectors": sorted(groups, key=lambda row: _number(row.get("above_50_pct"), -1), reverse=True),
            "stocks": rows,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "methodology": "Counts unique configured NSE stocks above 20/50/200 DMA, positive over 1D/5D/20D, and near 52-week high. This is a participation/market-health view, not a prediction.",
        }
    )


def get_moving_average_crossover_scan(
    *,
    period: str = "2y",
    interval: str = "1d",
    universe: str = "full_nse",
    direction: str = "bullish",
    lookback_periods: int = 20,
    min_market_cap_cr: float = 0.0,
    max_rows: int = 150,
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
    max_universe_stocks: int | None = None,
) -> dict[str, Any]:
    """Scan the selected universe for fresh 50-DMA / 200-DMA crossover events."""
    normalized_universe = _normalize_universe(universe)
    normalized_direction = _normalize_crossover_direction(direction)
    warnings: list[str] = []

    if normalized_universe == "local":
        records = _local_universe_records()
        universe_refreshed_at = None
        universe_source = "Configured local industry baskets"
    else:
        universe_df = load_stock_universe(
            universe=normalized_universe,
            refresh=refresh_universe,
            max_stocks=max_universe_stocks,
        )
        if universe_df.empty:
            records = _local_universe_records()
            universe_refreshed_at = None
            universe_source = "Configured local industry baskets fallback"
            warnings.append("Requested stock universe was unavailable; fell back to configured local baskets.")
        else:
            records = universe_df.to_dict(orient="records")
            universe_refreshed_at = _latest_universe_refresh(universe_df)
            universe_source = (
                "Local data/sectors CSV taxonomy + free NSE/BSE/Yahoo price history"
                if normalized_universe == "full_nse"
                else "NSE/BSE public universe metadata + free NSE/BSE/Yahoo price history"
            )

    if max_universe_stocks is not None and normalized_universe == "local":
        records = records[: max(0, int(max_universe_stocks))]

    tickers = [str(record.get("ticker") or "").strip().upper() for record in records if str(record.get("ticker") or "").strip()]
    price_map = get_price_histories(tickers, period=period, interval=interval, force_refresh=force_refresh_prices)

    source_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for record in records:
        ticker = str(record.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        prices = price_map.get(ticker)
        metrics = _price_metrics(prices)
        if not metrics:
            continue

        row = _snapshot_stock_row(record)
        row.update(metrics)
        row["market_cap_cr"] = _market_cap_crore(record.get("free_float_market_cap"))
        event = _latest_50_200_crossover_event(
            prices,
            direction=normalized_direction,
            lookback_periods=lookback_periods,
        )
        if event:
            row.update(event)
        source_rows.append(row)

    _apply_percentile_rs_ratings(source_rows)

    min_market_cap = max(0.0, float(min_market_cap_cr or 0.0))
    for row in source_rows:
        if not row.get("has_50_200_cross"):
            continue
        market_cap = _number(row.get("market_cap_cr"), None)
        if min_market_cap and market_cap is not None and market_cap < min_market_cap:
            continue
        signal_rows.append(_ma_crossover_display_row(row))

    signal_rows = sorted(
        signal_rows,
        key=lambda row: (
            _number(row.get("periods_since_cross"), 9999),
            -_number(row.get("rs_rating"), 0),
            -_number(row.get("return_since_cross_pct"), -999),
        ),
    )
    limited_rows = signal_rows[: max(1, int(max_rows))]
    sector_summary = _ma_crossover_group_summary(signal_rows, "sector")
    industry_summary = _ma_crossover_group_summary(signal_rows, "industry")

    eligible_rows = [
        row
        for row in source_rows
        if _number(row.get("sma_50"), None) is not None and _number(row.get("sma_200"), None) is not None
    ]
    return _json_safe(
        {
            "period": period,
            "interval": interval,
            "universe": normalized_universe,
            "universe_source": universe_source,
            "universe_refreshed_at": universe_refreshed_at,
            "universe_stock_count": len(records),
            "price_data_stock_count": len(source_rows),
            "eligible_stock_count": len(eligible_rows),
            "signal_stock_count": len(signal_rows),
            "returned_stock_count": len(limited_rows),
            "direction": normalized_direction,
            "lookback_periods": max(1, int(lookback_periods)),
            "min_market_cap_cr": min_market_cap,
            "force_refresh_prices": force_refresh_prices,
            "price_history_end_date": _latest_stock_date(source_rows),
            "stocks": limited_rows,
            "sector_summary": sector_summary,
            "industry_summary": industry_summary,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "methodology": (
                "Calculates 50-day and 200-day simple moving averages from the selected universe's daily price history. "
                "A bullish crossover is counted when 50-DMA moves from at/below 200-DMA to above it; a bearish crossover is the reverse. "
                "Rows are limited to crosses inside the selected lookback window and are ranked by recency, RS rating, and return since cross."
            ),
        }
    )


def get_top_gainers(
    *,
    period: str = "1y",
    interval: str = "1d",
    return_window: str = "1d",
    min_return_pct: float = 5.0,
    market_cap_min: float = 1000.0,
    min_industry_stocks: int = 3,
    max_rows: int = 50,
    max_industries: int = 20,
    universe: str = "full_nse",
    refresh_universe: bool = False,
    force_refresh_prices: bool = True,
    max_universe_stocks: int | None = None,
) -> dict[str, Any]:
    """Rank top gaining stocks and summarize which industries are driving the move."""
    key = RETURN_WINDOWS.get(str(return_window).lower(), "return_5d")
    normalized_universe = _normalize_universe(universe)
    warnings: list[str] = []

    if normalized_universe == "local":
        source_rows = _universe_stock_rows(period=period, interval=interval, max_stocks=max_universe_stocks, force_refresh_prices=force_refresh_prices)
        universe_refreshed_at = None
        universe_stock_count = len(source_rows)
        universe_source = "Configured local industry baskets"
    else:
        universe_df = load_stock_universe(
            universe=normalized_universe,
            refresh=refresh_universe,
            max_stocks=max_universe_stocks,
        )
        if universe_df.empty:
            source_rows = _universe_stock_rows(
                period=period,
                interval=interval,
                max_stocks=max_universe_stocks,
                force_refresh_prices=force_refresh_prices,
            )
            universe_refreshed_at = None
            universe_stock_count = len(source_rows)
            universe_source = "Configured local industry baskets fallback"
            warnings.append("Requested stock universe was unavailable; fell back to configured local baskets.")
        else:
            price_map = get_price_histories(
                list(universe_df["ticker"]),
                period=period,
                interval=interval,
                force_refresh=force_refresh_prices,
            )
            source_rows = []
            for record in universe_df.to_dict(orient="records"):
                row = _snapshot_stock_row(record)
                metrics = _price_metrics(price_map.get(record["ticker"]))
                if metrics:
                    row.update(metrics)
                row["market_cap_cr"] = _market_cap_crore(record.get("free_float_market_cap"))
                row["price_data_available"] = bool(metrics)
                source_rows.append(row)
            _apply_percentile_rs_ratings(source_rows)
            universe_refreshed_at = _latest_universe_refresh(universe_df)
            universe_stock_count = int(len(universe_df))
            universe_source = (
                "Local data/sectors CSV taxonomy for sector and industry membership + free NSE/BSE/Yahoo price history"
                if normalized_universe == "full_nse"
                else "NSE/BSE public universe metadata + free NSE/BSE/Yahoo price history"
            )

    _apply_percentile_rs_ratings(source_rows)
    for row in source_rows:
        row.setdefault("market_cap_cr", _market_cap_crore(row.get("free_float_market_cap") or row.get("market_cap")))

    eligible_rows = []
    market_cap_missing = 0
    for row in source_rows:
        ret = _number(row.get(key), None)
        if ret is None:
            continue
        market_cap_cr = _number(row.get("market_cap_cr"), None)
        if market_cap_cr is None:
            market_cap_missing += 1
        if market_cap_cr is not None and market_cap_min > 0 and market_cap_cr < float(market_cap_min):
            continue
        item = dict(row)
        item["selected_return"] = ret
        item["selected_return_pct"] = round(ret * 100, 2)
        item["market_cap_filter_pass"] = market_cap_cr is None or market_cap_cr >= float(market_cap_min)
        eligible_rows.append(item)

    if market_cap_missing and market_cap_min > 0:
        warnings.append(
            "Market-cap filter was applied where public universe market-cap data exists; rows without market-cap data were kept."
        )

    gainers = []
    for row in eligible_rows:
        ret = _number(row.get("selected_return"), None)
        if ret is None or ret * 100 < float(min_return_pct):
            continue
        item = dict(row)
        gainers.append(item)

    gainers = sorted(gainers, key=lambda row: _number(row.get("selected_return"), -999), reverse=True)
    ranked = gainers[: max(1, int(max_rows))]
    industry_totals = _top_gainer_industry_totals(eligible_rows)
    industry_summary, performers_by_industry = _top_gainer_industry_breakdown(
        gainers,
        industry_totals,
        min_industry_stocks=max(1, int(min_industry_stocks)),
        max_industries=max(1, int(max_industries)),
    )

    return _json_safe(
        {
            "period": period,
            "interval": interval,
            "return_window": return_window,
            "return_field": key,
            "min_return_pct": min_return_pct,
            "market_cap_min": market_cap_min,
            "min_industry_stocks": min_industry_stocks,
            "force_refresh_prices": force_refresh_prices,
            "universe": normalized_universe,
            "universe_source": universe_source,
            "universe_refreshed_at": universe_refreshed_at,
            "universe_stock_count": universe_stock_count,
            "eligible_stock_count": len(eligible_rows),
            "gainer_stock_count": len(gainers),
            "price_history_end_date": _latest_stock_date(source_rows),
            "stocks": ranked,
            "industry_summary": industry_summary,
            "performers_by_industry": performers_by_industry,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "methodology": (
                "Ranks stocks by selected return window, filters by stock return and available public market-cap data, "
                "then groups the winners by basic industry. Industry % = passing gainers in that industry / eligible "
                "stocks in that industry, matching the ChartsMaze-style top-gainers concept while using our own data."
            ),
        }
    )


def _market_cap_crore(value: Any) -> float | None:
    number = _number(value, None)
    if number is None:
        return None
    if number > 1_000_000:
        return round(number / 10_000_000, 2)
    return round(number, 2)


def _top_gainer_industry_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        industry = str(row.get("industry") or row.get("basic_industry") or "Unclassified")
        totals[industry] += 1
    return totals


def _top_gainer_industry_breakdown(
    gainers: list[dict[str, Any]],
    industry_totals: dict[str, int],
    *,
    min_industry_stocks: int,
    max_industries: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_industry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gainers:
        by_industry[str(row.get("industry") or row.get("basic_industry") or "Unclassified")].append(row)

    summary = []
    performers_by_industry: dict[str, list[dict[str, Any]]] = {}
    for industry, rows in by_industry.items():
        sorted_rows = sorted(rows, key=lambda row: _number(row.get("selected_return"), -999), reverse=True)
        performers_by_industry[industry] = sorted_rows
        passing_count = len(sorted_rows)
        if passing_count < min_industry_stocks:
            continue
        total_count = max(1, int(industry_totals.get(industry, passing_count)))
        top_row = sorted_rows[0]
        industry_gainer_pct = 100 * passing_count / total_count
        summary.append(
            {
                "industry": industry,
                "sector": top_row.get("sector"),
                "passing_count": passing_count,
                "eligible_count": total_count,
                "stock_count": passing_count,
                "industry_gainer_pct": round(industry_gainer_pct, 2),
                "avg_selected_return_pct": round(100 * _average(row.get("selected_return") for row in sorted_rows), 2),
                "top_return_pct": round(100 * (_number(top_row.get("selected_return"), 0) or 0), 2),
                "top_stock": top_row.get("ticker"),
                "label": f"{industry}({industry_gainer_pct:.1f}%)",
                "formula": f"{passing_count} / {total_count}",
            }
        )

    summary = sorted(
        summary,
        key=lambda row: (
            _number(row.get("top_return_pct"), -999),
            _number(row.get("avg_selected_return_pct"), -999),
            _number(row.get("passing_count"), -999),
        ),
        reverse=True,
    )[:max_industries]
    return summary, performers_by_industry


def get_relative_rotation_graph(
    *,
    period: str = "1y",
    interval: str = "1d",
    benchmark: str = BENCHMARK_TICKER,
    trail_length: int = 5,
    selected_sectors: list[str] | str | None = None,
    zone: list[str] | str | None = None,
    max_sectors: int | None = None,
    force_refresh_prices: bool = True,
) -> dict[str, Any]:
    """Return a ChartsMaze-style RRG sector map with recent rotation trails."""
    benchmark_ticker, benchmark_name = _resolve_rrg_benchmark(benchmark)
    trail_length = max(1, min(int(trail_length or 5), 30))
    selected_sector_ids = _normalize_rrg_sector_selection(selected_sectors)
    selected_zones = _normalize_rrg_zones(zone)
    benchmark_prices = _rrg_benchmark_price_history(
        benchmark_ticker,
        benchmark_name,
        period=period,
        interval=interval,
        force_refresh_prices=force_refresh_prices,
    )
    benchmark_close = _rrg_close_series(benchmark_prices)
    warnings: list[str] = []
    points: list[dict[str, Any]] = []
    trails: list[dict[str, Any]] = []

    if benchmark_close.empty:
        warnings.append(f"No benchmark price history available for {benchmark_name} ({benchmark_ticker}).")

    for sector_id, definition in _rrg_index_definitions().items():
        if selected_sector_ids and sector_id not in selected_sector_ids:
            continue
        prices = _rrg_price_history(definition, period=period, interval=interval, force_refresh_prices=force_refresh_prices)
        trail_points = _rrg_sector_trail(
            sector_id,
            definition,
            prices,
            benchmark_close,
            trail_length=trail_length,
            interval=interval,
        )
        if not trail_points:
            warnings.append(f"No RRG trail could be calculated for {definition['name']}.")
            continue
        current = trail_points[-1]
        if selected_zones and current["quadrant"] not in selected_zones:
            continue
        metrics = _price_metrics(prices)
        metrics["rrg_price_method"] = prices.attrs.get("rrg_price_method")
        point = _rrg_current_point(sector_id, definition, current, trail_points, metrics)
        points.append(point)
        trails.append(
            {
                "sector_id": sector_id,
                "name": definition["name"],
                "sector": definition["sector"],
                "current_quadrant": current["quadrant"],
                "status": _rrg_status(current, trail_points),
                "direction": _rrg_direction(trail_points),
                "points": trail_points,
            }
        )

    points = sorted(points, key=lambda row: _number(row.get("rotation_score"), 0), reverse=True)
    if max_sectors is not None:
        allowed = {row["sector_id"] for row in points[: max(0, int(max_sectors))]}
        points = [row for row in points if row["sector_id"] in allowed]
        trails = [row for row in trails if row["sector_id"] in allowed]

    quadrant_counts = _rrg_quadrant_counts(points)
    recommendation_points = _rrg_recommendation_points(points)
    leading = [row for row in recommendation_points if row["quadrant"] == "Leading"]
    improving = [row for row in recommendation_points if row["quadrant"] == "Improving"]

    return _json_safe(
        {
            "period": period,
            "interval": interval,
            "benchmark": benchmark_ticker,
            "benchmark_name": benchmark_name,
            "force_refresh_prices": force_refresh_prices,
            "trail_length": trail_length,
            "selected_zones": sorted(selected_zones) if selected_zones else ["Leading", "Improving", "Lagging", "Weakening"],
            "points": points,
            "trails": trails,
            "quadrant_counts": quadrant_counts,
            "top_leading_sector": leading[0] if leading else None,
            "top_upcoming_sector": improving[0] if improving else None,
            "end_date": trails[0]["points"][-1]["date"] if trails else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "calculation_version": "rrg_nse_index_calibrated_v5",
            "methodology": (
                "ChartsMaze-style RRG built from the app's own free price pipeline. "
                f"RS-Ratio is 100 x relative strength divided by its {RRG_RATIO_LOOKBACK}-period moving average. "
                f"RS-Momentum is 100 x RS-Ratio divided by its {RRG_MOMENTUM_LOOKBACK}-period moving average. "
                "Leading = RS and momentum above 100; Weakening = RS above 100 but momentum below; "
                "Improving = RS below 100 but momentum above; Lagging = both below 100. "
                "Top leader/upcoming metrics exclude broad-market proxy rows so the headline focuses on sector rotation. "
                "NSE index-close archives are preferred for every Nifty index row; equal-weight stock proxies are used only when an official index series is unavailable. "
                "This is an open JdK-style approximation calibrated against visible ChartsMaze daily RRG values; exact values can still differ because ChartsMaze may use proprietary smoothing, normalization, and holiday/date handling."
            ),
            "column_explanations": _rrg_column_explanations(),
            "warnings": warnings,
        }
    )


def _index_row(
    index_id: str,
    name: str,
    ticker: str,
    category: str,
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    rel20 = _number(metrics.get("return_20d")) - _number(benchmark.get("return_20d"))
    rel60 = _number(metrics.get("return_60d")) - _number(benchmark.get("return_60d"))
    rel120 = _number(metrics.get("return_120d")) - _number(benchmark.get("return_120d"))
    return {
        "id": index_id,
        "name": name,
        "ticker": ticker,
        "category": category,
        "close": metrics.get("close"),
        "return_1d": metrics.get("return_1d"),
        "return_5d": metrics.get("return_5d"),
        "return_20d": metrics.get("return_20d"),
        "return_60d": metrics.get("return_60d"),
        "return_120d": metrics.get("return_120d"),
        "rs_20d": round(rel20, 4),
        "rs_60d": round(rel60, 4),
        "relative_strength_score": _score_relative_strength(rel20, rel60, rel120),
        "trend_score": _score_trend(metrics),
        "rsi_14": metrics.get("rsi_14"),
        "adx_14": metrics.get("adx_14"),
        "above_sma_20": metrics.get("above_sma_20"),
        "above_sma_50": metrics.get("above_sma_50"),
        "above_sma_200": metrics.get("above_sma_200"),
        "distance_from_52w_high": metrics.get("distance_from_52w_high"),
        "provider": metrics.get("provider"),
        "data_points": metrics.get("data_points"),
    }


def _broad_sector_stock_rows(
    universe_df: pd.DataFrame,
    *,
    benchmark_metrics: dict[str, Any],
    period: str,
    interval: str,
    mode: str,
    ma_filter: dict[str, Any],
    rs_cutoff: float,
    near_high_pct: float,
    force_refresh_prices: bool,
) -> list[dict[str, Any]]:
    price_map = get_price_histories(list(universe_df["ticker"]), period=period, interval=interval, force_refresh=force_refresh_prices)
    rows = []
    for record in universe_df.to_dict(orient="records"):
        row = _snapshot_stock_row(record)
        metrics = _price_metrics(price_map.get(record["ticker"]))
        if metrics:
            row.update(metrics)
        row["rs_rating"] = None
        rows.append(row)

    _apply_percentile_rs_ratings(rows)

    for row in rows:
        passed, criterion_value, criterion_label = _sector_analytics_pass(
            row,
            mode=mode,
            ma_filter=ma_filter,
            rs_cutoff=rs_cutoff,
            near_high_pct=near_high_pct,
            rs_rating=_number(row.get("rs_rating"), 0),
        )
        row["passes_filter"] = passed
        row["criterion_value"] = criterion_value
        row["criterion_label"] = criterion_label
    return rows


def _snapshot_stock_row(record: dict[str, Any]) -> dict[str, Any]:
    near_high = _number(record.get("near_52w_high_pct"), None)
    return_30d = _number(record.get("return_30d_pct"), None)
    return_365d = _number(record.get("return_365d_pct"), None)
    year_high = _number(record.get("year_high"), None)
    year_low = _number(record.get("year_low"), None)
    market_cap = _number(record.get("free_float_market_cap"), None)
    basic_industry = record.get("basic_industry") or record.get("industry") or "Unclassified"
    industry_source = record.get("classification_source") or record.get("source") or "nse_broad_universe"
    return {
        "ticker": record.get("ticker"),
        "symbol": record.get("symbol") or str(record.get("ticker") or "").replace(".NS", ""),
        "name": record.get("name"),
        "isin": record.get("isin"),
        "sector_id": _industry_slug(record.get("sector")),
        "sector": record.get("sector"),
        "industry_id": _industry_slug(basic_industry),
        "industry": basic_industry,
        "basic_industry": basic_industry,
        "industry_source": industry_source,
        "market_cap": market_cap,
        "free_float_market_cap": market_cap,
        "close": _number(record.get("last_price"), None),
        "latest_date": None,
        "universe_refreshed_at": record.get("refreshed_at"),
        "data_points": 1,
        "return_1d": None,
        "return_5d": None,
        "return_20d": None if return_30d is None else round(return_30d / 100, 4),
        "return_60d": None,
        "return_120d": None,
        "return_252d": None if return_365d is None else round(return_365d / 100, 4),
        "high_52w": year_high,
        "low_52w": year_low,
        "distance_from_52w_high": None if near_high is None else round(-near_high / 100, 4),
        "sma_20": None,
        "sma_50": None,
        "sma_100": None,
        "sma_200": None,
        "ema_20": None,
        "ema_21": None,
        "ema_50": None,
        "ema_100": None,
        "ema_200": None,
        "above_sma_20": None,
        "above_sma_50": None,
        "above_sma_100": None,
        "above_sma_200": None,
        "above_ema_20": None,
        "above_ema_21": None,
        "above_ema_50": None,
        "above_ema_100": None,
        "above_ema_200": None,
        "provider": "nse_snapshot",
    }


def _apply_percentile_rs_ratings(rows: list[dict[str, Any]]) -> None:
    sortable = []
    for row in rows:
        score = _stock_rs_raw_score(row)
        row["rs_raw_score"] = round(score, 6) if score is not None else None
        row["rs_rating"] = None
        if score is not None:
            sortable.append((score, row))

    if not sortable:
        return

    sortable.sort(key=lambda item: item[0])
    if len(sortable) == 1:
        sortable[0][1]["rs_rating"] = 50
        return

    total = len(sortable) - 1
    for index, (_, row) in enumerate(sortable):
        rating = RS_RATING_MIN + (RS_RATING_MAX - RS_RATING_MIN) * index / total
        row["rs_rating"] = int(max(RS_RATING_MIN, min(RS_RATING_MAX, round(rating))))


def _stock_rs_raw_score(row: dict[str, Any]) -> float | None:
    weighted_score = 0.0
    used_weight = 0.0
    for field, weight in RS_RATING_WEIGHTS:
        value = _number(row.get(field), None)
        if value is None:
            continue
        weighted_score += value * weight
        used_weight += weight
    if used_weight <= 0:
        return None
    return weighted_score / used_weight


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "Unclassified")].append(row)
    return grouped


def _sector_analytics_stock_rows(
    sector_id: str,
    definition: dict[str, Any],
    *,
    benchmark_metrics: dict[str, Any],
    period: str,
    interval: str,
    mode: str,
    ma_filter: dict[str, Any],
    rs_cutoff: float,
    near_high_pct: float,
    force_refresh_prices: bool,
) -> list[dict[str, Any]]:
    ticker_meta = _ticker_metadata()
    rows = []
    for ticker in definition["stocks"]:
        metrics = _price_metrics(get_price_history(ticker, period=period, interval=interval, force_refresh=force_refresh_prices))
        if not metrics:
            continue
        industry_meta = _sector_stock_industry_meta(ticker, sector_id, definition, ticker_meta)
        rows.append(
            {
                "ticker": ticker,
                "symbol": ticker.replace(".NS", ""),
                "name": None,
                "sector_id": sector_id,
                "sector": definition["sector"],
                "industry_id": industry_meta["industry_id"],
                "industry": industry_meta["industry"],
                "basic_industry": industry_meta["industry"],
                "industry_source": industry_meta["industry_source"],
                "market_cap": None,
                "passes_filter": False,
                "criterion_value": None,
                "criterion_label": None,
                "rs_rating": None,
                **metrics,
            }
        )

    _apply_percentile_rs_ratings(rows)
    for row in rows:
        passed, criterion_value, criterion_label = _sector_analytics_pass(
            row,
            mode=mode,
            ma_filter=ma_filter,
            rs_cutoff=rs_cutoff,
            near_high_pct=near_high_pct,
            rs_rating=_number(row.get("rs_rating"), 0),
        )
        row["passes_filter"] = passed
        row["criterion_value"] = criterion_value
        row["criterion_label"] = criterion_label
    return rows


def _sector_stock_industry_meta(
    ticker: str,
    sector_id: str,
    definition: dict[str, Any],
    ticker_meta: dict[str, dict[str, str]],
) -> dict[str, str]:
    override = STOCK_INDUSTRY_OVERRIDES.get(ticker)
    if override:
        return {
            "industry_id": _industry_slug(override),
            "industry": override,
            "sector": definition["sector"],
            "industry_source": "local_nse_industry_map",
        }
    fallback = ticker_meta.get(
        ticker,
        {
            "industry_id": f"{sector_id}_unclassified",
            "industry": definition["sector"],
            "sector": definition["sector"],
        },
    )
    return {
        "industry_id": fallback["industry_id"],
        "industry": fallback["industry"],
        "sector": fallback["sector"],
        "industry_source": "local_static_group",
    }


def _sector_analytics_row(
    sector_id: str,
    definition: dict[str, Any],
    stock_rows: list[dict[str, Any]],
    mode: str,
    ma_filter: dict[str, Any],
    near_high_pct: float,
) -> dict[str, Any]:
    eligible_rows = _sector_analytics_eligible_rows(stock_rows, mode, ma_filter)
    passing = [row for row in eligible_rows if row["passes_filter"]]
    metric_pct = 100 * len(passing) / len(eligible_rows) if eligible_rows else 0
    ranking_eligible = _sector_ranking_eligible(definition["name"], len(eligible_rows))
    return {
        "sector_id": sector_id,
        "name": definition["name"],
        "sector": definition["sector"],
        "mode": mode,
        "metric_pct": round(metric_pct, 2),
        "ranking_eligible": ranking_eligible,
        "sample_confidence": _sector_sample_confidence(len(eligible_rows)),
        "ranking_note": None
        if ranking_eligible
        else f"Shown for transparency; not used as top sector because eligible stock count is below {MIN_SECTOR_STOCKS_FOR_RANKING} or sector is non-standard.",
        "passing_count": len(passing),
        "eligible_count": len(eligible_rows),
        "stock_count": len(stock_rows),
        "coverage_pct": round(100 * len(eligible_rows) / len(stock_rows), 2) if stock_rows else 0,
        "avg_rs_rating": round(_average(row.get("rs_rating") for row in stock_rows), 2),
        "avg_return_20d_pct": round(100 * _average(row.get("return_20d") for row in stock_rows), 2),
        "avg_return_60d_pct": round(100 * _average(row.get("return_60d") for row in stock_rows), 2),
        "above_20_pct": round(_percent(row.get("above_sma_20") for row in stock_rows), 2),
        "above_50_pct": round(_percent(row.get("above_sma_50") for row in stock_rows), 2),
        "above_100_pct": round(_percent(row.get("above_sma_100") for row in stock_rows), 2),
        "above_200_pct": round(_percent(row.get("above_sma_200") for row in stock_rows), 2),
        "above_21_ema_pct": round(_percent(row.get("above_ema_21") for row in stock_rows), 2),
        "near_52w_high_pct": round(
            _percent(_number(row.get("distance_from_52w_high"), -1) >= -(near_high_pct / 100) for row in stock_rows),
            2,
        ),
        "stage": _sector_analytics_stage(metric_pct),
    }


def _sector_analytics_eligible_rows(
    stock_rows: list[dict[str, Any]],
    mode: str,
    ma_filter: dict[str, Any],
) -> list[dict[str, Any]]:
    if mode == "moving_average":
        value_field = ma_filter["value_field"]
        return [row for row in stock_rows if _number(row.get(value_field), None) is not None]
    if mode == "relative_strength":
        return [row for row in stock_rows if _number(row.get("rs_rating"), None) is not None]
    return [row for row in stock_rows if _number(row.get("distance_from_52w_high"), None) is not None]


def _sector_industry_contribution(
    stock_rows: list[dict[str, Any]],
    selected_sector: dict[str, Any] | None,
    mode: str,
    ma_filter: dict[str, Any],
) -> list[dict[str, Any]]:
    if not stock_rows or not selected_sector:
        return []
    eligible_sector_rows = _sector_analytics_eligible_rows(stock_rows, mode, ma_filter)
    eligible_tickers = {row["ticker"] for row in eligible_sector_rows}
    by_industry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stock_rows:
        by_industry[row["industry"]].append(row)
    sector_eligible_count = max(1, len(eligible_sector_rows))
    rows = []
    for industry, rows_for_industry in by_industry.items():
        eligible_rows = [row for row in rows_for_industry if row["ticker"] in eligible_tickers]
        passing = [row for row in eligible_rows if row["passes_filter"]]
        rows.append(
            {
                "industry": industry,
                "sector": selected_sector.get("sector"),
                "stock_count": len(rows_for_industry),
                "eligible_count": len(eligible_rows),
                "passing_count": len(passing),
                "contribution_pct": round(100 * len(passing) / sector_eligible_count, 2),
                "pass_pct_within_industry": round(100 * len(passing) / len(eligible_rows), 2) if eligible_rows else 0,
                "avg_rs_rating": round(_average(row.get("rs_rating") for row in rows_for_industry), 2),
                "top_stock": _top_sector_stock(rows_for_industry).get("ticker"),
                "passing_tickers": [row["ticker"] for row in passing],
                "eligible_tickers": [row["ticker"] for row in eligible_rows],
                "stock_tickers": [row["ticker"] for row in rows_for_industry],
                "formula": f"{len(passing)} / {sector_eligible_count}",
                "industry_source": rows_for_industry[0].get("industry_source"),
            }
        )
    return sorted(rows, key=lambda row: (row["contribution_pct"], row["avg_rs_rating"]), reverse=True)


def _sector_analytics_drilldowns_by_sector(
    stock_rows_by_sector: dict[str, list[dict[str, Any]]],
    sectors: list[dict[str, Any]],
    mode: str,
    ma_filter: dict[str, Any],
    *,
    max_stocks: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    sector_by_id = {str(row.get("sector_id")): row for row in sectors}
    industries_by_sector: dict[str, list[dict[str, Any]]] = {}
    stocks_by_sector: dict[str, list[dict[str, Any]]] = {}
    constituent_stocks_by_sector: dict[str, list[dict[str, Any]]] = {}
    for sector_id, stock_rows in stock_rows_by_sector.items():
        selected_sector = sector_by_id.get(str(sector_id))
        if not selected_sector:
            continue
        industries_by_sector[sector_id] = _sector_industry_contribution(stock_rows, selected_sector, mode, ma_filter)
        stocks_by_sector[sector_id] = _sector_strong_stocks(stock_rows, mode, max_stocks=max_stocks)
        constituent_stocks_by_sector[sector_id] = _sector_constituent_stocks(stock_rows, mode)
    return {
        "industries_by_sector": industries_by_sector,
        "stocks_by_sector": stocks_by_sector,
        "constituent_stocks_by_sector": constituent_stocks_by_sector,
    }


def _rank_sector_analytics_rows(sectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        sectors,
        key=lambda row: (
            bool(row.get("ranking_eligible")),
            _number(row.get("metric_pct"), 0),
            _number(row.get("eligible_count"), 0),
            _number(row.get("stock_count"), 0),
            _number(row.get("avg_rs_rating"), 0),
        ),
        reverse=True,
    )


def _sector_ranking_eligible(sector_name: str, stock_count: int) -> bool:
    return str(sector_name or "Unclassified") not in NON_RANKING_SECTORS and int(stock_count or 0) >= MIN_SECTOR_STOCKS_FOR_RANKING


def _sector_sample_confidence(stock_count: int) -> str:
    if stock_count < MIN_SECTOR_STOCKS_FOR_RANKING:
        return "Low"
    if stock_count < 15:
        return "Medium"
    return "High"


def _latest_universe_refresh(universe_df: pd.DataFrame) -> str | None:
    if universe_df.empty or "refreshed_at" not in universe_df.columns:
        return None
    values = [str(value) for value in universe_df["refreshed_at"].dropna().tolist() if str(value)]
    return max(values) if values else None


def _latest_stock_date(stock_rows: list[dict[str, Any]]) -> str | None:
    values = [str(row.get("latest_date")) for row in stock_rows if row.get("latest_date")]
    return max(values) if values else None


def _sector_strong_stocks(stock_rows: list[dict[str, Any]], mode: str, *, max_stocks: int) -> list[dict[str, Any]]:
    ranked = sorted(
        stock_rows,
        key=lambda row: (
            bool(row.get("passes_filter")),
            _number(row.get("rs_rating"), 0),
            _sector_stock_sort_value(row, mode),
        ),
        reverse=True,
    )
    return [
        {
            "ticker": row.get("ticker"),
            "symbol": row.get("symbol"),
            "name": row.get("name"),
            "industry": row.get("industry"),
            "basic_industry": row.get("basic_industry") or row.get("industry"),
            "sector": row.get("sector"),
            "market_cap": row.get("market_cap"),
            "latest_date": row.get("latest_date"),
            "passes_filter": row.get("passes_filter"),
            "criterion": row.get("criterion_label"),
            "criterion_value": row.get("criterion_value"),
            "rs_rating": row.get("rs_rating"),
            "return_1d_pct": round(100 * _number(row.get("return_1d"), 0), 2),
            "return_5d_pct": round(100 * _number(row.get("return_5d"), 0), 2),
            "return_20d_pct": round(100 * _number(row.get("return_20d")), 2),
            "return_60d_pct": round(100 * _number(row.get("return_60d")), 2),
            "sma_20": row.get("sma_20"),
            "sma_50": row.get("sma_50"),
            "sma_100": row.get("sma_100"),
            "sma_200": row.get("sma_200"),
            "ema_21": row.get("ema_21"),
            "ema_50": row.get("ema_50"),
            "ema_100": row.get("ema_100"),
            "ema_200": row.get("ema_200"),
            "above_sma_50": row.get("above_sma_50"),
            "above_sma_200": row.get("above_sma_200"),
            "above_ema_21": row.get("above_ema_21"),
            "distance_from_52w_high_pct": round(100 * _number(row.get("distance_from_52w_high")), 2),
            "close": row.get("close"),
        }
        for row in ranked[: max(1, int(max_stocks))]
    ]


def _sector_constituent_stocks(stock_rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return _sector_strong_stocks(stock_rows, mode, max_stocks=max(1, len(stock_rows)))


def _sector_analytics_pass(
    metrics: dict[str, Any],
    *,
    mode: str,
    ma_filter: dict[str, Any],
    rs_cutoff: float,
    near_high_pct: float,
    rs_rating: float,
) -> tuple[bool, float | None, str]:
    if mode == "moving_average":
        average_value = _number(metrics.get(ma_filter["value_field"]), None)
        return metrics.get(ma_filter["above_field"]) is True, average_value, f"Above {ma_filter['label']}"
    if mode == "relative_strength":
        return rs_rating >= rs_cutoff, rs_rating, f"RS >= {rs_cutoff:g}"
    distance_pct = 100 * _number(metrics.get("distance_from_52w_high"), -999)
    return distance_pct >= -near_high_pct, round(distance_pct, 2), f"Within {near_high_pct:g}% of 52w high"


def _sector_stock_sort_value(row: dict[str, Any], mode: str) -> float:
    if mode == "near_52w_high":
        return _number(row.get("distance_from_52w_high"), -999)
    if mode == "relative_strength":
        return _number(row.get("rs_rating"), 0)
    return _number(row.get("return_20d"), -999)


def _top_sector_stock(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda row: (_number(row.get("rs_rating"), 0), _number(row.get("return_20d"), -999)))


def _normalize_sector_analytics_mode(mode: str) -> str:
    key = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ma": "moving_average",
        "moving_avg": "moving_average",
        "moving_average": "moving_average",
        "rs": "relative_strength",
        "relative_strength": "relative_strength",
        "near_high": "near_52w_high",
        "near_52_week_high": "near_52w_high",
        "near_52w_high": "near_52w_high",
        "52w_high": "near_52w_high",
    }
    if key not in aliases:
        expected = ", ".join(SECTOR_ANALYTICS_MODES)
        raise ValueError(f"Unknown sector analytics mode: {mode}. Expected one of: {expected}")
    return aliases[key]


def _normalize_universe(universe: str) -> str:
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


def _normalize_crossover_direction(direction: str) -> str:
    key = str(direction or "bullish").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bull": "bullish",
        "bullish": "bullish",
        "golden": "bullish",
        "golden_cross": "bullish",
        "50_above_200": "bullish",
        "bear": "bearish",
        "bearish": "bearish",
        "death": "bearish",
        "death_cross": "bearish",
        "50_below_200": "bearish",
        "both": "both",
        "all": "both",
        "any": "both",
    }
    if key not in aliases:
        raise ValueError("direction must be bullish, bearish, or both.")
    return aliases[key]


def _local_universe_records() -> list[dict[str, Any]]:
    rows = []
    for ticker, meta in _ticker_metadata().items():
        industry = meta.get("industry") or "Unclassified"
        sector = meta.get("sector") or "Unclassified"
        rows.append(
            {
                "ticker": ticker,
                "symbol": ticker.replace(".NS", "").replace(".BO", ""),
                "name": None,
                "sector": sector,
                "industry": industry,
                "basic_industry": industry,
                "active": True,
                "free_float_market_cap": None,
                "source": "local_static_group",
                "classification_source": "local_static_group",
                "refreshed_at": None,
            }
        )
    return rows


def _latest_50_200_crossover_event(
    df: pd.DataFrame | None,
    *,
    direction: str,
    lookback_periods: int,
) -> dict[str, Any]:
    if df is None or df.empty or "close" not in df.columns:
        return {}
    clean = df.dropna(subset=["close"]).copy()
    if len(clean) < 200:
        return {}

    indicators = add_indicators(clean)
    required = {"close", "sma_50", "sma_200", "sma_50_200_cross"}
    if not required.issubset(indicators.columns):
        return {}
    indicators = indicators.dropna(subset=["close", "sma_50", "sma_200"]).reset_index(drop=True)
    if indicators.empty:
        return {}

    signal_values = {"bullish": {1}, "bearish": {-1}, "both": {1, -1}}[direction]
    lookback = max(1, int(lookback_periods or 1))
    scan = indicators.tail(lookback).copy()
    signals = pd.to_numeric(scan["sma_50_200_cross"], errors="coerce")
    matches = scan[signals.isin(signal_values)]
    if matches.empty:
        return {}

    event = matches.iloc[-1]
    latest = indicators.iloc[-1]
    event_position = int(matches.index[-1])
    signal_value = int(_number(event.get("sma_50_200_cross"), 0))
    latest_sma_50 = _number(latest.get("sma_50"), None)
    latest_sma_200 = _number(latest.get("sma_200"), None)
    event_sma_50 = _number(event.get("sma_50"), None)
    event_sma_200 = _number(event.get("sma_200"), None)
    latest_close = _number(latest.get("close"), None)
    event_close = _number(event.get("close"), None)
    return_since_cross = None
    if latest_close is not None and event_close not in {None, 0}:
        return_since_cross = round(latest_close / event_close - 1, 4)

    return {
        "has_50_200_cross": True,
        "cross_direction": "bullish" if signal_value == 1 else "bearish",
        "cross_signal": "Golden Cross" if signal_value == 1 else "Death Cross",
        "cross_date": _row_iso_date(event),
        "periods_since_cross": max(0, len(indicators) - 1 - event_position),
        "cross_close": event_close,
        "cross_sma_50": event_sma_50,
        "cross_sma_200": event_sma_200,
        "cross_sma_gap_pct": _ma_gap_pct(event_sma_50, event_sma_200),
        "latest_sma_gap_pct": _ma_gap_pct(latest_sma_50, latest_sma_200),
        "latest_50_200_state": "50-DMA above 200-DMA"
        if latest_sma_50 is not None and latest_sma_200 is not None and latest_sma_50 > latest_sma_200
        else "50-DMA below 200-DMA",
        "return_since_cross": return_since_cross,
    }


def _row_iso_date(row: pd.Series) -> str | None:
    if "date" not in row:
        return None
    value = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.isna(value):
        return None
    return value.date().isoformat()


def _ma_gap_pct(fast: Any, slow: Any) -> float | None:
    fast_value = _number(fast, None)
    slow_value = _number(slow, None)
    if fast_value is None or slow_value in {None, 0}:
        return None
    return round(100 * (fast_value / slow_value - 1), 2)


def _ma_crossover_display_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": row.get("ticker"),
        "symbol": row.get("symbol"),
        "name": row.get("name"),
        "sector": row.get("sector"),
        "industry": row.get("industry"),
        "market_cap_cr": row.get("market_cap_cr"),
        "signal": row.get("cross_signal"),
        "direction": row.get("cross_direction"),
        "cross_date": row.get("cross_date"),
        "periods_since_cross": row.get("periods_since_cross"),
        "latest_date": row.get("latest_date"),
        "close": row.get("close"),
        "cross_close": row.get("cross_close"),
        "return_since_cross_pct": round(100 * _number(row.get("return_since_cross")), 2),
        "sma_50": row.get("sma_50"),
        "sma_200": row.get("sma_200"),
        "sma_gap_pct": row.get("latest_sma_gap_pct"),
        "cross_sma_gap_pct": row.get("cross_sma_gap_pct"),
        "current_state": row.get("latest_50_200_state"),
        "rs_rating": row.get("rs_rating"),
        "return_20d_pct": round(100 * _number(row.get("return_20d")), 2),
        "return_60d_pct": round(100 * _number(row.get("return_60d")), 2),
        "above_sma_50": row.get("above_sma_50"),
        "above_sma_200": row.get("above_sma_200"),
        "provider": row.get("provider"),
        "data_points": row.get("data_points"),
    }


def _ma_crossover_group_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "Unclassified")].append(row)
    output = []
    for group, group_rows in grouped.items():
        bullish = [row for row in group_rows if row.get("direction") == "bullish"]
        bearish = [row for row in group_rows if row.get("direction") == "bearish"]
        dates = [str(row.get("cross_date")) for row in group_rows if row.get("cross_date")]
        output.append(
            {
                group_key: group,
                "signal_count": len(group_rows),
                "bullish_count": len(bullish),
                "bearish_count": len(bearish),
                "avg_rs_rating": round(_average(row.get("rs_rating") for row in group_rows), 2),
                "avg_return_since_cross_pct": round(_average(row.get("return_since_cross_pct") for row in group_rows), 2),
                "latest_cross_date": max(dates) if dates else None,
                "top_stock": group_rows[0].get("ticker"),
            }
        )
    return sorted(output, key=lambda row: (row["signal_count"], row["avg_rs_rating"]), reverse=True)


def _normalize_ma_period(ma_period: int) -> int:
    value = int(ma_period)
    if value not in {20, 21, 50, 100, 200}:
        raise ValueError("ma_period must be one of 20, 21, 50, 100, or 200.")
    return value


def _normalize_ma_filter(*, ma_period: int, ma_type: str | None = None) -> dict[str, Any]:
    if ma_type:
        key = str(ma_type).strip().lower().replace("-", " ").replace("_", " ")
        key = " ".join(key.split())
        aliases = {
            "20": ("sma", 20),
            "20 ma": ("sma", 20),
            "20 sma": ("sma", 20),
            "20 dma": ("sma", 20),
            "50": ("sma", 50),
            "50 ma": ("sma", 50),
            "50 sma": ("sma", 50),
            "50 dma": ("sma", 50),
            "100": ("sma", 100),
            "100 ma": ("sma", 100),
            "100 sma": ("sma", 100),
            "100 dma": ("sma", 100),
            "200": ("sma", 200),
            "200 ma": ("sma", 200),
            "200 sma": ("sma", 200),
            "200 dma": ("sma", 200),
            "21": ("ema", 21),
            "21 ma": ("ema", 21),
            "21 ema": ("ema", 21),
            "ema 21": ("ema", 21),
        }
        if key not in aliases:
            raise ValueError("ma_type must be one of 200 MA, 50 MA, 20 MA, or 21 EMA.")
        average_kind, period = aliases[key]
    else:
        period = _normalize_ma_period(ma_period)
        average_kind = "ema" if period == 21 else "sma"

    label = f"{period} {'EMA' if average_kind == 'ema' else 'MA'}"
    return {
        "kind": average_kind,
        "period": period,
        "label": label,
        "value_field": f"{average_kind}_{period}",
        "above_field": f"above_{average_kind}_{period}",
    }


def _resolve_sector_analytics_selection(selected_sector: str | None, sectors: list[dict[str, Any]]) -> str | None:
    if not sectors:
        return None
    if not selected_sector:
        return sectors[0]["sector_id"]
    key = selected_sector.strip().lower().replace("-", "_").replace(" ", "_")
    for sector in sectors:
        if key in {sector["sector_id"], sector["name"].lower().replace(" ", "_")}:
            return sector["sector_id"]
    return sectors[0]["sector_id"]


def _sector_analytics_stage(metric_pct: float) -> str:
    if metric_pct >= 50:
        return "Strong"
    if metric_pct >= 30:
        return "Improving"
    if metric_pct >= 15:
        return "Watch"
    return "Weak"


def _sector_analytics_metric_label(mode: str, ma_filter: dict[str, Any], rs_cutoff: float, near_high_pct: float) -> str:
    if mode == "moving_average":
        return f"% of stocks trading above {ma_filter['label']}"
    if mode == "relative_strength":
        return f"% of stocks trading above RS {rs_cutoff:g}"
    return f"% of stocks trading within {near_high_pct:g}% of 52w high"


def _sector_analytics_methodology(mode: str, ma_filter: dict[str, Any], rs_cutoff: float, near_high_pct: float) -> str:
    if mode == "moving_average":
        average_name = "exponential moving average" if ma_filter["kind"] == "ema" else "simple moving average"
        rule = (
            f"counts each sector's stocks whose latest close is above the {ma_filter['period']}-day {average_name}; "
            "stocks without enough history for that moving average are excluded from the percentage denominator"
        )
    elif mode == "relative_strength":
        rule = f"counts each sector's stocks whose computed RS rating is at or above {rs_cutoff:g}"
    else:
        rule = f"counts each sector's stocks trading within {near_high_pct:g}% of their 52-week high"
    return (
        f"This view {rule}. Sectors are ranked by passing-stock percentage. "
        "Industry contribution is the number of passing stocks in that industry divided by the selected sector's eligible stock count. "
        "RS is a 1-99 percentile-style rank across the loaded stock universe using 20D, 60D, 120D, and 252D returns. "
        "Stock rows are sorted by pass status, then RS and setup strength. Data comes from the configured free Yahoo-backed price provider."
    )


def _sector_analytics_column_explanations() -> list[dict[str, str]]:
    return [
        {"column": "metric_pct", "meaning": "Percentage of eligible sector constituents passing the selected rule."},
        {"column": "passing_count", "meaning": "Number of stocks passing the selected MA, RS, or 52-week-high filter."},
        {"column": "eligible_count", "meaning": "Number of stocks with enough data for the selected filter. For 200MA, a stock needs a computed 200-day average."},
        {"column": "stock_count", "meaning": "Total constituents found in the selected public universe before filter-specific eligibility."},
        {"column": "coverage_pct", "meaning": "Eligible stock count divided by total stock count."},
        {"column": "avg_rs_rating", "meaning": "Average local 1-99 percentile-style relative-strength rating across sector constituents."},
        {"column": "contribution_pct", "meaning": "Industry pass count divided by selected sector eligible stock count."},
        {"column": "formula", "meaning": "The exact contribution calculation: passing stocks in this industry / eligible stocks in selected sector."},
        {"column": "pass_pct_within_industry", "meaning": "Share of eligible stocks inside that industry that pass the selected rule."},
        {"column": "passing_tickers", "meaning": "Tickers from that industry that passed the selected rule."},
        {"column": "rs_rating", "meaning": "1-99 percentile-style rank from weighted 20D, 60D, 120D, and 252D stock returns across the loaded universe."},
        {"column": "market_cap", "meaning": "Free-float market cap from the public NSE total-market snapshot when available."},
        {"column": "sma_* / ema_*", "meaning": "Latest simple/exponential moving averages calculated from free Yahoo-backed price history."},
        {"column": "distance_from_52w_high_pct", "meaning": "Latest close distance from the 52-week high; 0 means at the high, negative means below it."},
    ]


def _industry_stock_metrics(
    tickers: tuple[str, ...],
    *,
    period: str,
    interval: str,
    weighting: str,
    include_fundamentals: bool,
    force_refresh_prices: bool,
) -> list[dict[str, Any]]:
    rows = []
    for ticker in tickers:
        metrics = _price_metrics(get_price_history(ticker, period=period, interval=interval, force_refresh=force_refresh_prices))
        if not metrics:
            continue
        market_cap = None
        if include_fundamentals or weighting == "market_cap":
            market_cap = _number(get_basic_fundamentals(ticker).get("marketCap"), None)
        rows.append({"ticker": ticker, "market_cap": market_cap, **metrics})
    return rows


def _industry_row(
    industry_id: str,
    definition: dict[str, Any],
    stock_rows: list[dict[str, Any]],
    benchmark_metrics: dict[str, Any],
    weighting: str,
) -> dict[str, Any]:
    metrics = _aggregate_metrics(stock_rows, weighting)
    rel20 = _number(metrics.get("return_20d")) - _number(benchmark_metrics.get("return_20d"))
    rel60 = _number(metrics.get("return_60d")) - _number(benchmark_metrics.get("return_60d"))
    rel120 = _number(metrics.get("return_120d")) - _number(benchmark_metrics.get("return_120d"))
    return {
        "industry_id": industry_id,
        "name": definition["name"],
        "sector": definition["sector"],
        "stock_count": len(stock_rows),
        "configured_stock_count": len(definition["stocks"]),
        "return_1d": metrics.get("return_1d"),
        "return_5d": metrics.get("return_5d"),
        "return_20d": metrics.get("return_20d"),
        "return_60d": metrics.get("return_60d"),
        "return_120d": metrics.get("return_120d"),
        "rs_20d": round(rel20, 4),
        "rs_60d": round(rel60, 4),
        "rs_120d": round(rel120, 4),
        "relative_strength_score": _score_relative_strength(rel20, rel60, rel120),
        "trend_score": _score_trend(metrics),
        "acceleration_score": _score_acceleration(metrics, benchmark_metrics),
        "breadth_score": metrics.get("breadth_score"),
        "above_20_pct": metrics.get("above_20_pct"),
        "above_21_ema_pct": metrics.get("above_21_ema_pct"),
        "above_50_pct": metrics.get("above_50_pct"),
        "above_200_pct": metrics.get("above_200_pct"),
        "positive_20d_pct": metrics.get("positive_20d_pct"),
        "near_52w_high_pct": metrics.get("near_52w_high_pct"),
        "candidate_stocks": list(definition["stocks"]),
    }


def _aggregate_metrics(stock_rows: list[dict[str, Any]], weighting: str) -> dict[str, Any]:
    weights = [max(_number(row.get("market_cap"), 0), 0) for row in stock_rows] if weighting == "market_cap" else []
    return {
        "return_1d": _weighted_average((row.get("return_1d") for row in stock_rows), weights=weights),
        "return_5d": _weighted_average((row.get("return_5d") for row in stock_rows), weights=weights),
        "return_20d": _weighted_average((row.get("return_20d") for row in stock_rows), weights=weights),
        "return_60d": _weighted_average((row.get("return_60d") for row in stock_rows), weights=weights),
        "return_120d": _weighted_average((row.get("return_120d") for row in stock_rows), weights=weights),
        "rsi_14": _average(row.get("rsi_14") for row in stock_rows),
        "adx_14": _average(row.get("adx_14") for row in stock_rows),
        "above_sma_20": _percent(row.get("above_sma_20") for row in stock_rows) >= 50,
        "above_sma_50": _percent(row.get("above_sma_50") for row in stock_rows) >= 50,
        "above_sma_200": _percent(row.get("above_sma_200") for row in stock_rows) >= 50,
        "above_ema_21": _percent(row.get("above_ema_21") for row in stock_rows) >= 50,
        "above_20_pct": round(_percent(row.get("above_sma_20") for row in stock_rows), 2),
        "above_21_ema_pct": round(_percent(row.get("above_ema_21") for row in stock_rows), 2),
        "above_50_pct": round(_percent(row.get("above_sma_50") for row in stock_rows), 2),
        "above_200_pct": round(_percent(row.get("above_sma_200") for row in stock_rows), 2),
        "positive_20d_pct": round(_percent(_number(row.get("return_20d")) > 0 for row in stock_rows), 2),
        "near_52w_high_pct": round(_percent(_number(row.get("distance_from_52w_high"), -1) > -0.1 for row in stock_rows), 2),
        "breadth_score": round(
            0.25 * _percent(row.get("above_sma_20") for row in stock_rows)
            + 0.30 * _percent(row.get("above_sma_50") for row in stock_rows)
            + 0.25 * _percent(row.get("above_sma_200") for row in stock_rows)
            + 0.20 * _percent(_number(row.get("return_20d")) > 0 for row in stock_rows),
            2,
        ),
        "data_points": round(_average(row.get("data_points") for row in stock_rows), 2),
    }


def _apply_ranks(rows: list[dict[str, Any]], value_key: str, rank_key: str) -> None:
    ranked = sorted(rows, key=lambda row: _number(row.get(value_key), -999), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row[rank_key] = rank


def _industry_score(row: dict[str, Any], total: int) -> float:
    score_1w = _rank_score(row.get("rank_1w"), total)
    score_1m = _rank_score(row.get("rank_1m"), total)
    score_3m = _rank_score(row.get("rank_3m"), total)
    return round(
        0.15 * score_1w
        + 0.30 * score_1m
        + 0.20 * score_3m
        + 0.15 * _number(row.get("relative_strength_score"), 50)
        + 0.10 * _number(row.get("acceleration_score"), 50)
        + 0.10 * _number(row.get("breadth_score"), 50),
        2,
    )


def _industry_stage(row: dict[str, Any], total: int) -> str:
    top_quartile = max(1, int(total * 0.25))
    half = max(1, int(total * 0.5))
    rank_1w = int(row.get("rank_1w") or total)
    rank_1m = int(row.get("rank_1m") or total)
    rank_3m = int(row.get("rank_3m") or total)
    if rank_1m <= top_quartile and rank_3m <= top_quartile and _number(row.get("breadth_score")) >= 55:
        return "Proven leadership"
    if rank_1w <= rank_1m <= rank_3m and _number(row.get("return_20d")) > 0:
        return "Gaining traction"
    if rank_3m <= top_quartile and rank_1w > half:
        return "Cooling leader"
    if _number(row.get("return_20d")) < 0 and _number(row.get("return_60d")) < 0 and _number(row.get("breadth_score")) < 45:
        return "Weak/Avoid"
    return "Neutral"


def _industry_root_causes(row: dict[str, Any]) -> list[str]:
    causes = [
        f"1W/1M/3M ranks: {row.get('rank_1w')} / {row.get('rank_1m')} / {row.get('rank_3m')}.",
        f"20D and 60D RS vs Nifty: {_pct(row.get('rs_20d'))} / {_pct(row.get('rs_60d'))}.",
        f"Breadth: {row.get('above_50_pct')}% above 50-DMA and {row.get('positive_20d_pct')}% positive over 20D.",
        f"Acceleration score {row.get('acceleration_score')} and trend score {row.get('trend_score')}.",
    ]
    if row.get("stage") == "Gaining traction":
        causes.append("Shorter timeframe rank is improving versus longer timeframe ranks.")
    elif row.get("stage") == "Proven leadership":
        causes.append("Industry is already ranked strongly on 1M and 3M windows with acceptable breadth.")
    return causes


def _rank_score(rank: Any, total: int) -> float:
    if total <= 1:
        return 100
    return 100 * (total - int(rank)) / (total - 1)


def _resolve_industry(industry: str) -> tuple[str, dict[str, Any]]:
    key = (industry or "").strip().lower().replace("-", "_").replace(" ", "_")
    for industry_id, definition in INDUSTRY_DEFINITIONS.items():
        if key in {industry_id, definition["name"].lower().replace(" ", "_").replace("&", "").replace("__", "_")}:
            return industry_id, definition
    expected = ", ".join(sorted(INDUSTRY_DEFINITIONS))
    raise ValueError(f"Unknown industry: {industry}. Expected one of: {expected}")


def _resolve_broad_industry(industry: str, universe_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if universe_df.empty:
        raise ValueError("Broad NSE universe is unavailable. Refresh the stock universe first.")
    key = _industry_slug(industry)
    industry_df = universe_df[universe_df["basic_industry"].map(_industry_slug) == key].copy()
    if industry_df.empty:
        industry_df = universe_df[universe_df["industry"].map(_industry_slug) == key].copy()
    if industry_df.empty:
        expected = ", ".join(sorted(universe_df["basic_industry"].dropna().astype(str).unique())[:25])
        raise ValueError(f"Unknown broad industry: {industry}. Examples: {expected}")
    return _industry_slug(str(industry_df["basic_industry"].iloc[0])), industry_df


def _universe_stock_rows(
    *,
    period: str,
    interval: str,
    max_stocks: int | None = None,
    force_refresh_prices: bool = True,
) -> list[dict[str, Any]]:
    ticker_meta = _ticker_metadata()
    tickers = list(ticker_meta)
    if max_stocks is not None:
        tickers = tickers[: max(0, int(max_stocks))]
    rows = []
    for ticker in tickers:
        metrics = _price_metrics(get_price_history(ticker, period=period, interval=interval, force_refresh=force_refresh_prices))
        if not metrics:
            continue
        rows.append({"ticker": ticker, **ticker_meta[ticker], **metrics})
    return rows


def _ticker_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for industry_id, definition in INDUSTRY_DEFINITIONS.items():
        for ticker in definition["stocks"]:
            metadata.setdefault(
                ticker,
                {
                    "industry_id": industry_id,
                    "industry": definition["name"],
                    "sector": definition["sector"],
                },
            )
    return metadata


def _industry_slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unclassified"


def _breadth_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stock_count": len(rows),
        "above_20_pct": round(_percent(row.get("above_sma_20") for row in rows), 2),
        "above_50_pct": round(_percent(row.get("above_sma_50") for row in rows), 2),
        "above_200_pct": round(_percent(row.get("above_sma_200") for row in rows), 2),
        "positive_1d_pct": round(_percent(_number(row.get("return_1d")) > 0 for row in rows), 2),
        "positive_5d_pct": round(_percent(_number(row.get("return_5d")) > 0 for row in rows), 2),
        "positive_20d_pct": round(_percent(_number(row.get("return_20d")) > 0 for row in rows), 2),
        "near_52w_high_pct": round(_percent(_number(row.get("distance_from_52w_high"), -1) > -0.1 for row in rows), 2),
        "avg_return_20d_pct": round(100 * _average(row.get("return_20d") for row in rows), 2),
    }


def _rrg_index_definitions() -> dict[str, dict[str, Any]]:
    definitions = {key: dict(value) for key, value in SECTOR_DEFINITIONS.items()}
    definitions.update({key: dict(value) for key, value in RRG_ADDITIONAL_INDEX_DEFINITIONS.items()})
    for definition in definitions.values():
        definition.setdefault("stocks", ())
    return definitions


def _resolve_rrg_benchmark(benchmark: str | None) -> tuple[str, str]:
    key = str(benchmark or BENCHMARK_TICKER).strip()
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "^nsei", "nifty_50", "nifty50", "nifty"}:
        return BENCHMARK_TICKER, "Nifty 50"
    for sector_id, definition in _rrg_index_definitions().items():
        aliases = {
            sector_id,
            str(definition.get("name", "")).lower().replace("-", "_").replace(" ", "_"),
            str(definition.get("index_ticker", "")).lower(),
        }
        if normalized in aliases or key.lower() in aliases:
            return str(definition["index_ticker"]), str(definition["name"])
    return key, key


def _normalize_rrg_sector_selection(selected_sectors: list[str] | str | None) -> set[str]:
    if not selected_sectors:
        return set()
    values = selected_sectors
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",")]
    selected = set()
    for value in values:
        key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        for sector_id, definition in _rrg_index_definitions().items():
            aliases = {
                sector_id,
                str(definition.get("name", "")).lower().replace("-", "_").replace(" ", "_"),
                str(definition.get("sector", "")).lower().replace("-", "_").replace(" ", "_"),
            }
            if key in aliases:
                selected.add(sector_id)
                break
    return selected


def _normalize_rrg_zones(zone: list[str] | str | None) -> set[str]:
    if not zone:
        return set()
    values = zone
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",")]
    valid = {"leading": "Leading", "weakening": "Weakening", "improving": "Improving", "lagging": "Lagging"}
    zones = set()
    for value in values:
        key = str(value or "").strip().lower()
        if key in {"all", "any", ""}:
            return set()
        normalized = valid.get(key)
        if normalized:
            zones.add(normalized)
    return zones


def _rrg_price_history(definition: dict[str, Any], *, period: str, interval: str, force_refresh_prices: bool = True) -> pd.DataFrame:
    prices = _get_index_price_history(definition, period=period, interval=interval)
    if len(_rrg_close_series(prices)) > RRG_MIN_HISTORY_POINTS:
        prices.attrs["rrg_price_method"] = "index_ticker"
        return prices
    proxy = _rrg_equal_weight_proxy(definition.get("stocks", ()), period=period, interval=interval, force_refresh_prices=force_refresh_prices)
    if not proxy.empty:
        proxy.attrs["selected_ticker"] = "equal_weight_proxy"
        proxy.attrs["rrg_price_method"] = "equal_weight_proxy"
    return proxy


def _rrg_benchmark_price_history(
    benchmark_ticker: str,
    benchmark_name: str,
    *,
    period: str,
    interval: str,
    force_refresh_prices: bool = True,
) -> pd.DataFrame:
    prices = get_nse_index_price_history(benchmark_name, period=period, interval=interval)
    if not prices.empty:
        return prices
    return get_price_history(benchmark_ticker, period=period, interval=interval, force_refresh=force_refresh_prices)


def _rrg_equal_weight_proxy(
    tickers: tuple[str, ...] | list[str],
    *,
    period: str,
    interval: str,
    force_refresh_prices: bool = True,
) -> pd.DataFrame:
    series = []
    for ticker in tickers:
        close = _rrg_close_series(get_price_history(ticker, period=period, interval=interval, force_refresh=force_refresh_prices))
        if len(close) <= RRG_MIN_HISTORY_POINTS:
            continue
        first = _number(close.iloc[0], None)
        if first is None or first <= 0:
            continue
        series.append(close / first * 100)
    if not series:
        return pd.DataFrame()
    aligned = pd.concat(series, axis=1).dropna(how="all")
    if aligned.empty:
        return pd.DataFrame()
    proxy_close = aligned.mean(axis=1).dropna()
    if proxy_close.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "date": proxy_close.index,
            "open": proxy_close.values,
            "high": proxy_close.values,
            "low": proxy_close.values,
            "close": proxy_close.values,
            "volume": 0,
        }
    )


def _rrg_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
        return pd.Series(dtype=float)
    clean = df[["date", "close"]].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    clean = clean.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"]).sort_values("date")
    if clean.empty:
        return pd.Series(dtype=float)
    return clean.set_index("date")["close"]


def _rrg_sector_trail(
    sector_id: str,
    definition: dict[str, Any],
    prices: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    trail_length: int,
    interval: str,
) -> list[dict[str, Any]]:
    sector_close = _rrg_close_series(prices)
    if sector_close.empty or benchmark_close.empty:
        return []
    aligned = pd.concat({"sector": sector_close, "benchmark": benchmark_close}, axis=1).dropna()
    min_points = RRG_MIN_HISTORY_POINTS
    if len(aligned) <= min_points:
        return []

    rs_line = (aligned["sector"] / aligned["benchmark"]).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(rs_line) <= min_points:
        return []
    rs_average = rs_line.rolling(RRG_RATIO_LOOKBACK).mean().replace(0, pd.NA)
    rs_ratio = 100 * (rs_line / rs_average)
    rs_ratio = rs_ratio.rolling(RRG_SMOOTH_PERIOD).mean()

    momentum_average = rs_ratio.rolling(RRG_MOMENTUM_LOOKBACK).mean().replace(0, pd.NA)
    rs_momentum = 100 * (rs_ratio / momentum_average)
    rs_momentum = rs_momentum.rolling(RRG_SMOOTH_PERIOD).mean()

    indicators = pd.concat({"sector": aligned["sector"], "benchmark": aligned["benchmark"], "rs_ratio": rs_ratio, "rs_momentum": rs_momentum}, axis=1).dropna()
    if indicators.empty:
        return []

    start_index = max(0, len(indicators) - trail_length)
    short_periods, long_periods = _rrg_relative_return_windows(interval)
    trail = []
    for index in range(start_index, len(indicators)):
        date_value = indicators.index[index]
        history_window = aligned.loc[:date_value]
        rs_20d = _rrg_relative_return(history_window, short_periods)
        rs_60d = _rrg_relative_return(history_window, long_periods)
        if rs_20d is None or rs_60d is None:
            continue
        ratio_value = float(indicators["rs_ratio"].iloc[index])
        momentum_value = float(indicators["rs_momentum"].iloc[index])
        trail.append(
            {
                "date": date_value.isoformat() if hasattr(date_value, "isoformat") else str(date_value),
                "sector_id": sector_id,
                "name": definition["name"],
                "rs_20d_pct": round(rs_20d * 100, 2),
                "rs_60d_pct": round(rs_60d * 100, 2),
                "x_rs_60d_pct": round(ratio_value - 100, 2),
                "y_rs_momentum_pct": round(momentum_value - 100, 2),
                "rs_ratio": round(ratio_value, 2),
                "rs_momentum": round(momentum_value, 2),
                "quadrant": _rrg_quadrant(ratio_value, momentum_value),
            }
        )
    return trail


def _rrg_relative_return_windows(interval: str) -> tuple[int, int]:
    normalized = str(interval or "").strip().lower()
    if normalized in {"1wk", "1w", "weekly"}:
        return 4, 13
    if normalized in {"1mo", "1m", "monthly"}:
        return 1, 3
    return 20, 60


def _rrg_relative_return(window: pd.DataFrame, periods: int) -> float | None:
    if len(window) <= periods:
        return None
    sector_start = _number(window["sector"].iloc[-periods - 1], None)
    sector_end = _number(window["sector"].iloc[-1], None)
    benchmark_start = _number(window["benchmark"].iloc[-periods - 1], None)
    benchmark_end = _number(window["benchmark"].iloc[-1], None)
    if not sector_start or not benchmark_start or sector_end is None or benchmark_end is None:
        return None
    sector_return = sector_end / sector_start - 1
    benchmark_return = benchmark_end / benchmark_start - 1
    return round(sector_return - benchmark_return, 4)


def _rrg_current_point(
    sector_id: str,
    definition: dict[str, Any],
    current: dict[str, Any],
    trail_points: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    rs_20d = _number(current.get("rs_20d_pct")) / 100
    rs_60d = _number(current.get("rs_60d_pct")) / 100
    rs_ratio_delta = (_number(current.get("rs_ratio")) - 100) / 100
    rs_momentum_delta = (_number(current.get("rs_momentum")) - 100) / 100
    price_method = metrics.get("rrg_price_method") or "index_ticker"
    rotation_score = _rrg_rotation_score(rs_20d, rs_60d, rs_ratio_delta, rs_momentum_delta)
    if price_method == "equal_weight_proxy":
        rotation_score = max(0, round(rotation_score - 20, 2))
    return {
        "sector_id": sector_id,
        "name": definition["name"],
        "sector": definition["sector"],
        "date": current.get("date"),
        "x_rs_60d_pct": current.get("x_rs_60d_pct"),
        "y_rs_momentum_pct": current.get("y_rs_momentum_pct"),
        "rs_ratio": current.get("rs_ratio"),
        "rs_momentum": current.get("rs_momentum"),
        "rs_20d_pct": current.get("rs_20d_pct"),
        "rs_60d_pct": current.get("rs_60d_pct"),
        "rotation_score": rotation_score,
        "stage": _rrg_stage(current),
        "quadrant": current.get("quadrant"),
        "status": _rrg_status(current, trail_points),
        "direction": _rrg_direction(trail_points),
        "close": metrics.get("close"),
        "return_20d_pct": round(100 * _number(metrics.get("return_20d")), 2),
        "return_60d_pct": round(100 * _number(metrics.get("return_60d")), 2),
        "trend_score": _score_trend(metrics) if metrics else None,
        "selected_ticker": metrics.get("selected_ticker") or definition.get("index_ticker"),
        "price_method": price_method,
        "data_points": metrics.get("data_points"),
    }


def _rrg_rotation_score(rs_20d: float, rs_60d: float, rs_ratio_delta: float, rs_momentum_delta: float) -> float:
    score = 50 + (rs_ratio_delta * 120) + (rs_momentum_delta * 120) + (rs_60d * 40) + (rs_20d * 30)
    return round(max(0, min(100, score)), 2)


def _rrg_stage(current: dict[str, Any]) -> str:
    quadrant = current.get("quadrant")
    if quadrant == "Leading":
        return "Current leader"
    if quadrant == "Weakening":
        return "Leader losing momentum"
    if quadrant == "Improving":
        return "Upcoming watch"
    return "Weak/Avoid"


def _rrg_status(current: dict[str, Any], trail_points: list[dict[str, Any]]) -> str:
    quadrant = current.get("quadrant")
    direction = _rrg_direction(trail_points)
    if quadrant == "Leading":
        if direction in {"north_east", "east", "north"}:
            return "currently_running"
        if direction in {"south_west", "south", "west"}:
            return "cooling_leader"
        return "leading_watch"
    if quadrant == "Improving":
        return "upcoming"
    if quadrant == "Weakening":
        return "cooling"
    if quadrant == "Lagging" and direction in {"north_east", "north"}:
        return "early_recovery"
    return "avoid_or_weak"


def _rrg_direction(trail_points: list[dict[str, Any]]) -> str:
    if len(trail_points) < 2:
        return "flat"
    start = trail_points[-2]
    end = trail_points[-1]
    dx = _number(end.get("rs_ratio")) - _number(start.get("rs_ratio"))
    dy = _number(end.get("rs_momentum")) - _number(start.get("rs_momentum"))
    threshold = 0.05
    horizontal = "east" if dx > threshold else "west" if dx < -threshold else ""
    vertical = "north" if dy > threshold else "south" if dy < -threshold else ""
    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    return vertical or horizontal or "flat"


def _rrg_quadrant_counts(points: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"Leading": 0, "Improving": 0, "Lagging": 0, "Weakening": 0}
    for point in points:
        quadrant = point.get("quadrant")
        if quadrant in counts:
            counts[quadrant] += 1
    return counts


def _rrg_recommendation_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        point
        for point in points
        if point.get("sector") != "Broad Market" and point.get("price_method") != "equal_weight_proxy"
    ]
    return candidates or points


def _rrg_column_explanations() -> list[dict[str, str]]:
    return [
        {"column": "rs_ratio", "meaning": "100 x relative strength divided by its moving average. Above 100 means the sector is outperforming its recent relative trend."},
        {"column": "rs_momentum", "meaning": "100 x RS-Ratio divided by its moving average. Above 100 means relative momentum is improving."},
        {"column": "quadrant", "meaning": "Leading, Weakening, Improving, or Lagging based on RS-Ratio and RS-Momentum around 100."},
        {"column": "trail_length", "meaning": "Number of latest daily or weekly points plotted as the sector's rotation tail."},
        {"column": "rotation_score", "meaning": "0-100 local score combining normalized RS-Ratio, normalized RS-Momentum, and 20D/60D relative returns. Equal-weight proxy indices receive a data-quality discount versus official index histories."},
        {"column": "direction", "meaning": "Latest trail movement direction, such as north_east for strengthening RS and momentum."},
        {"column": "price_method", "meaning": "index_ticker means Yahoo/free provider returned index history; equal_weight_proxy means the app approximated the index from configured constituents."},
    ]


def _rrg_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100 and rs_momentum < 100:
        return "Weakening"
    if rs_ratio < 100 and rs_momentum >= 100:
        return "Improving"
    return "Lagging"


def _weighted_average(values: Any, *, weights: list[float]) -> float | None:
    rows = [(_number(value, None), weights[index] if index < len(weights) else 0) for index, value in enumerate(values)]
    clean = [(value, weight) for value, weight in rows if value is not None]
    if not clean:
        return None
    weight_sum = sum(weight for _, weight in clean)
    if weight_sum <= 0:
        return _average(value for value, _ in clean)
    return round(sum(value * weight for value, weight in clean) / weight_sum, 4)


def _average(values: Any) -> float:
    clean = [_number(value, None) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return 0.0
    return round(sum(clean) / len(clean), 4)


def _percent(values: Any) -> float:
    clean = [value for value in values if value is not None]
    if not clean:
        return 0.0
    return 100 * sum(bool(value) for value in clean) / len(clean)


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"

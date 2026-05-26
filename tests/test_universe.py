from stock_advisor.data import universe


def test_nse_payload_to_universe_rows_maps_sector_and_ticker():
    payload = {
        "name": "NIFTY TOTAL MARKET",
        "data": [
            {"symbol": "NIFTY TOTAL MARKET"},
            {
                "symbol": "SUNPHARMA",
                "series": "EQ",
                "ffmc": 123,
                "lastPrice": 100,
                "yearHigh": 110,
                "yearLow": 80,
                "nearWKH": 9.1,
                "perChange30d": 4.2,
                "perChange365d": 12.3,
                "meta": {
                    "symbol": "SUNPHARMA",
                    "companyName": "Sun Pharmaceutical Industries Limited",
                    "industry": "Pharmaceuticals",
                    "isin": "INE044A01036",
                    "isDelisted": False,
                    "isSuspended": False,
                    "isETFSec": False,
                },
            },
            {
                "symbol": "BANKETF",
                "series": "EQ",
                "meta": {"symbol": "BANKETF", "industry": "ETF", "isETFSec": True},
            },
        ],
    }

    rows = universe._nse_payload_to_universe_rows(payload, index_name="NIFTY TOTAL MARKET")

    assert len(rows) == 1
    assert rows[0]["ticker"] == "SUNPHARMA.NS"
    assert rows[0]["sector"] == "Healthcare"
    assert rows[0]["basic_industry"] == "Pharmaceuticals"
    assert rows[0]["active"] is True


def test_infer_sector_from_basic_industry_handles_common_chartmaze_buckets():
    assert universe.infer_sector_from_basic_industry("Private Sector Bank") == "Financial Services"
    assert universe.infer_sector_from_basic_industry("Oil Exploration & Production") == "Oil, Gas & Consumable fuels"
    assert universe.infer_sector_from_basic_industry("Auto Components & Equipments") == "Auto"
    assert universe.infer_sector_from_basic_industry("TV Broadcasting & Software Production") == "Media Entertainment & Publication"
    assert universe.infer_sector_from_basic_industry("Plywood Boards/ Laminates") == "Forest Materials"
    assert universe.infer_sector_from_basic_industry("Iron & Steel Products") == "Metals & Mining"


def test_list_sector_constituents_groups_stocks_by_sector_and_industry(monkeypatch):
    rows = universe.pd.DataFrame(
        [
            {
                "ticker": "AAA.NS",
                "symbol": "AAA",
                "name": "AAA Ltd",
                "isin": "INEAAA",
                "sector": "Healthcare",
                "industry": "Pharmaceuticals",
                "basic_industry": "Pharmaceuticals",
                "free_float_market_cap": 100,
                "active": True,
                "last_price": 10,
                "year_high": 12,
                "year_low": 8,
                "near_52w_high_pct": 10,
                "return_30d_pct": 1,
                "return_365d_pct": 2,
                "refreshed_at": "2026-05-18T00:00:00+00:00",
            },
            {
                "ticker": "BBB.NS",
                "symbol": "BBB",
                "name": "BBB Ltd",
                "isin": "INEBBB",
                "sector": "Healthcare",
                "industry": "Hospitals",
                "basic_industry": "Hospitals",
                "free_float_market_cap": 50,
                "active": True,
                "last_price": 20,
                "year_high": 25,
                "year_low": 15,
                "near_52w_high_pct": 20,
                "return_30d_pct": 3,
                "return_365d_pct": 4,
                "refreshed_at": "2026-05-18T00:00:00+00:00",
            },
        ]
    )
    monkeypatch.setattr(universe, "load_stock_universe", lambda **kwargs: rows)

    result = universe.list_sector_constituents(sector="Healthcare")

    assert result["count"] == 2
    assert result["sector_count"] == 1
    assert result["sectors"][0]["sector"] == "Healthcare"
    assert result["sectors"][0]["stock_count"] == 2
    assert result["sectors"][0]["industries"][0]["stock_count"] == 1
    assert result["sectors"][0]["stocks"][0]["symbol"] == "AAA"


def test_nse_quote_to_universe_row_uses_public_industry_info():
    payload = {
        "info": {
            "symbol": "CENTURYPLY",
            "companyName": "Century Plyboards (India) Limited",
            "isin": "INE348B01021",
            "isDelisted": False,
            "isSuspended": False,
            "isETFSec": False,
        },
        "metadata": {"series": "EQ"},
        "industryInfo": {
            "sector": "Consumer Durables",
            "industry": "Consumer Durables",
            "basicIndustry": "Plywood Boards/ Laminates",
        },
        "priceInfo": {
            "lastPrice": 750,
            "weekHighLow": {"max": 900, "min": 500},
        },
    }

    row = universe._nse_quote_to_universe_row(payload, master_row={}, refreshed_at="2026-05-18T00:00:00+00:00")

    assert row["ticker"] == "CENTURYPLY.NS"
    assert row["sector"] == "Forest Materials"
    assert row["industry"] == "Consumer Durables"
    assert row["basic_industry"] == "Plywood Boards/ Laminates"
    assert row["near_52w_high_pct"] == 16.6667


def test_load_stock_universe_normalizes_chartmaze_like_metal_and_forest_groups(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text(
        "\n".join(
            [
                ",".join(universe.UNIVERSE_COLUMNS),
                "AEROFLEX.NS,AEROFLEX,Aeroflex Industries,INE,Capital Goods,Industrial Products,Iron & Steel Products,NSE FULL EQUITY,nse,True,EQ,,100,120,80,,,,2026-05-18T00:00:00+00:00",
                "CENTURYPLY.NS,CENTURYPLY,Century Ply,INE,Consumer Durables,Consumer Durables,Plywood Boards/ Laminates,NSE FULL EQUITY,nse,True,EQ,,100,120,80,,,,2026-05-18T00:00:00+00:00",
            ]
        ),
        encoding="utf-8",
    )

    result = universe.load_stock_universe(universe="full_nse", path=path)

    aeroflex = result[result["symbol"] == "AEROFLEX"].iloc[0]
    century = result[result["symbol"] == "CENTURYPLY"].iloc[0]
    assert aeroflex["sector"] == "Metals & Mining"
    assert aeroflex["basic_industry"] == "Metal Fabrication"
    assert century["sector"] == "Forest Materials"
    assert century["basic_industry"] == "Wood Products"


def test_full_nse_universe_returns_empty_until_refreshed_for_missing_file(tmp_path):
    result = universe.load_stock_universe(universe="full_nse", path=tmp_path / "missing.csv")

    assert result.empty


def test_sector_label_normalization_keeps_chartmaze_style_names():
    assert universe._normalize_sector_label("Fast Moving Consumer Goods") == "FMCG"
    assert universe._normalize_sector_label("Automobile and Auto Components") == "Auto"
    assert universe._normalize_sector_label("Oil Gas & Consumable Fuels") == "Oil, Gas & Consumable fuels"


def test_dhan_scrip_master_to_universe_rows_builds_nse_bse_equity_tickers():
    rows = [
        {
            "EXCH_ID": "BSE",
            "SEGMENT": "E",
            "SECURITY_ID": 500325,
            "ISIN": "INE002A01018",
            "INSTRUMENT": "EQUITY",
            "UNDERLYING_SYMBOL": "RELIANCE",
            "SYMBOL_NAME": "RELIANCE INDUSTRIES LTD",
            "DISPLAY_NAME": "Reliance Industries",
            "INSTRUMENT_TYPE": "ES",
            "SERIES": "A",
            "BUY_SELL_INDICATOR": "A",
        },
        {
            "EXCH_ID": "NSE",
            "SEGMENT": "E",
            "SECURITY_ID": 2885,
            "ISIN": "INE002A01018",
            "INSTRUMENT": "EQUITY",
            "UNDERLYING_SYMBOL": "RELIANCE",
            "SYMBOL_NAME": "RELIANCE INDUSTRIES LTD",
            "DISPLAY_NAME": "Reliance Industries",
            "INSTRUMENT_TYPE": "ES",
            "SERIES": "EQ",
            "BUY_SELL_INDICATOR": "A",
        },
        {
            "EXCH_ID": "NSE",
            "SEGMENT": "D",
            "SECURITY_ID": 999,
            "INSTRUMENT": "FUTSTK",
            "UNDERLYING_SYMBOL": "RELIANCE",
            "INSTRUMENT_TYPE": "FS",
        },
    ]

    result = universe._dhan_scrip_master_to_universe_rows(rows, exchange=None, refreshed_at="2026-05-25T00:00:00+00:00")

    assert len(result) == 2
    tickers = {row["ticker"] for row in result}
    assert tickers == {"RELIANCE.NS", "500325.BO"}
    assert {row["exchange"] for row in result} == {"NSE", "BSE"}


def test_merge_exchange_rows_dedupes_dual_listed_by_isin():
    rows = [
        {
            "ticker": "ABC.NS",
            "symbol": "ABC",
            "isin": "INEABC01010",
            "exchange": "NSE",
            "security_id": "123",
            "nse_ticker": "ABC.NS",
            "nse_security_id": "123",
        },
        {
            "ticker": "500123.BO",
            "symbol": "ABC",
            "isin": "INEABC01010",
            "exchange": "BSE",
            "security_id": "500123",
            "bse_ticker": "500123.BO",
            "bse_security_id": "500123",
        },
        {
            "ticker": "200072.BO",
            "symbol": "MCL",
            "isin": "INE813V01022",
            "exchange": "BSE",
            "security_id": "200072",
            "bse_ticker": "200072.BO",
            "bse_security_id": "200072",
        },
    ]

    result = universe._merge_exchange_rows(rows)
    by_isin = {row["isin"]: row for row in result}

    assert len(result) == 2
    assert by_isin["INEABC01010"]["ticker"] == "ABC.NS"
    assert by_isin["INEABC01010"]["exchange"] == "NSE+BSE"
    assert by_isin["INEABC01010"]["bse_ticker"] == "500123.BO"
    assert by_isin["INE813V01022"]["exchange"] == "BSE"

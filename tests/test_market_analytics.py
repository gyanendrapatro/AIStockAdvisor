import numpy as np
import pandas as pd

from stock_advisor.analysis import market_analytics


def _frame(drift: float, rows: int = 260) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    steps = np.arange(rows)
    close = 100 * ((1 + drift) ** steps) + np.sin(np.linspace(0, 8, rows)) * 1.5
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, rows),
        }
    )


def _split_frame(first_drift: float, recent_drift: float, rows: int = 260, switch: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    close = []
    value = 100.0
    for index in range(rows):
        value *= 1 + (first_drift if index < switch else recent_drift)
        close.append(value)
    close = np.array(close)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, rows),
        }
    )


def _patch_universe(monkeypatch):
    monkeypatch.setattr(
        market_analytics,
        "INDUSTRY_DEFINITIONS",
        {
            "leaders": {
                "name": "Leaders",
                "sector": "Growth",
                "stocks": ("AAA.NS", "BBB.NS", "CCC.NS"),
            },
            "laggards": {
                "name": "Laggards",
                "sector": "Cyclical",
                "stocks": ("DDD.NS", "EEE.NS", "FFF.NS"),
            },
        },
    )
    monkeypatch.setattr(
        market_analytics,
        "SECTOR_DEFINITIONS",
        {
            "growth": {
                "name": "Growth Index",
                "index_ticker": "^GROWTH",
                "sector": "Growth",
                "stocks": ("AAA.NS", "BBB.NS", "CCC.NS"),
            },
            "cyclical": {
                "name": "Cyclical Index",
                "index_ticker": "^CYCLICAL",
                "sector": "Cyclical",
                "stocks": ("DDD.NS", "EEE.NS", "FFF.NS"),
            },
        },
    )
    monkeypatch.setattr(market_analytics, "RRG_ADDITIONAL_INDEX_DEFINITIONS", {})


def test_industry_analytics_ranks_multi_window_strength(monkeypatch):
    _patch_universe(monkeypatch)
    drift = {
        "^NSEI": 0.0002,
        "AAA.NS": 0.0014,
        "BBB.NS": 0.0011,
        "CCC.NS": 0.0010,
        "DDD.NS": -0.0002,
        "EEE.NS": -0.0003,
        "FFF.NS": -0.0004,
    }
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: _frame(drift[ticker]))

    result = market_analytics.get_industry_analytics(min_stocks=2)

    assert result["top_industry"]["industry_id"] == "leaders"
    assert result["industries"][0]["composite_score"] > result["industries"][1]["composite_score"]
    assert result["industries"][0]["rank_1m"] == 1
    assert result["industries"][0]["rs_20d"] > 0


def test_rank_industry_stocks_picks_best_stock(monkeypatch):
    _patch_universe(monkeypatch)
    drift = {
        "AAA.NS": 0.0018,
        "BBB.NS": 0.0010,
        "CCC.NS": 0.0005,
    }
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: _frame(drift[ticker]))
    monkeypatch.setattr(
        market_analytics,
        "get_basic_fundamentals",
        lambda ticker: {
            "shortName": ticker,
            "forwardPE": 18,
            "debtToEquity": 20,
            "profitMargins": 0.12,
            "revenueGrowth": 0.1,
            "earningsGrowth": 0.08,
        },
    )

    result = market_analytics.rank_industry_stocks("leaders", max_stocks=3)

    assert result["industry_id"] == "leaders"
    assert result["top_stock"]["ticker"] == "AAA.NS"
    assert result["stocks"][0]["relative_strength"]["vs_sector_20d"] > 0


def test_market_indices_breadth_and_top_gainers(monkeypatch):
    _patch_universe(monkeypatch)
    drift = {
        "^NSEI": 0.0002,
        "^GROWTH": 0.0012,
        "^CYCLICAL": -0.0002,
        "AAA.NS": 0.0018,
        "BBB.NS": 0.0010,
        "CCC.NS": 0.0005,
        "DDD.NS": -0.0002,
        "EEE.NS": -0.0003,
        "FFF.NS": -0.0004,
    }
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: _frame(drift[ticker]))
    monkeypatch.setattr(
        market_analytics,
        "_get_index_price_history",
        lambda definition, period, interval: _frame(drift[definition["index_ticker"]]),
    )

    indices = market_analytics.get_market_indices()
    breadth = market_analytics.get_market_breadth()
    gainers = market_analytics.get_top_gainers(return_window="20d", max_rows=3)

    assert any(row["id"] == "growth" for row in indices["indices"])
    assert breadth["summary"]["stock_count"] == 6
    assert breadth["summary"]["above_50_pct"] > 0
    assert gainers["stocks"][0]["ticker"] == "AAA.NS"
    assert gainers["industry_summary"][0]["industry"] == "Leaders"


def test_sector_analytics_supports_chartmaze_style_modes(monkeypatch):
    _patch_universe(monkeypatch)
    drift = {
        "^NSEI": 0.0002,
        "AAA.NS": 0.0018,
        "BBB.NS": 0.0010,
        "CCC.NS": 0.0005,
        "DDD.NS": -0.0002,
        "EEE.NS": -0.0003,
        "FFF.NS": -0.0004,
    }
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: _frame(drift[ticker]))

    ma = market_analytics.get_sector_analytics(mode="moving_average", ma_period=200, selected_sector="growth")
    rs = market_analytics.get_sector_analytics(mode="relative_strength", rs_cutoff=55)
    near_high = market_analytics.get_sector_analytics(mode="near_52w_high", near_high_pct=5)

    assert ma["top_sector"]["sector_id"] == "growth"
    assert ma["selected_sector"]["sector_id"] == "growth"
    assert ma["industries"][0]["industry"] == "Leaders"
    assert ma["stocks"][0]["passes_filter"] is True
    assert rs["top_sector"]["sector_id"] == "growth"
    assert rs["metric_label"] == "% of stocks trading above RS 55"
    assert near_high["top_sector"]["sector_id"] == "growth"
    assert "52w high" in near_high["metric_label"]


def test_sector_analytics_supports_21_ema_filter(monkeypatch):
    _patch_universe(monkeypatch)
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: _frame(0.001))

    result = market_analytics.get_sector_analytics(mode="moving_average", ma_type="21 EMA", selected_sector="growth")

    assert result["metric_label"] == "% of stocks trading above 21 EMA"
    assert result["filter"]["ma_period"] == 21
    assert result["filter"]["ma_type"] == "21 EMA"
    assert result["stocks"][0]["criterion"] == "Above 21 EMA"
    assert result["selected_sector"]["above_21_ema_pct"] == 100


def test_sector_analytics_industry_contribution_breakdown(monkeypatch):
    monkeypatch.setattr(
        market_analytics,
        "SECTOR_DEFINITIONS",
        {
            "auto": {
                "name": "Nifty Auto",
                "index_ticker": "^CNXAUTO",
                "sector": "Auto",
                "stocks": ("MARUTI.NS", "MOTHERSON.NS", "BOSCHLTD.NS"),
            },
        },
    )
    monkeypatch.setattr(
        market_analytics,
        "get_price_history",
        lambda ticker, period, interval: _frame(0.001),
    )

    result = market_analytics.get_sector_analytics(mode="moving_average", ma_period=200, selected_sector="auto")
    industries = {row["industry"]: row for row in result["industries"]}

    assert industries["Auto Components"]["formula"] == "2 / 3"
    assert industries["Auto Components"]["contribution_pct"] == 66.67
    assert industries["Auto Components"]["passing_tickers"] == ["MOTHERSON.NS", "BOSCHLTD.NS"]
    assert industries["Passenger Vehicles"]["formula"] == "1 / 3"


def test_broad_sector_analytics_uses_stock_breadth_universe(monkeypatch):
    universe = pd.DataFrame(
        [
            {
                "ticker": "AAA.NS",
                "symbol": "AAA",
                "name": "AAA Pharma",
                "sector": "Healthcare",
                "basic_industry": "Pharmaceuticals",
                "free_float_market_cap": 1000,
                "last_price": 100,
                "year_high": 102,
                "year_low": 70,
                "near_52w_high_pct": 1.5,
                "return_30d_pct": 12,
                "return_365d_pct": 20,
                "refreshed_at": "2026-05-18T21:23:50+00:00",
                "active": True,
            },
            {
                "ticker": "BBB.NS",
                "symbol": "BBB",
                "name": "BBB Hospitals",
                "sector": "Healthcare",
                "basic_industry": "Hospitals",
                "free_float_market_cap": 500,
                "last_price": 90,
                "year_high": 100,
                "year_low": 60,
                "near_52w_high_pct": 10,
                "return_30d_pct": 5,
                "return_365d_pct": 12,
                "refreshed_at": "2026-05-18T21:23:50+00:00",
                "active": True,
            },
            {
                "ticker": "CCC.NS",
                "symbol": "CCC",
                "name": "CCC Auto",
                "sector": "Auto",
                "basic_industry": "Auto Components & Equipments",
                "free_float_market_cap": 700,
                "last_price": 120,
                "year_high": 121,
                "year_low": 80,
                "near_52w_high_pct": 0.8,
                "return_30d_pct": 3,
                "return_365d_pct": 7,
                "refreshed_at": "2026-05-18T21:23:50+00:00",
                "active": True,
            },
            {
                "ticker": "DDD.NS",
                "symbol": "DDD",
                "name": "DDD Two Wheelers",
                "sector": "Auto",
                "basic_industry": "2/3 Wheelers",
                "free_float_market_cap": 300,
                "last_price": 85,
                "year_high": 100,
                "year_low": 50,
                "near_52w_high_pct": 15,
                "return_30d_pct": -2,
                "return_365d_pct": -5,
                "refreshed_at": "2026-05-18T21:23:50+00:00",
                "active": True,
            },
        ]
    )
    monkeypatch.setattr(market_analytics, "load_stock_universe", lambda **kwargs: universe)
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: _frame(0.0002))
    monkeypatch.setattr(
        market_analytics,
        "get_price_histories",
        lambda tickers, period, interval: {
            ticker: (_split_frame(0.002, -0.004) if ticker in {"BBB.NS", "DDD.NS"} else _frame(0.001))
            for ticker in tickers
        },
    )

    result = market_analytics.get_sector_analytics(
        mode="near_52w_high",
        near_high_pct=5,
        selected_sector="healthcare",
        universe="broad",
    )

    assert result["universe"] == "broad"
    assert result["selected_sector"]["name"] == "Healthcare"
    assert result["selected_sector"]["metric_pct"] == 50
    assert result["industries"][0]["industry"] == "Pharmaceuticals"
    assert result["industries"][0]["formula"] == "1 / 2"
    assert result["stocks"][0]["ticker"] == "AAA.NS"
    assert result["stocks"][0]["symbol"] == "AAA"
    assert result["stocks"][0]["basic_industry"] == "Pharmaceuticals"
    assert result["stocks"][0]["market_cap"] == 1000
    assert result["stocks"][0]["sma_200"] is not None
    assert result["stocks"][0]["ema_21"] is not None
    assert result["stocks"][0]["rs_rating"] is not None
    assert result["stocks"][0]["distance_from_52w_high_pct"] > -5
    assert len(result["constituent_stocks"]) == 2
    assert {"symbol", "sector", "basic_industry", "market_cap", "close", "sma_200", "ema_21", "rs_rating", "distance_from_52w_high_pct"}.issubset(
        result["constituent_stocks"][0]
    )
    assert result["universe_refreshed_at"] is not None
    assert result["price_history_end_date"] is not None


def test_broad_sector_analytics_does_not_promote_tiny_sectors(monkeypatch):
    rows = [
        {
            "ticker": "JKPAPER.NS",
            "symbol": "JKPAPER",
            "name": "JK Paper",
            "sector": "Forest Materials",
            "basic_industry": "Paper & Paper Products",
            "free_float_market_cap": 100,
            "last_price": 100,
            "year_high": 101,
            "year_low": 50,
            "near_52w_high_pct": 1,
            "return_30d_pct": 10,
            "return_365d_pct": 20,
            "refreshed_at": "2026-05-18T21:23:50+00:00",
        }
    ]
    for index in range(5):
        rows.append(
            {
                "ticker": f"METAL{index}.NS",
                "symbol": f"METAL{index}",
                "name": f"Metal {index}",
                "sector": "Metals & Mining",
                "basic_industry": "Iron & Steel",
                "free_float_market_cap": 1000,
                "last_price": 100,
                "year_high": 101,
                "year_low": 50,
                "near_52w_high_pct": 1,
                "return_30d_pct": 10,
                "return_365d_pct": 20,
                "refreshed_at": "2026-05-18T21:23:50+00:00",
            }
        )
    universe = pd.DataFrame(rows)
    monkeypatch.setattr(market_analytics, "load_stock_universe", lambda **kwargs: universe)
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: _frame(0.001))
    monkeypatch.setattr(market_analytics, "get_price_histories", lambda tickers, period, interval: {ticker: _frame(0.001) for ticker in tickers})

    result = market_analytics.get_sector_analytics(mode="moving_average", ma_type="200 MA", universe="broad")

    forest = next(row for row in result["sectors"] if row["name"] == "Forest Materials")
    assert forest["metric_pct"] == 100
    assert forest["ranking_eligible"] is False
    assert result["top_sector"]["name"] == "Metals & Mining"
    assert result["top_sector"]["ranking_eligible"] is True


def test_broad_industry_analytics_uses_batched_price_history(monkeypatch):
    universe = pd.DataFrame(
        [
            {"ticker": "AAA.NS", "sector": "Healthcare", "basic_industry": "Pharmaceuticals", "free_float_market_cap": 100, "active": True},
            {"ticker": "BBB.NS", "sector": "Healthcare", "basic_industry": "Pharmaceuticals", "free_float_market_cap": 100, "active": True},
            {"ticker": "CCC.NS", "sector": "Auto", "basic_industry": "Auto Components", "free_float_market_cap": 100, "active": True},
            {"ticker": "DDD.NS", "sector": "Auto", "basic_industry": "Auto Components", "free_float_market_cap": 100, "active": True},
        ]
    )
    monkeypatch.setattr(market_analytics, "load_stock_universe", lambda **kwargs: universe)
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: _frame(0.0002))
    monkeypatch.setattr(
        market_analytics,
        "get_price_histories",
        lambda tickers, period, interval: {
            "AAA.NS": _frame(0.0015),
            "BBB.NS": _frame(0.0012),
            "CCC.NS": _frame(-0.0002),
            "DDD.NS": _frame(-0.0004),
        },
    )

    result = market_analytics.get_industry_analytics(universe="broad", min_stocks=2)

    assert result["universe"] == "broad"
    assert result["top_industry"]["name"] == "Pharmaceuticals"
    assert result["top_industry"]["stock_count"] == 2
    assert result["top_industry"]["configured_stock_count"] == 2


def test_relative_rotation_graph_quadrants(monkeypatch):
    _patch_universe(monkeypatch)
    frames = {
        "^NSEI": _split_frame(0.0002, 0.0002),
        "^GROWTH": _split_frame(0.0005, 0.0040, switch=240),
        "^CYCLICAL": _split_frame(-0.0002, -0.0040, switch=240),
    }
    monkeypatch.setattr(market_analytics, "get_price_history", lambda ticker, period, interval: frames[ticker])
    monkeypatch.setattr(
        market_analytics,
        "_get_index_price_history",
        lambda definition, period, interval: frames[definition["index_ticker"]],
    )

    result = market_analytics.get_relative_rotation_graph(trail_length=5)
    points = {row["sector_id"]: row for row in result["points"]}

    assert points["growth"]["quadrant"] == "Leading"
    assert points["cyclical"]["quadrant"] == "Lagging"
    assert result["quadrant_counts"]["Leading"] == 1
    assert result["quadrant_counts"]["Lagging"] == 1
    assert len(result["trails"][0]["points"]) == 5
    assert "rs_ratio" in result["trails"][0]["points"][-1]


def test_rrg_additional_index_definitions_do_not_use_unrelated_etfs():
    definitions = market_analytics.RRG_ADDITIONAL_INDEX_DEFINITIONS

    assert definitions["chemicals"]["index_ticker"] == "CHEMICAL.NS"
    assert "CHEMCON.NS" not in definitions["chemicals"].get("index_tickers", ())
    assert definitions["capital_markets"]["index_ticker"] == "MOCAPITAL.NS"
    assert definitions["housing"]["preferred_price_method"] == "equal_weight_proxy"
    assert definitions["midsmall_400"]["preferred_price_method"] == "equal_weight_proxy"
    assert definitions["housing"]["index_ticker"] != "HNGSNGBEES.NS"
    assert definitions["midsmall_400"]["index_ticker"] != "MOM100.NS"


def test_market_indices_universe_matches_chartsmaze_grid_count():
    names = {"Nifty 50"} | {definition["name"] for definition in market_analytics._rrg_index_definitions().values()}

    assert len(names) == 36
    assert {
        "Nifty Pharma",
        "Nifty Metals & Mining",
        "Nifty Capital Markets",
        "Nifty Midsmallcap 400",
        "Nifty Tourism",
    }.issubset(names)

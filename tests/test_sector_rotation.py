import numpy as np
import pandas as pd

from stock_advisor.analysis import sector_rotation


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


def _patch_sector_universe(monkeypatch):
    monkeypatch.setattr(
        sector_rotation,
        "SECTOR_DEFINITIONS",
        {
            "strong": {
                "name": "Strong Sector",
                "index_ticker": "^STRONG",
                "sector": "Strong",
                "stocks": ("AAA.NS", "BBB.NS"),
            },
            "weak": {
                "name": "Weak Sector",
                "index_ticker": "^WEAK",
                "sector": "Weak",
                "stocks": ("CCC.NS", "DDD.NS"),
            },
        },
    )


def test_get_sector_rotation_ranks_relative_strength(monkeypatch):
    _patch_sector_universe(monkeypatch)
    drift = {
        "^NSEI": 0.0002,
        "^STRONG": 0.0013,
        "^WEAK": -0.0004,
        "AAA.NS": 0.0015,
        "BBB.NS": 0.0009,
        "CCC.NS": -0.0002,
        "DDD.NS": -0.0006,
    }
    monkeypatch.setattr(
        sector_rotation,
        "get_price_history",
        lambda ticker, period, interval: _frame(drift[ticker]),
    )

    result = sector_rotation.get_sector_rotation(max_sectors=2, max_breadth_stocks=2)

    assert result["top_sector"]["sector_id"] == "strong"
    assert result["sectors"][0]["rotation_score"] > result["sectors"][1]["rotation_score"]
    assert result["sectors"][0]["relative_strength"]["vs_benchmark_20d"] > 0


def test_rank_sector_stocks_picks_best_stock_inside_sector(monkeypatch):
    _patch_sector_universe(monkeypatch)
    drift = {
        "^STRONG": 0.0009,
        "AAA.NS": 0.0018,
        "BBB.NS": 0.0003,
    }
    monkeypatch.setattr(
        sector_rotation,
        "get_price_history",
        lambda ticker, period, interval: _frame(drift[ticker]),
    )
    monkeypatch.setattr(
        sector_rotation,
        "get_basic_fundamentals",
        lambda ticker: {
            "shortName": ticker,
            "forwardPE": 18,
            "debtToEquity": 30,
            "profitMargins": 0.12,
            "revenueGrowth": 0.1,
            "earningsGrowth": 0.08,
            "_sources": ["unit-test"],
        },
    )

    result = sector_rotation.rank_sector_stocks("strong", max_stocks=2)

    assert result["sector_id"] == "strong"
    assert result["top_stock"]["ticker"] == "AAA.NS"
    assert result["stocks"][0]["relative_strength"]["vs_sector_20d"] > 0


def test_discover_sector_opportunities_returns_top_sector_candidates(monkeypatch):
    _patch_sector_universe(monkeypatch)
    drift = {
        "^NSEI": 0.0002,
        "^STRONG": 0.0013,
        "^WEAK": -0.0004,
        "AAA.NS": 0.0015,
        "BBB.NS": 0.0009,
        "CCC.NS": -0.0002,
        "DDD.NS": -0.0006,
    }
    monkeypatch.setattr(
        sector_rotation,
        "get_price_history",
        lambda ticker, period, interval: _frame(drift[ticker]),
    )
    monkeypatch.setattr(sector_rotation, "get_basic_fundamentals", lambda ticker: {})

    result = sector_rotation.discover_sector_opportunities(top_sectors=1, stocks_per_sector=2)

    assert result["top_sectors"][0]["sector_id"] == "strong"
    assert result["opportunities"][0]["top_stock"]["ticker"] == "AAA.NS"

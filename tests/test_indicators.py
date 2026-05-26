import json

import numpy as np
import pandas as pd

from stock_advisor.analysis.indicators import add_indicators, latest_indicators


def _market_frame(rows=260):
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    base = np.linspace(100, 180, rows)
    wave = np.sin(np.linspace(0, 12, rows)) * 4
    close = base + wave
    open_ = close * 0.995
    high = close * 1.015
    low = close * 0.985
    volume = np.linspace(1_000_000, 2_000_000, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_add_indicators_covers_production_indicator_suite():
    result = add_indicators(_market_frame())

    expected_columns = {
        "sma_100",
        "sma_200",
        "ema_100",
        "ema_21",
        "ema_200",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "sma_50_200_cross",
        "stoch_k_14",
        "stoch_d_3",
        "roc_12",
        "cci_20",
        "williams_r_14",
        "bb_upper_20",
        "bb_lower_20",
        "bb_percent_b_20",
        "keltner_upper_20",
        "donchian_high_20",
        "obv",
        "mfi_14",
        "vwap",
        "adl",
        "cmf_20",
        "recent_swing_high_20",
        "pivot_point",
        "high_52w",
        "distance_from_52w_high",
        "max_drawdown",
        "downside_volatility_20d",
        "gap_risk_20d",
        "liquidity_score",
    }

    assert expected_columns.issubset(result.columns)
    last = result.iloc[-1]
    assert pd.notna(last["sma_200"])
    assert pd.notna(last["adx_14"])
    assert pd.notna(last["bb_percent_b_20"])
    assert pd.notna(last["vwap"])
    assert pd.notna(last["liquidity_score"])


def test_latest_indicators_is_json_safe():
    result = latest_indicators(_market_frame())

    assert result["sma_200"] is not None
    assert result["high_52w"] is not None
    json.dumps(result, allow_nan=False)

import numpy as np
import pandas as pd

from stock_advisor.analysis.chart_patterns import chart_pattern_indicator_fields, detect_chart_patterns


def _frame(close_values):
    close = np.array(close_values, dtype=float)
    dates = pd.date_range("2025-01-01", periods=len(close), freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 1_600_000, len(close)),
        }
    )


def test_detect_chart_patterns_identifies_uptrend_structure():
    base = np.linspace(80, 130, 120)
    wave = np.sin(np.linspace(0, 10 * np.pi, 120)) * 3
    result = detect_chart_patterns(_frame(base + wave))

    names = {pattern["pattern"] for pattern in result["patterns"]}

    assert {"higher_high_higher_low", "ascending_channel"} & names
    assert result["pattern_score"] > 50
    assert result["chart_pattern_direction"] == "bullish"


def test_detect_chart_patterns_identifies_double_bottom():
    close = [
        112,
        109,
        104,
        98,
        91,
        84,
        88,
        95,
        103,
        111,
        104,
        96,
        87,
        85,
        89,
        97,
        106,
        114,
        118,
        121,
    ]
    close = list(np.linspace(130, 116, 35)) + close + list(np.linspace(119, 126, 20))
    result = detect_chart_patterns(_frame(close), lookback=100)

    assert any(pattern["pattern"] == "double_bottom" for pattern in result["patterns"])
    assert result["pattern_score"] > 50


def test_detect_chart_patterns_identifies_rectangle_breakout():
    base = 105 + np.sin(np.linspace(0, 9 * np.pi, 70)) * 4
    close = list(base) + [111, 114, 116]
    result = detect_chart_patterns(_frame(close), lookback=100)

    assert any(pattern["pattern"] == "rectangle_breakout" for pattern in result["patterns"])
    assert result["chart_pattern_direction"] == "bullish"


def test_chart_pattern_indicator_fields_are_compact_for_scoring():
    result = {
        "pattern_score": 72,
        "chart_pattern_direction": "bullish",
        "dominant_pattern": {"pattern": "double_bottom", "confidence": 76},
    }

    fields = chart_pattern_indicator_fields(result)

    assert fields == {
        "chart_pattern_score": 72,
        "chart_pattern_direction": "bullish",
        "dominant_chart_pattern": "double_bottom",
        "chart_pattern_confidence": 76,
    }

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    close = out["close"].astype(float)
    high = out.get("high", close).astype(float)
    low = out.get("low", close).astype(float)
    open_ = out.get("open", close).astype(float)
    volume = out.get("volume", pd.Series(index=out.index, data=np.nan)).astype(float)

    returns = close.pct_change()
    prev_close = close.shift(1)
    typical_price = (high + low + close) / 3
    tr = _true_range(high, low, close)

    _add_trend_indicators(out, close, high, low, tr)
    _add_momentum_indicators(out, close, high, low, typical_price)
    _add_volatility_indicators(out, close, high, low, tr)
    _add_volume_indicators(out, close, high, low, volume, typical_price)
    _add_support_resistance(out, close, high, low)
    _add_risk_quality(out, close, open_, volume, returns, prev_close)
    return out


def latest_indicators(df: pd.DataFrame) -> dict[str, Any]:
    ind = add_indicators(df)
    if ind.empty:
        return {}
    row = ind.dropna(how="all").iloc[-1].to_dict()
    return {k: _safe_value(v) for k, v in row.items() if k not in {"date"}}


def _add_trend_indicators(out: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series, tr: pd.Series) -> None:
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_100"] = close.rolling(100).mean()
    out["sma_200"] = close.rolling(200).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["ema_100"] = close.ewm(span=100, adjust=False).mean()
    out["ema_200"] = close.ewm(span=200, adjust=False).mean()

    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_histogram"] = out["macd"] - out["macd_signal"]

    plus_di, minus_di, adx = _adx(high, low, tr)
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di
    out["adx_14"] = adx

    out["sma_20_50_cross"] = _cross_signal(out["sma_20"], out["sma_50"])
    out["sma_50_200_cross"] = _cross_signal(out["sma_50"], out["sma_200"])
    out["ema_12_26_cross"] = _cross_signal(out["ema_12"], out["ema_26"])
    out["trend_alignment_score"] = _trend_alignment(close, out["sma_20"], out["sma_50"], out["sma_100"], out["sma_200"])


def _add_momentum_indicators(
    out: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    typical_price: pd.Series,
) -> None:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    range_14 = (high_14 - low_14).replace(0, np.nan)
    out["stoch_k_14"] = 100 * (close - low_14) / range_14
    out["stoch_d_3"] = out["stoch_k_14"].rolling(3).mean()
    out["williams_r_14"] = -100 * (high_14 - close) / range_14
    out["roc_12"] = close.pct_change(12) * 100

    mean_dev = (typical_price - typical_price.rolling(20).mean()).abs().rolling(20).mean()
    out["cci_20"] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * mean_dev.replace(0, np.nan))


def _add_volatility_indicators(
    out: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    tr: pd.Series,
) -> None:
    out["atr_14"] = tr.rolling(14).mean()
    out["atr_percent_14"] = out["atr_14"] / close

    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_middle_20"] = bb_middle
    out["bb_upper_20"] = bb_middle + 2 * bb_std
    out["bb_lower_20"] = bb_middle - 2 * bb_std
    out["bb_width_20"] = (out["bb_upper_20"] - out["bb_lower_20"]) / bb_middle.replace(0, np.nan)
    out["bb_percent_b_20"] = (close - out["bb_lower_20"]) / (out["bb_upper_20"] - out["bb_lower_20"]).replace(0, np.nan)

    keltner_middle = close.ewm(span=20, adjust=False).mean()
    out["keltner_middle_20"] = keltner_middle
    out["keltner_upper_20"] = keltner_middle + 2 * out["atr_14"]
    out["keltner_lower_20"] = keltner_middle - 2 * out["atr_14"]
    out["keltner_width_20"] = (out["keltner_upper_20"] - out["keltner_lower_20"]) / keltner_middle.replace(0, np.nan)

    out["donchian_high_20"] = high.rolling(20).max()
    out["donchian_low_20"] = low.rolling(20).min()
    out["donchian_mid_20"] = (out["donchian_high_20"] + out["donchian_low_20"]) / 2
    out["donchian_width_20"] = (out["donchian_high_20"] - out["donchian_low_20"]) / close.replace(0, np.nan)

    out["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS)


def _add_volume_indicators(
    out: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    typical_price: pd.Series,
) -> None:
    out["volume_sma_20"] = volume.rolling(20).mean()
    out["volume_ratio"] = volume / out["volume_sma_20"].replace(0, np.nan)
    out["dollar_volume"] = close * volume
    out["avg_dollar_volume_20"] = out["dollar_volume"].rolling(20).mean()

    direction = np.sign(close.diff()).fillna(0)
    out["obv"] = (direction * volume.fillna(0)).cumsum()
    out["mfi_14"] = _money_flow_index(high, low, close, volume)

    cumulative_volume = volume.fillna(0).cumsum().replace(0, np.nan)
    out["vwap"] = (typical_price * volume.fillna(0)).cumsum() / cumulative_volume

    money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_volume = money_flow_multiplier.fillna(0) * volume.fillna(0)
    out["adl"] = money_flow_volume.cumsum()
    out["cmf_20"] = money_flow_volume.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)


def _add_support_resistance(out: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series) -> None:
    out["recent_swing_high_20"] = high.rolling(20).max()
    out["recent_swing_low_20"] = low.rolling(20).min()
    out["distance_to_swing_high_20"] = close / out["recent_swing_high_20"].replace(0, np.nan) - 1
    out["distance_from_swing_low_20"] = close / out["recent_swing_low_20"].replace(0, np.nan) - 1

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3
    out["pivot_point"] = pivot
    out["resistance_1"] = 2 * pivot - prev_low
    out["support_1"] = 2 * pivot - prev_high
    out["resistance_2"] = pivot + (prev_high - prev_low)
    out["support_2"] = pivot - (prev_high - prev_low)

    high_52w = high.rolling(TRADING_DAYS, min_periods=60).max()
    low_52w = low.rolling(TRADING_DAYS, min_periods=60).min()
    out["high_52w"] = high_52w
    out["low_52w"] = low_52w
    out["distance_from_52w_high"] = close / high_52w.replace(0, np.nan) - 1
    out["distance_from_52w_low"] = close / low_52w.replace(0, np.nan) - 1


def _add_risk_quality(
    out: pd.DataFrame,
    close: pd.Series,
    open_: pd.Series,
    volume: pd.Series,
    returns: pd.Series,
    prev_close: pd.Series,
) -> None:
    out["return_20d"] = close.pct_change(20)
    drawdown = close / close.cummax().replace(0, np.nan) - 1
    out["drawdown"] = drawdown
    out["max_drawdown"] = drawdown.expanding().min()
    out["downside_volatility_20d"] = returns.where(returns < 0, 0).rolling(20).std() * np.sqrt(TRADING_DAYS)
    out["gap_pct"] = open_ / prev_close.replace(0, np.nan) - 1
    out["gap_risk_20d"] = out["gap_pct"].abs().rolling(20).max()
    out["zero_volume_days_20"] = volume.eq(0).rolling(20).sum()
    out["liquidity_score"] = _liquidity_score(out["avg_dollar_volume_20"])


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, tr: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return plus_di, minus_di, adx


def _cross_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    current = np.sign(fast - slow)
    previous = np.sign(fast.shift(1) - slow.shift(1))
    return pd.Series(np.where((current > 0) & (previous <= 0), 1, np.where((current < 0) & (previous >= 0), -1, 0)), index=fast.index)


def _trend_alignment(close: pd.Series, *averages: pd.Series) -> pd.Series:
    valid_counts = sum(avg.notna().astype(int) for avg in averages)
    aligned_counts = sum((close > avg).astype(int).where(avg.notna(), 0) for avg in averages)
    return 100 * aligned_counts / valid_counts.replace(0, np.nan)


def _money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    direction = typical_price.diff()
    positive_flow = raw_money_flow.where(direction > 0, 0)
    negative_flow = raw_money_flow.where(direction < 0, 0)
    money_ratio = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + money_ratio))


def _liquidity_score(avg_dollar_volume: pd.Series) -> pd.Series:
    score = 20 * np.log10(avg_dollar_volume.replace(0, np.nan))
    return pd.Series(score, index=avg_dollar_volume.index).clip(lower=0, upper=100)


def _safe_value(v: Any) -> Any:
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, bool):
        return v
    try:
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
    except Exception:
        pass
    return v

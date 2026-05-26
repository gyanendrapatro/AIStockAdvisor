from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class StockScore:
    ticker: str
    final_score: float
    signal: str
    confidence: float
    technical_score: float
    fundamental_score: float
    news_score: float
    risk_score: float
    momentum_liquidity_score: float
    event_intelligence_score: float | None
    reasons: list[str]
    risks: list[str]


def score_stock(
    ticker: str,
    indicators: dict,
    fundamentals: dict,
    news: list[dict],
    intelligence: dict | None = None,
) -> StockScore:
    technical, tech_reasons = _technical_score(indicators)
    fundamental, fund_reasons = _fundamental_score(fundamentals)
    news_score, news_reasons = _news_score(news)
    risk, risk_reasons = _risk_score(indicators, fundamentals)
    momentum_liquidity, momentum_reasons = _momentum_liquidity_score(indicators)
    intelligence_score, intelligence_reasons, intelligence_risks = _intelligence_score(intelligence)

    if intelligence is None:
        final = round(
            0.35 * technical
            + 0.20 * fundamental
            + 0.20 * news_score
            + 0.15 * risk
            + 0.10 * momentum_liquidity,
            2,
        )
        exported_intelligence_score = None
    else:
        final = round(
            0.30 * technical
            + 0.20 * fundamental
            + 0.15 * news_score
            + 0.15 * risk
            + 0.10 * momentum_liquidity
            + 0.10 * intelligence_score,
            2,
        )
        exported_intelligence_score = intelligence_score
    signal = _signal(final)
    confidence = round(min(95, max(35, abs(final - 50) * 1.3 + 45)), 1)
    if intelligence is None:
        reasons = tech_reasons + fund_reasons + news_reasons + momentum_reasons
    else:
        reasons = (
            tech_reasons[:2]
            + fund_reasons[:2]
            + intelligence_reasons[:3]
            + news_reasons[:1]
            + momentum_reasons[:1]
        )
    return StockScore(
        ticker,
        final,
        signal,
        confidence,
        technical,
        fundamental,
        news_score,
        risk,
        momentum_liquidity,
        exported_intelligence_score,
        reasons[:8],
        (risk_reasons + intelligence_risks)[:5],
    )


def _technical_score(i: dict) -> tuple[float, list[str]]:
    if not i:
        return 50, ["No technical data available"]
    score = 50
    reasons = []
    close = i.get("close")
    sma20 = i.get("sma_20")
    sma50 = i.get("sma_50")
    sma100 = i.get("sma_100")
    sma200 = i.get("sma_200")
    rsi = i.get("rsi_14")
    macd = i.get("macd")
    macd_sig = i.get("macd_signal")
    adx = i.get("adx_14")
    plus_di = i.get("plus_di_14")
    minus_di = i.get("minus_di_14")
    trend_alignment = i.get("trend_alignment_score")
    bb_percent_b = i.get("bb_percent_b_20")
    donchian_high = i.get("donchian_high_20")
    sma_cross = i.get("sma_50_200_cross")
    chart_pattern_score = i.get("chart_pattern_score")
    chart_pattern_direction = i.get("chart_pattern_direction")
    dominant_chart_pattern = i.get("dominant_chart_pattern")

    if close and sma20 and close > sma20:
        score += 10; reasons.append("Price is above 20-day average")
    if close and sma50 and close > sma50:
        score += 10; reasons.append("Price is above 50-day average")
    if close and sma100 and close > sma100:
        score += 5; reasons.append("Price is above 100-day average")
    if close and sma200 and close > sma200:
        score += 7; reasons.append("Price is above 200-day average")
    if sma20 and sma50 and sma20 > sma50:
        score += 8; reasons.append("Short-term trend is above medium-term trend")
    if sma50 and sma200 and sma50 > sma200:
        score += 6; reasons.append("50-day average is above 200-day average")
    if sma_cross == 1:
        score += 8; reasons.append("Fresh bullish 50/200-day crossover")
    elif sma_cross == -1:
        score -= 8; reasons.append("Fresh bearish 50/200-day crossover")
    if trend_alignment is not None:
        if trend_alignment >= 75:
            score += 6; reasons.append("Trend alignment is broadly bullish")
        elif trend_alignment <= 25:
            score -= 6; reasons.append("Trend alignment is weak")
    if rsi is not None:
        if 45 <= rsi <= 65:
            score += 7; reasons.append("RSI is in a healthy range")
        elif rsi < 30:
            score += 4; reasons.append("RSI is oversold; possible rebound setup")
        elif rsi > 75:
            score -= 10; reasons.append("RSI is overbought")
    if macd is not None and macd_sig is not None and macd > macd_sig:
        score += 8; reasons.append("MACD is above signal line")
    if adx is not None and adx >= 25:
        if plus_di is not None and minus_di is not None and plus_di > minus_di:
            score += 8; reasons.append("ADX confirms a strong bullish trend")
        elif plus_di is not None and minus_di is not None and minus_di > plus_di:
            score -= 8; reasons.append("ADX confirms a strong bearish trend")
    if bb_percent_b is not None:
        if 0.2 <= bb_percent_b <= 0.8:
            score += 3; reasons.append("Price sits within normal Bollinger range")
        elif bb_percent_b > 1.05:
            score -= 5; reasons.append("Price is extended above Bollinger band")
    if close and donchian_high and close >= donchian_high * 0.995:
        score += 5; reasons.append("Price is near a 20-day breakout level")
    if chart_pattern_score is not None:
        try:
            pattern_score = float(chart_pattern_score)
        except (TypeError, ValueError):
            pattern_score = 50
        pattern_name = str(dominant_chart_pattern or "detected pattern").replace("_", " ")
        if pattern_score >= 60:
            score += min(8, (pattern_score - 50) * 0.25)
            reasons.append(f"Chart pattern bias is bullish: {pattern_name}")
        elif pattern_score <= 40:
            score -= min(8, (50 - pattern_score) * 0.25)
            reasons.append(f"Chart pattern bias is bearish: {pattern_name}")
        elif chart_pattern_direction == "neutral" and dominant_chart_pattern:
            reasons.append(f"Chart pattern is neutral: {pattern_name}")
    return max(0, min(100, round(score, 2))), reasons or ["Technical setup is neutral"]


def _fundamental_score(f: dict) -> tuple[float, list[str]]:
    if not f:
        return 50, ["No fundamental data available"]
    score = 50
    reasons = []
    pe = f.get("trailingPE") or f.get("forwardPE")
    debt = f.get("debtToEquity")
    margins = f.get("profitMargins")
    revenue_growth = f.get("revenueGrowth")
    roe = f.get("returnOnEquity")
    sec_revenue = f.get("sec_revenue")
    sec_net_income = f.get("sec_net_income")
    sec_liabilities = f.get("sec_liabilities")
    sec_equity = f.get("sec_equity")

    if margins is None and sec_revenue and sec_net_income:
        margins = sec_net_income / sec_revenue
    if debt is None and sec_equity and sec_liabilities:
        debt = (sec_liabilities / sec_equity) * 100
    if roe is None and sec_equity and sec_net_income:
        roe = sec_net_income / sec_equity

    if pe and 0 < pe < 35:
        score += 10; reasons.append("Valuation is not extremely stretched by PE")
    elif pe and pe > 60:
        score -= 10; reasons.append("PE valuation appears stretched")
    if debt is not None and debt < 100:
        score += 8; reasons.append("Debt-to-equity appears manageable")
    elif debt and debt > 200:
        score -= 8; reasons.append("Debt-to-equity is high")
    if margins and margins > 0.08:
        score += 10; reasons.append("Profit margin is positive and meaningful")
    if revenue_growth and revenue_growth > 0:
        score += 10; reasons.append("Revenue growth is positive")
    if roe and roe > 0.10:
        score += 8; reasons.append("Return on equity is healthy")
    if f.get("sec_revenue") is not None:
        reasons.append("SEC EDGAR filing facts are available")
    ownership_score, ownership_reasons = _ownership_governance_score(f)
    score = 0.85 * score + 0.15 * ownership_score
    reasons.extend(ownership_reasons)
    return max(0, min(100, round(score, 2))), reasons or ["Fundamentals look neutral"]


def _news_score(news: list[dict]) -> tuple[float, list[str]]:
    if not news:
        return 50, ["No recent news data configured"]
    vals = []
    titles = []
    stale_titles = []
    for item in news:
        weight = _news_recency_weight(item)
        s = item.get("overall_sentiment_score")
        if weight > 0:
            try:
                vals.append((float(s), weight))
            except Exception:
                pass
            if item.get("title"):
                titles.append(item["title"])
        elif item.get("title"):
            stale_titles.append(item["title"])
    if not vals:
        reason = "No recent news found in the last 30 days"
        if stale_titles:
            return 50, [reason, "Stale article ignored: " + stale_titles[0]]
        return 50, ["News available but no sentiment score"]
    total_weight = sum(weight for _, weight in vals)
    avg = sum(score * weight for score, weight in vals) / total_weight if total_weight else 0
    score = 50 + max(-30, min(30, avg * 100))
    return round(score, 2), ["Recent news sentiment: " + ("positive" if avg > 0.05 else "negative" if avg < -0.05 else "neutral")] + titles[:2]


def _news_recency_weight(item: dict) -> float:
    try:
        days_old = float(item.get("days_old"))
    except (TypeError, ValueError):
        return 0.4 if item.get("time_published") else 0.0
    if days_old <= 2:
        return 1.0
    if days_old <= 7:
        return 0.8
    if days_old <= 30:
        return 0.5
    return 0.0


def _intelligence_score(intelligence: dict | None) -> tuple[float, list[str], list[str]]:
    if intelligence is None:
        return 50, [], []
    if not intelligence:
        return 50, ["No company intelligence evidence available"], []

    sector_fit = intelligence.get("sector_fit", {}) or {}
    try:
        score = float(sector_fit.get("score", 50))
    except (TypeError, ValueError):
        score = 50

    reasons = list(sector_fit.get("reasons", [])[:3])
    risks = []
    freshness = intelligence.get("freshness", {}) or {}
    if not freshness.get("last_30_days"):
        score = min(score, 65)
        reasons.append("No company-specific evidence found in the last 30 days")
    elif not freshness.get("last_7_days"):
        score = min(score, 75)
        reasons.append("No company-specific evidence found in the last 7 days")

    positive_catalysts = intelligence.get("positive_catalysts", []) or []
    risk_flags = intelligence.get("risk_flags", []) or []
    if positive_catalysts:
        score += min(10, len(positive_catalysts) * 3)
        reasons.append("Material catalyst evidence is available")
    if risk_flags:
        score -= min(15, len(risk_flags) * 5)
        risks.extend(str(item.get("title")) for item in risk_flags if item.get("title"))

    for warning in intelligence.get("warnings", []) or []:
        if "official" in warning.lower() or "coverage" in warning.lower():
            risks.append(warning)
            break

    return max(0, min(100, round(score, 2))), reasons, risks


def _risk_score(i: dict, f: dict) -> tuple[float, list[str]]:
    score = 70
    risks = []
    vol = i.get("volatility_20d") if i else None
    downside_vol = i.get("downside_volatility_20d") if i else None
    max_drawdown = i.get("max_drawdown") if i else None
    gap_risk = i.get("gap_risk_20d") if i else None
    atr_percent = i.get("atr_percent_14") if i else None
    avg_dollar_volume = i.get("avg_dollar_volume_20") if i else None
    beta = f.get("beta") if f else None
    promoter_pledge = f.get("promoter_pledge") if f else None
    promoter_change = f.get("promoter_holding_qoq_change") if f else None
    atr = i.get("atr_14") if i else None
    close = i.get("close") if i else None

    if vol and vol > 0.55:
        score -= 20; risks.append("High recent volatility")
    elif vol and vol < 0.30:
        score += 5
    if beta and beta > 1.5:
        score -= 10; risks.append("High beta versus market")
    if downside_vol and downside_vol > 0.45:
        score -= 10; risks.append("High downside volatility")
    if max_drawdown and max_drawdown < -0.30:
        score -= 12; risks.append("Large historical drawdown in available window")
    if gap_risk and gap_risk > 0.06:
        score -= 8; risks.append("Recent gap risk is elevated")
    if atr and close and atr / close > 0.05:
        score -= 10; risks.append("Large ATR relative to price")
    if atr_percent and atr_percent < 0.025:
        score += 3
    if avg_dollar_volume is not None and avg_dollar_volume < 5_000_000:
        score -= 10; risks.append("Liquidity is thin by dollar volume")
    if promoter_pledge is not None:
        if promoter_pledge > 20:
            score -= 15; risks.append("Promoter pledge is high")
        elif promoter_pledge > 5:
            score -= 6; risks.append("Promoter pledge exists")
    if promoter_change is not None and promoter_change < -2:
        score -= 8; risks.append("Promoter holding declined materially quarter over quarter")
    return max(0, min(100, round(score, 2))), risks or ["No major risk flags from available data"]


def _momentum_liquidity_score(i: dict) -> tuple[float, list[str]]:
    if not i:
        return 50, ["No momentum or liquidity data available"]
    score = 50
    reasons = []
    ret20 = i.get("return_20d")
    vol_ratio = i.get("volume_ratio")
    volatility = i.get("volatility_20d")
    stoch_k = i.get("stoch_k_14")
    roc = i.get("roc_12")
    cci = i.get("cci_20")
    williams = i.get("williams_r_14")
    mfi = i.get("mfi_14")
    cmf = i.get("cmf_20")
    obv = i.get("obv")
    vwap = i.get("vwap")
    close = i.get("close")
    liquidity_score = i.get("liquidity_score")
    distance_from_52w_high = i.get("distance_from_52w_high")
    distance_from_52w_low = i.get("distance_from_52w_low")

    if ret20 is not None:
        if ret20 > 0.08:
            score += 16; reasons.append("20-day momentum is strong")
        elif ret20 > 0:
            score += 8; reasons.append("20-day momentum is positive")
        elif ret20 < -0.08:
            score -= 12; reasons.append("20-day momentum is weak")
    if vol_ratio is not None:
        if vol_ratio > 1.5:
            score += 12; reasons.append("Volume is materially above recent average")
        elif vol_ratio > 1.1:
            score += 6; reasons.append("Volume is above recent average")
        elif vol_ratio < 0.6:
            score -= 6; reasons.append("Volume is below recent average")
    if roc is not None:
        if roc > 5:
            score += 8; reasons.append("Rate of change is positive")
        elif roc < -5:
            score -= 8; reasons.append("Rate of change is negative")
    if stoch_k is not None:
        if 20 <= stoch_k <= 80:
            score += 4; reasons.append("Stochastic oscillator is not at an extreme")
        elif stoch_k > 90:
            score -= 4; reasons.append("Stochastic oscillator is overextended")
    if cci is not None:
        if cci > 100:
            score += 5; reasons.append("CCI shows strong upside momentum")
        elif cci < -100:
            score -= 5; reasons.append("CCI shows downside momentum")
    if williams is not None:
        if -80 < williams < -20:
            score += 3; reasons.append("Williams %R is away from extremes")
        elif williams > -10:
            score -= 3; reasons.append("Williams %R is overbought")
    if mfi is not None:
        if 35 <= mfi <= 75:
            score += 5; reasons.append("Money flow is constructive")
        elif mfi > 85:
            score -= 4; reasons.append("Money flow is overheated")
        elif mfi < 20:
            score -= 4; reasons.append("Money flow is weak")
    if cmf is not None:
        if cmf > 0.05:
            score += 6; reasons.append("Accumulation/distribution is positive")
        elif cmf < -0.05:
            score -= 6; reasons.append("Accumulation/distribution is negative")
    if close and vwap:
        if close > vwap:
            score += 4; reasons.append("Price is above VWAP")
        else:
            score -= 2; reasons.append("Price is below VWAP")
    if obv is not None and obv > 0:
        score += 2
    if liquidity_score is not None:
        if liquidity_score >= 80:
            score += 5; reasons.append("Liquidity profile is strong")
        elif liquidity_score < 45:
            score -= 8; reasons.append("Liquidity profile is weak")
    if distance_from_52w_high is not None and distance_from_52w_high > -0.05:
        score += 5; reasons.append("Price is within 5% of its 52-week high")
    if distance_from_52w_low is not None and distance_from_52w_low < 0.10:
        score -= 5; reasons.append("Price is close to its 52-week low")
    if volatility is not None and volatility > 0.8:
        score -= 8; reasons.append("Very high volatility reduces setup quality")
    return max(0, min(100, round(score, 2))), reasons or ["Momentum and liquidity are neutral"]


def _signal(score: float) -> str:
    if score >= 80:
        return "Strong Buy Watch"
    if score >= 65:
        return "Buy Watch"
    if score >= 45:
        return "Hold / Monitor"
    if score >= 30:
        return "Avoid / Weak"
    return "Sell / Strong Avoid"


def to_dict(score: StockScore) -> dict:
    return asdict(score)


def _ownership_governance_score(f: dict) -> tuple[float, list[str]]:
    score = 50
    reasons = []
    promoter = f.get("promoter_holding")
    promoter_change = f.get("promoter_holding_qoq_change")
    pledge = f.get("promoter_pledge")
    fii = f.get("fii_holding")
    fii_change = f.get("fii_holding_qoq_change")
    dii = f.get("dii_holding")
    dii_change = f.get("dii_holding_qoq_change")
    mf = f.get("mf_holding")
    shareholder_change = f.get("shareholder_count_qoq_change")

    if promoter is not None:
        if promoter >= 50:
            score += 14; reasons.append("Promoter holding is strong")
        elif promoter >= 35:
            score += 7; reasons.append("Promoter holding is meaningful")
        elif promoter < 10:
            score -= 6; reasons.append("Promoter holding is low or company is professionally managed")
    if promoter_change is not None:
        if promoter_change > 1:
            score += 8; reasons.append("Promoter holding increased quarter over quarter")
        elif promoter_change < -1:
            score -= 10; reasons.append("Promoter holding decreased quarter over quarter")
    if pledge is not None:
        if pledge == 0:
            score += 10; reasons.append("No promoter pledge reported")
        elif pledge <= 5:
            score -= 3; reasons.append("Promoter pledge is low")
        elif pledge <= 20:
            score -= 12; reasons.append("Promoter pledge is notable")
        else:
            score -= 25; reasons.append("Promoter pledge is high")
    institutional = sum(value or 0 for value in [fii, dii, mf])
    if institutional >= 20:
        score += 8; reasons.append("Institutional ownership is meaningful")
    elif institutional > 0:
        score += 3; reasons.append("Institutional ownership is present")
    if fii_change is not None and fii_change > 1:
        score += 4; reasons.append("FII holding increased quarter over quarter")
    elif fii_change is not None and fii_change < -1:
        score -= 4; reasons.append("FII holding decreased quarter over quarter")
    if dii_change is not None and dii_change > 1:
        score += 4; reasons.append("DII holding increased quarter over quarter")
    elif dii_change is not None and dii_change < -1:
        score -= 4; reasons.append("DII holding decreased quarter over quarter")
    if shareholder_change is not None and shareholder_change > 0:
        score += 2; reasons.append("Shareholder base expanded quarter over quarter")
    return max(0, min(100, round(score, 2))), reasons

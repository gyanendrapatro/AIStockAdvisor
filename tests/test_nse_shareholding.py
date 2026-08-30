from stock_advisor.data.nse_shareholding import (
    build_shareholding_index,
    extract_total_paid_up_shares,
)

# Real fragment from a 2025-10-31 taxonomy filing (Vipul Limited), captured live this session.
_XBRL_2025_TAXONOMY = """
<in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">25721691</in-bse-shp:NumberOfFullyPaidUpEquityShares>
<in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="PublicShareholding_ContextI">115237789</in-bse-shp:NumberOfFullyPaidUpEquityShares>
<in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="ShareholdingPattern_ContextI">140959480</in-bse-shp:NumberOfFullyPaidUpEquityShares>
"""

# Real fragment from a 2022-09-30 taxonomy filing (Tata Steel, dated 31-Dec-2023), captured live
# this session -- the grand-total context ID here has no "_Context" infix, unlike the 2025 file.
_XBRL_2022_TAXONOMY = """
<in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="IndianI">4143594780</in-bse-shp:NumberOfFullyPaidUpEquityShares>
<in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="PublicShareholdingI">8069374919</in-bse-shp:NumberOfFullyPaidUpEquityShares>
<in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="ShareholdingPatternI">12212969699</in-bse-shp:NumberOfFullyPaidUpEquityShares>
"""


def test_extract_total_paid_up_shares_current_taxonomy():
    assert extract_total_paid_up_shares(_XBRL_2025_TAXONOMY) == 140959480.0


def test_extract_total_paid_up_shares_older_taxonomy():
    assert extract_total_paid_up_shares(_XBRL_2022_TAXONOMY) == 12212969699.0


def test_extract_total_paid_up_shares_returns_none_when_no_grand_total_context():
    assert extract_total_paid_up_shares('<in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="SomeOtherContextI">123</in-bse-shp:NumberOfFullyPaidUpEquityShares>') is None
    assert extract_total_paid_up_shares("") is None


def test_build_shareholding_index_keeps_latest_filing_per_key():
    records = [
        {"symbol": "AAKAAR", "isin": "INE1GYP01013", "date": "31-DEC-2025"},
        {"symbol": "AAKAAR", "isin": "INE1GYP01013", "date": "31-MAR-2026"},
    ]
    by_symbol, by_isin = build_shareholding_index(records)
    assert by_symbol["AAKAAR"]["date"] == "31-MAR-2026"
    assert by_isin["INE1GYP01013"]["date"] == "31-MAR-2026"


def test_build_shareholding_index_keys_by_symbol_and_isin():
    records = [{"symbol": "RELIANCE", "isin": "INE002A01018", "date": "30-JUN-2026"}]
    by_symbol, by_isin = build_shareholding_index(records)
    assert "RELIANCE" in by_symbol
    assert "INE002A01018" in by_isin


def test_build_shareholding_index_skips_blank_keys():
    records = [{"symbol": "", "isin": "", "date": "30-JUN-2026"}]
    by_symbol, by_isin = build_shareholding_index(records)
    assert by_symbol == {}
    assert by_isin == {}

from stock_advisor.data.screener import _parse_crore_rupees, _parse_rupees, screener_slug


def test_screener_slug_strips_exchange_suffix():
    assert screener_slug("RELIANCE.NS") == "RELIANCE"
    assert screener_slug("544412.BO") == "544412"
    assert screener_slug("AAPL") is None  # no .NS/.BO suffix -> not an Indian equity ticker


def test_parse_crore_rupees_ignores_trailing_period_in_cr():
    # "Cr." has its own period, distinct from the number's decimal point -- a naive
    # strip-non-digits parse would glue them into an unparseable "93.3." (the bug this guards).
    assert _parse_crore_rupees("₹ 93.3 Cr.") == 933000000.0


def test_parse_crore_rupees_handles_indian_comma_grouping():
    assert _parse_crore_rupees("₹ 17,45,629 Cr.") == 17456290000000.0


def test_parse_crore_rupees_returns_none_when_blank():
    assert _parse_crore_rupees("₹ Cr.") is None


def test_parse_rupees_handles_comma_and_decimal():
    assert _parse_rupees("₹ 1,290") == 1290.0
    assert _parse_rupees("₹ 65.8") == 65.8
    assert _parse_rupees("₹") is None

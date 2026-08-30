import pypdf

from stock_advisor.data import iics_classification as iics


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdfReader:
    def __init__(self, *_args, **_kwargs) -> None:
        # Mirrors the real PDF's wrapped, whitespace-heavy layout -- codes never wrap, but
        # names/definitions do, which is exactly why parse_iics_taxonomy_pdf only extracts codes.
        self.pages = [
            _FakePage(
                "MEI_CODE Macro Economic \nIndicator\nSECT_CODE Sector IND_CODE Industry "
                "BASIC_IND_CODE Basic Industry Definition\n"
                "IN01 Commodities IN0101 Chemicals IN010101 Chemicals & \nPetrochemicals\n"
                "IN010101001 Commodity Chemicals Manufacturers of basic and industrial \n"
                "chemicals like synthetic fibres, films, organic and inorganic \nchemicals etc.\n"
                "IN01 Commodities IN0101 Chemicals IN010101 Chemicals & \nPetrochemicals\n"
                "IN010101002 Specialty Chemicals Manufacturers of chemicals used in the \n"
                "manufacture of a variety of products.\n"
                "IN03 Energy IN0301 Oil, Gas & \nConsumable Fuels\nIN030103 Petroleum Products\n"
                "IN030103001 Refineries & Marketing Manufacturer of petroleum products.\n"
            )
        ]


def test_parse_iics_taxonomy_pdf_extracts_nested_codes(monkeypatch):
    monkeypatch.setattr(pypdf, "PdfReader", _FakePdfReader)

    rows = iics.parse_iics_taxonomy_pdf(b"fake-pdf-bytes")

    assert len(rows) == 3
    assert rows[0] == {
        "macro_code": "IN01",
        "sector_code": "IN0101",
        "industry_code": "IN010101",
        "basic_industry_code": "IN010101001",
    }
    assert rows[2]["basic_industry_code"] == "IN030103001"
    assert rows[2]["macro_code"] == "IN03"


def test_parse_iics_taxonomy_pdf_dedupes_repeated_basic_industry_codes(monkeypatch):
    # Every row in the real PDF is one basic industry per line -- codes never repeat -- but guard
    # against it anyway (e.g. a page-break artifact re-emitting a header row).
    class _DupPage:
        def extract_text(self):
            return "IN01 X IN0101 Y IN010101 Z IN010101001 W " * 2

    class _DupReader:
        def __init__(self, *_a, **_k):
            self.pages = [_DupPage()]

    monkeypatch.setattr(pypdf, "PdfReader", _DupReader)

    rows = iics.parse_iics_taxonomy_pdf(b"fake-pdf-bytes")

    assert len(rows) == 1


_MARKET_PAGE_HTML = """
<html><body>
<div class="breadcrumb hidden-if-empty">
  <ul>
    <li><a href="/market/">Industries</a></li>
    <li><a href="/market/IN03/">Energy</a></li>
    <li><a href="/market/IN03/IN0301/">Oil, Gas &amp; Consumable Fuels</a></li>
    <li><a href="/market/IN03/IN0301/IN030103/">Petroleum Products</a></li>
    <li>Refineries &amp; Marketing</li>
  </ul>
</div>
<div class="sub" data-page-info>13 results found: Showing page 1 of 1</div>
<a href="/company/RELIANCE/consolidated/">Reliance</a>
<a href="/company/523232/">Some BSE Co</a>
<a href="/company/RELIANCE/consolidated/">Reliance duplicate link</a>
</body></html>
"""


def test_fetch_iics_basic_industry_companies_parses_breadcrumb_and_slugs(monkeypatch):
    class _FakeResponse:
        status_code = 200
        text = _MARKET_PAGE_HTML

        def raise_for_status(self):
            return None

    monkeypatch.setattr(iics.requests, "get", lambda url, headers=None, timeout=None: _FakeResponse())

    result = iics.fetch_iics_basic_industry_companies("IN03", "IN0301", "IN030103", "IN030103001")

    assert result["names"] == {
        "macro_sector": "Energy",
        "sector": "Oil, Gas & Consumable Fuels",
        "industry": "Petroleum Products",
        "basic_industry": "Refineries & Marketing",
    }
    assert result["slugs"] == ["523232", "RELIANCE"]  # deduped, sorted
    assert result["page"] == 1
    assert result["total_pages"] == 1


def test_resolve_iics_slug_to_ticker_distinguishes_numeric_and_alpha():
    assert iics.resolve_iics_slug_to_ticker("RELIANCE") == "RELIANCE.NS"
    assert iics.resolve_iics_slug_to_ticker("523232") == "523232.BO"
    assert iics.resolve_iics_slug_to_ticker("") is None


def test_build_iics_classification_mapping_walks_and_resolves(monkeypatch):
    monkeypatch.setattr(iics, "DEFAULT_REQUEST_DELAY_SECONDS", 0)

    def _fake_fetch(macro_code, sector_code, industry_code, basic_industry_code, *, session=None, timeout=20.0):
        return {
            "names": {
                "macro_sector": "Energy",
                "sector": "Oil, Gas & Consumable Fuels",
                "industry": "Petroleum Products",
                "basic_industry": "Refineries & Marketing",
            },
            "slugs": ["RELIANCE", "523232"],
            "page": 1,
            "total_pages": 1,
        }

    monkeypatch.setattr(iics, "fetch_iics_basic_industry_companies", _fake_fetch)

    taxonomy_rows = [
        {"macro_code": "IN03", "sector_code": "IN0301", "industry_code": "IN030103", "basic_industry_code": "IN030103001"}
    ]
    result = iics.build_iics_classification_mapping(taxonomy_rows, delay_seconds=0)

    assert result["basic_industries_walked"] == 1
    assert result["basic_industries_failed"] == 0
    assert result["incomplete_basic_industries"] == []
    assert result["tickers_resolved"] == 2
    by_ticker = {row["ticker"]: row for row in result["rows"]}
    assert by_ticker["RELIANCE.NS"]["basic_industry"] == "Refineries & Marketing"
    assert by_ticker["523232.BO"]["macro_sector_code"] == "IN03"


def test_build_iics_classification_mapping_flags_incomplete_pages(monkeypatch):
    def _fake_fetch(macro_code, sector_code, industry_code, basic_industry_code, *, session=None, timeout=20.0):
        return {"names": {}, "slugs": [], "page": 1, "total_pages": 2}

    monkeypatch.setattr(iics, "fetch_iics_basic_industry_companies", _fake_fetch)

    taxonomy_rows = [
        {"macro_code": "IN03", "sector_code": "IN0301", "industry_code": "IN030103", "basic_industry_code": "IN030103001"}
    ]
    result = iics.build_iics_classification_mapping(taxonomy_rows, delay_seconds=0)

    assert result["incomplete_basic_industries"] == ["IN030103001"]

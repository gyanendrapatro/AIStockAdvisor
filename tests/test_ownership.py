from stock_advisor.data.ownership import get_ownership_fundamentals, ownership_data_file_exists


def test_get_ownership_fundamentals_reads_yaml_history(tmp_path):
    path = tmp_path / "ownership.yaml"
    path.write_text(
        """
RELIANCE.NS:
  history:
    - quarter: "2025-12"
      promoter_holding: 50.1
      promoter_pledge: 1.5
      fii_holding: 18.0
      dii_holding: 12.0
      mf_holding: 8.0
      public_holding: 20.0
      shareholder_count: 1000000
      source: "manual"
    - quarter: "2026-03"
      promoter_holding: 51.3
      promoter_pledge: 0.0
      fii_holding: 19.2
      dii_holding: 12.5
      mf_holding: 8.2
      public_holding: 17.0
      shareholder_count: 1100000
      source: "manual"
""",
        encoding="utf-8",
    )

    result = get_ownership_fundamentals("RELIANCE.NS", path)

    assert result["_sources"] == ["local_ownership"]
    assert result["ownership_quarter"] == "2026-03"
    assert result["promoter_holding"] == 51.3
    assert result["promoter_holding_qoq_change"] == 1.2
    assert result["promoter_pledge"] == 0
    assert result["shareholder_count_qoq_change"] == 100000


def test_get_ownership_fundamentals_reads_csv(tmp_path):
    path = tmp_path / "ownership.csv"
    path.write_text(
        "ticker,quarter,promoter_holding,promoter_pledge,fii_holding,dii_holding,source\n"
        "TCS,2026-03,71.8,0,12.2,9.1,manual\n",
        encoding="utf-8",
    )

    result = get_ownership_fundamentals("TCS.NS", path)

    assert result["promoter_holding"] == 71.8
    assert result["ownership_source"] == "manual"


def test_ownership_data_file_exists_checks_resolved_path(tmp_path):
    missing_path = tmp_path / "missing.yaml"
    present_path = tmp_path / "ownership.yaml"
    present_path.write_text("{}", encoding="utf-8")

    assert ownership_data_file_exists(missing_path) is False
    assert ownership_data_file_exists(present_path) is True

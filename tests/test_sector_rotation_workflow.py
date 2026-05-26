from stock_advisor.agents import sector_rotation_workflow


def test_sector_rotation_workflow_runs_fresh_steps(monkeypatch):
    monkeypatch.setattr(
        sector_rotation_workflow,
        "get_sector_rotation",
        lambda **kwargs: {
            "top_sector": {"sector_id": "auto"},
            "sectors": [
                {
                    "sector_id": "auto",
                    "name": "Auto",
                    "stage": "Emerging",
                    "rotation_score": 70,
                    "trend_score": 65,
                    "acceleration_score": 70,
                    "relative_strength": {"vs_benchmark_20d": 0.02, "vs_benchmark_60d": 0.0},
                    "metrics": {"return_5d": 0.01, "return_20d": 0.05, "return_60d": 0.08},
                    "breadth": {"breadth_score": 60, "above_sma_50_percent": 75},
                }
            ],
            "warnings": ["sector warning"],
            "kwargs": kwargs,
        },
    )
    monkeypatch.setattr(
        sector_rotation_workflow,
        "rank_sector_stocks",
        lambda sector, **kwargs: {
            "sector_id": sector,
            "stocks": [
                {
                    "ticker": "AAA.NS",
                    "stage": "Improving",
                    "stock_score": 75,
                    "trend_score": 70,
                    "pattern_score": 62,
                    "volume_score": 58,
                    "risk_quality_score": 72,
                    "relative_strength": {"vs_sector_20d": 0.03, "vs_sector_60d": 0.01},
                    "reasons": ["Outperforming its sector over 20 days."],
                }
            ],
            "warnings": ["stock warning"],
            "kwargs": kwargs,
        },
    )

    result = sector_rotation_workflow.run_sector_rotation_workflow(
        period="auto",
        max_sectors=5,
        max_breadth_stocks=3,
        stocks_per_sector=4,
        selected_sector="auto",
    )

    assert result["workflow"] == "sector_rotation_workflow"
    assert result["mode"] == "fresh_realtime_run"
    assert result["cache_used"] is False
    assert result["inputs"]["requested_period"] == "auto"
    assert result["inputs"]["analysis_period"] == "1y"
    assert result["inputs"]["auto_period"] is True
    assert result["rotation"]["kwargs"]["max_sectors"] == 5
    assert result["rotation"]["kwargs"]["period"] == "1y"
    assert result["ranked_stocks"]["sector_id"] == "auto"
    assert result["ranked_stocks"]["kwargs"]["max_stocks"] == 4
    assert result["ranked_stocks"]["kwargs"]["period"] == "1y"
    assert result["decision_summary"]["target_next_sector"]["movement_status"] == "upcoming"
    assert result["decision_summary"]["target_next_sector"]["return_20d_direction"] == "up"
    assert result["decision_summary"]["target_next_sector"]["rs_60d_direction"] == "flat"
    assert result["decision_summary"]["top_stock_target"]["ticker"] == "AAA.NS"
    assert "Target next: Auto" in result["decision_summary"]["headline"]
    assert result["decision_method"]["auto_period"] is True
    assert any("Auto period selected" in step for step in result["steps"])
    assert result["indicator_explanations"]["sector_filters"][0]["name"] == "Sector period"
    assert "Auto" in result["indicator_explanations"]["sector_filters"][0]["how_to_read"]
    assert any(row["column"] == "rotation_score" for row in result["indicator_explanations"]["sector_columns"])
    assert any(row["column"] == "movement_arrow" for row in result["indicator_explanations"]["sector_columns"])
    assert any(row["column"] == "stock_score" for row in result["indicator_explanations"]["stock_columns"])
    assert result["warnings"] == ["sector warning", "stock warning"]

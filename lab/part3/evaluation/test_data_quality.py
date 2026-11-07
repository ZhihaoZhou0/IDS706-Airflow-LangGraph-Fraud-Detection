"""Evaluation tests for Part 3 data quality and safe termination."""

from lab.part3.multi_agent_graph import build_graph


def test_missing_dataset_stops_before_training():
    graph = build_graph()

    result = graph.invoke(
        {
            "mode": "real",
            "data_path": "nonexistent_test_file.csv",
            "max_cycles": 3,
            "max_retries": 2,
        },
        config={"recursion_limit": 50},
    )

    assert result["stop_reason"] == "data_quality_failed"
    assert result["data_quality_report"]["passed"] is False
    assert result.get("cycle", 0) == 0

    agents = [
        step["agent"]
        for step in result["execution_trace"]
    ]

    assert "data_quality_agent" in agents
    assert "training_agent" not in agents
    assert "recovery_agent" not in agents


def test_simulation_still_reaches_target():
    graph = build_graph()

    result = graph.invoke(
        {
            "mode": "simulation",
            "max_cycles": 3,
            "target_f1": 0.3,
            "max_retries": 2,
        },
        config={"recursion_limit": 50},
    )

    assert result["stop_reason"] == "target_reached"
    assert result["best_spec"]["model_family"] == (
        "hist_gradient_boosting"
    )
    assert result["best_val_f1"] == 0.35
    assert result.get("holdout_metrics") is None

def test_recovery_respects_retry_limit():
    graph = build_graph()

    result = graph.invoke(
        {
            "mode": "simulation",
            "data_checked": True,
            "plan": {"model_family": "invalid_model"},
            "max_retries": 1,
            "max_cycles": 3,
            "target_f1": 0.3,
        },
        config={"recursion_limit": 50},
    )

    agents = [
        step["agent"]
        for step in result["execution_trace"]
    ]

    assert result["retry_count"] == 1
    assert agents.count("recovery_agent") == 1
    assert result["stop_reason"] == "target_reached"
    assert result["best_spec"]["model_family"] == (
        "hist_gradient_boosting"
    )

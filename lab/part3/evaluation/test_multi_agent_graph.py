"""Integration tests for the Part 3 multi-agent workflow."""

from lab.part3.multi_agent_graph import build_graph


def run_graph(state):
    return build_graph().invoke(
        state,
        config={"recursion_limit": 50},
    )


def agent_names(state):
    return [
        event["agent"]
        for event in state.get("execution_trace", [])
    ]


def test_normal_workflow_reaches_target():
    result = run_graph({
        "max_cycles": 3,
        "target_f1": 0.30,
        "max_retries": 2,
    })

    assert result["stop_reason"] == "target_reached"
    assert result["cycle"] == 3
    assert result["best_val_f1"] == 0.35
    assert "planning_agent" in agent_names(result)
    assert "evaluation_agent" in agent_names(result)


def test_recovery_after_invalid_model():
    result = run_graph({
        "data_checked": True,
        "plan": {"model_family": "invalid_model"},
        "max_cycles": 2,
        "target_f1": 0.30,
        "max_retries": 2,
    })

    agents = agent_names(result)

    assert result["retry_count"] == 1
    assert "recovery_agent" in agents
    assert result["cycle"] == 2
    assert agents.index("recovery_agent") < agents.index(
        "evaluation_agent"
    )


def test_zero_retries_stops_immediately():
    result = run_graph({
        "data_checked": True,
        "plan": {"model_family": "invalid_model"},
        "max_cycles": 3,
        "target_f1": 0.30,
        "max_retries": 0,
    })

    assert result["stop_reason"] == "retry_limit"
    assert result.get("retry_count", 0) == 0
    assert result.get("cycle", 0) == 0
    assert "recovery_agent" not in agent_names(result)

"""Part 3 adaptive multi-agent graph: routing and recovery smoke tests."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from lab.part3.planner import choose_model, choose_next_agent


class MultiAgentState(TypedDict, total=False):
    data_checked: bool
    data_quality_report: dict
    mode: str
    data_path: str
    seed: int
    holdout_frac: float
    training_error: str
    retry_count: int
    max_retries: int
    current_result: dict
    best_val_f1: float
    target_f1: float
    cycle: int
    max_cycles: int
    history: list
    best_spec: dict
    plan: dict
    execution_trace: list
    stop_reason: str
    holdout_metrics: dict
    holdout_error: str


def supervisor_agent(state: MultiAgentState) -> dict:
    """Record the next action selected by the dynamic planner."""
    decision = choose_next_agent(state)
    trace = [*state.get("execution_trace", []), {
        "agent": "supervisor_agent",
        "decision": decision,
    }]
    return {"execution_trace": trace}


def route_from_supervisor(state: MultiAgentState) -> str:
    return choose_next_agent(state)


def data_quality_agent(state: MultiAgentState) -> dict:
    if state.get("mode", "simulation") == "real":
        from lab.part3.data_quality import validate_fraud_dataset

        report = validate_fraud_dataset(
            state.get(
                "data_path",
                "artifacts/ieee_fraud/agent_input.csv",
            )
        )
    else:
        report = {"passed": True, "mode": "simulation"}

    passed = bool(report["passed"])

    return {
        "data_checked": True,
        "data_quality_report": report,
        "training_error": "" if passed else report.get(
            "error", "Data quality validation failed"
        ),
        "execution_trace": [
            *state.get("execution_trace", []),
            {
                "agent": "data_quality_agent",
                "status": "passed" if passed else "failed",
            },
        ],
    }


def planning_agent(state: MultiAgentState) -> dict:
    plan = choose_model(state)
    return {
        "plan": plan,
        "current_result": None,
        "execution_trace": [
            *state.get("execution_trace", []),
            {"agent": "planning_agent", **plan},
        ],
    }


def training_agent(state: MultiAgentState) -> dict:
    """Deterministic simulation; real training will be integrated later."""
    plan = state.get("plan") or choose_model(state)
    family = plan["model_family"]

    mode = state.get("mode", "simulation")

    try:
        if mode == "real":
            from lab.part3.real_training import train_real_model

            result = train_real_model(
                family,
                data_path=state.get(
                    "data_path",
                    "artifacts/ieee_fraud/agent_input.csv",
                ),
                seed=state.get("seed", 42),
                holdout_frac=state.get("holdout_frac", 0.2),
                hyperparameters=plan.get("hyperparameters"),
            )
            score = float(result["validation_metrics"]["f1"])

        elif mode == "simulation":
            scores = {
                "logistic_regression": 0.08,
                "random_forest": 0.26,
                "hist_gradient_boosting": 0.35,
            }

            if family not in scores:
                raise ValueError(f"Unsupported model: {family}")

            score = scores[family]
            result = {
                "model_family": family,
                "validation_metrics": {"f1": score},
            }

        else:
            raise ValueError(f"Unsupported training mode: {mode}")

    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        return {
            "current_result": None,
            "training_error": str(exc),
            "execution_trace": [
                *state.get("execution_trace", []),
                {
                    "agent": "training_agent",
                    "status": "failed",
                    "error": str(exc),
                },
            ],
        }
    return {
        "current_result": result,
        "training_error": "",
        "execution_trace": [
            *state.get("execution_trace", []),
            {"agent": "training_agent", "status": "succeeded", "f1": score},
        ],
    }


def evaluation_agent(state: MultiAgentState) -> dict:
    result = state["current_result"]
    f1 = float(result["validation_metrics"]["f1"])
    history = [
        *state.get("history", []),
        {"model_family": result["model_family"], "f1": f1},
    ]
    best_f1 = float(state.get("best_val_f1", -1.0))
    improved = f1 > best_f1
    return {
        "history": history,
        "cycle": state.get("cycle", 0) + 1,
        "best_val_f1": f1 if improved else best_f1,
        "best_spec": {"model_family": result["model_family"]}
        if improved else state.get("best_spec", {}),
        "execution_trace": [
            *state.get("execution_trace", []),
            {"agent": "evaluation_agent", "f1": f1, "improved": improved},
        ],
    }


def recovery_agent(state: MultiAgentState) -> dict:
    retries = state.get("retry_count", 0) + 1
    return {
        "retry_count": retries,
        "plan": {
            "model_family": "random_forest",
            "reason": "recover_from_training_error",
        },
        "training_error": "",
        "current_result": None,
        "execution_trace": [
            *state.get("execution_trace", []),
            {"agent": "recovery_agent", "retry": retries},
        ],
    }


def finalize_agent(state: MultiAgentState) -> dict:
    """Finalize the workflow with an explicit stopping reason."""

    quality_report = state.get("data_quality_report") or {}

    if quality_report.get("passed") is False:
        reason = "data_quality_failed"
    elif state.get("training_error"):
        reason = "retry_limit"
    elif state.get("best_val_f1", 0) >= state.get("target_f1", 0.3):
        reason = "target_reached"
    else:
        reason = "max_cycles"
    holdout_metrics = None
    holdout_error = None

    if (
        state.get("mode", "simulation") == "real"
        and reason in ("target_reached", "max_cycles")
        and state.get("best_spec")
    ):
        from lab.part3.real_training import train_real_model

        try:
            best_spec = state["best_spec"]
            holdout_metrics = train_real_model(
                best_spec["model_family"],
                data_path=state.get(
                    "data_path",
                    "artifacts/ieee_fraud/agent_input.csv",
                ),
                seed=state.get("seed", 42),
                holdout_frac=state.get("holdout_frac", 0.2),
                hyperparameters=best_spec.get("hyperparameters"),
                on_holdout=True,
            )
        except Exception as exc:
            holdout_error = str(exc)

    return {
        "stop_reason": reason,
        "holdout_metrics": holdout_metrics,
        "holdout_error": holdout_error,
        "execution_trace": [
            *state.get("execution_trace", []),
            {
                "agent": "finalize_agent",
                "reason": reason,
                "holdout_completed": holdout_metrics is not None,
            },
        ],
    }


def build_graph():
    graph = StateGraph(MultiAgentState)

    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("data_quality_agent", data_quality_agent)
    graph.add_node("planning_agent", planning_agent)
    graph.add_node("training_agent", training_agent)
    graph.add_node("evaluation_agent", evaluation_agent)
    graph.add_node("recovery_agent", recovery_agent)
    graph.add_node("finalize_agent", finalize_agent)

    graph.add_edge(START, "supervisor_agent")

    graph.add_conditional_edges(
        "supervisor_agent",
        route_from_supervisor,
        {
            "data_quality_agent": "data_quality_agent",
            "planning_agent": "planning_agent",
            "training_agent": "training_agent",
            "recovery_agent": "recovery_agent",
            "finalize_agent": "finalize_agent",
        },
    )

    graph.add_edge("data_quality_agent", "supervisor_agent")
    graph.add_edge("planning_agent", "supervisor_agent")
    graph.add_conditional_edges(
        "training_agent",
        lambda state: (
            "supervisor_agent" if state.get("training_error")
            else "evaluation_agent"
        ),
        {
            "supervisor_agent": "supervisor_agent",
            "evaluation_agent": "evaluation_agent",
        },
    )

    graph.add_edge("evaluation_agent", "supervisor_agent")
    graph.add_edge("recovery_agent", "supervisor_agent")
    graph.add_edge("finalize_agent", END)

    return graph.compile()

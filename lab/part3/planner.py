"""Dynamic planning policies for the Part 3 multi-agent workflow."""

from __future__ import annotations

from typing import Any


MODEL_SEQUENCE = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)


def choose_next_agent(state: dict[str, Any]) -> str:
    """Select the next agent based on current workflow state."""

    if not state.get("data_checked", False):
        return "data_quality_agent"

    if state.get("data_quality_report", {}).get("passed") is False:
        return "finalize_agent"

    if state.get("training_error"):
        if state.get("retry_count", 0) < state.get("max_retries", 2):
            return "recovery_agent"
        return "finalize_agent"

    if state.get("current_result") is None:
        if not state.get("plan"):
            return "planning_agent"
        return "training_agent"

    best_f1 = float(state.get("best_val_f1", 0.0))
    target_f1 = float(state.get("target_f1", 0.3))
    completed = int(state.get("cycle", 0))
    max_cycles = int(state.get("max_cycles", 3))

    if best_f1 >= target_f1 or completed >= max_cycles:
        return "finalize_agent"

    return "planning_agent"


def choose_model(state: dict[str, Any]) -> dict[str, Any]:
    """Choose a candidate model based on previous experiments."""

    history = state.get("history") or []
    tried = {item.get("model_family") for item in history}

    for family in MODEL_SEQUENCE:
        if family not in tried:
            return {
                "model_family": family,
                "reason": "explore_untried_model",
            }

    best = state.get("best_spec") or {}

    return {
        "model_family": best.get(
            "model_family", "hist_gradient_boosting"
        ),
        "reason": "refine_best_model",
    }

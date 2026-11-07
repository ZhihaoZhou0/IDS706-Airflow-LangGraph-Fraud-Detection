"""Tests for Part 3 dynamic planning decisions."""

import pytest

from lab.part3.planner import choose_model, choose_next_agent


@pytest.mark.parametrize(
    "state,expected",
    [
        ({}, "data_quality_agent"),
        ({"data_checked": True}, "planning_agent"),
        (
            {
                "data_checked": True,
                "plan": {"model_family": "logistic_regression"},
            },
            "training_agent",
        ),
        (
            {
                "data_checked": True,
                "training_error": "training failed",
                "retry_count": 0,
                "max_retries": 2,
            },
            "recovery_agent",
        ),
        (
            {
                "data_checked": True,
                "training_error": "training failed",
                "retry_count": 2,
                "max_retries": 2,
            },
            "finalize_agent",
        ),
        (
            {
                "data_checked": True,
                "current_result": {"status": "succeeded"},
                "best_val_f1": 0.15,
                "target_f1": 0.30,
                "cycle": 1,
                "max_cycles": 3,
            },
            "planning_agent",
        ),
        (
            {
                "data_checked": True,
                "current_result": {"status": "succeeded"},
                "best_val_f1": 0.35,
                "target_f1": 0.30,
                "cycle": 2,
                "max_cycles": 3,
            },
            "finalize_agent",
        ),
    ],
)
def test_agent_routing(state, expected):
    assert choose_next_agent(state) == expected


def test_model_selection_starts_with_logistic_regression():
    result = choose_model({"history": []})
    assert result["model_family"] == "logistic_regression"


def test_model_selection_switches_to_random_forest():
    result = choose_model(
        {"history": [{"model_family": "logistic_regression"}]}
    )
    assert result["model_family"] == "random_forest"


def test_model_selection_switches_to_boosting():
    result = choose_model(
        {
            "history": [
                {"model_family": "logistic_regression"},
                {"model_family": "random_forest"},
            ]
        }
    )
    assert result["model_family"] == "hist_gradient_boosting"


def test_model_selection_refines_best_model():
    result = choose_model(
        {
            "history": [
                {"model_family": "logistic_regression"},
                {"model_family": "random_forest"},
                {"model_family": "hist_gradient_boosting"},
            ],
            "best_spec": {"model_family": "random_forest"},
        }
    )
    assert result["model_family"] == "random_forest"
    assert result["reason"] == "refine_best_model"

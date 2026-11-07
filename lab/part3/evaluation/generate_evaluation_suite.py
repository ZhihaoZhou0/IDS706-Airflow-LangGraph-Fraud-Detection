"""Generate the Part 3 evaluation suite from actual pytest results."""

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path("lab/part3")
XML_PATH = ROOT / "results/pytest_results.xml"
SUITE_PATH = ROOT / "evaluation/evaluation_suite.json"
SUMMARY_PATH = ROOT / "results/evaluation_summary.json"

ROUTING_INPUTS = [
    ({}, "data_quality_agent"),
    ({"data_checked": True}, "planning_agent"),
    ({"data_checked": True, "plan": {"model_family": "logistic_regression"}}, "training_agent"),
    ({"data_checked": True, "training_error": "training failed", "retry_count": 0, "max_retries": 2}, "recovery_agent"),
    ({"data_checked": True, "training_error": "training failed", "retry_count": 2, "max_retries": 2}, "finalize_agent"),
    ({"data_checked": True, "current_result": {"status": "succeeded"}, "best_val_f1": 0.15, "target_f1": 0.30, "cycle": 1, "max_cycles": 3}, "planning_agent"),
    ({"data_checked": True, "current_result": {"status": "succeeded"}, "best_val_f1": 0.35, "target_f1": 0.30, "cycle": 2, "max_cycles": 3}, "finalize_agent"),
]

TEST_SPECS = {
    "test_model_selection_starts_with_logistic_regression": (
        "Model Selection", {"history": []}, "Select logistic_regression"
    ),
    "test_model_selection_switches_to_random_forest": (
        "Model Selection", {"history": [{"model_family": "logistic_regression"}]}, "Select random_forest"
    ),
    "test_model_selection_switches_to_boosting": (
        "Model Selection", {"history": [{"model_family": "logistic_regression"}, {"model_family": "random_forest"}]}, "Select hist_gradient_boosting"
    ),
    "test_model_selection_refines_best_model": (
        "Model Selection",
        {"history": [{"model_family": "logistic_regression"}, {"model_family": "random_forest"}, {"model_family": "hist_gradient_boosting"}], "best_spec": {"model_family": "random_forest"}},
        "Select random_forest with reason refine_best_model"
    ),
    "test_normal_workflow_reaches_target": (
        "Workflow Integration",
        {"max_cycles": 3, "target_f1": 0.30, "max_retries": 2},
        "Stop at target_reached, complete 3 cycles, simulated F1=0.35, planning/evaluation agents executed"
    ),
    "test_recovery_after_invalid_model": (
        "Workflow Integration",
        {"data_checked": True, "plan": {"model_family": "invalid_model"}, "max_cycles": 2, "target_f1": 0.30, "max_retries": 2},
        "One recovery, 2 completed cycles, recovery agent executes before evaluation"
    ),
    "test_zero_retries_stops_immediately": (
        "Workflow Integration",
        {"data_checked": True, "plan": {"model_family": "invalid_model"}, "max_cycles": 3, "target_f1": 0.30, "max_retries": 0},
        "Stop at retry_limit without recovery or completed training cycles"
    ),
    "test_missing_dataset_stops_before_training": (
        "Data Quality & Safety",
        {"mode": "real", "data_path": "nonexistent_test_file.csv", "max_cycles": 3, "max_retries": 2},
        "Stop at data_quality_failed; training and recovery agents not executed"
    ),
    "test_simulation_still_reaches_target": (
        "Data Quality & Safety",
        {"mode": "simulation", "max_cycles": 3, "target_f1": 0.3, "max_retries": 2},
        "Select hist_gradient_boosting, reach simulated F1=0.35, no holdout metrics"
    ),
    "test_recovery_respects_retry_limit": (
        "Data Quality & Safety",
        {"mode": "simulation", "data_checked": True, "plan": {"model_family": "invalid_model"}, "max_retries": 1, "max_cycles": 3, "target_f1": 0.3},
        "Exactly one recovery; target_reached with hist_gradient_boosting"
    ),
}


def main():
    tree = ET.parse(XML_PATH)
    cases = tree.findall(".//testcase")
    records = []

    for case in cases:
        name = case.attrib["name"]
        classname = case.attrib.get("classname", "")
        full_id = f"{classname}::{name}"

        if name.startswith("test_agent_routing["):
            index = int(name.split("[state", 1)[1].split("-", 1)[0])
            state, expected = ROUTING_INPUTS[index]
            category = "Agent Routing"
            test_input = state
            expected_behavior = f"Route to {expected}"
        else:
            if name not in TEST_SPECS:
                raise ValueError(f"Missing evaluation specification: {name}")
            category, test_input, expected_behavior = TEST_SPECS[name]

        failed = (
            case.find("failure") is not None
            or case.find("error") is not None
        )
        skipped = case.find("skipped") is not None
        passed = not failed and not skipped

        records.append({
            "test_id": full_id,
            "category": category,
            "input": test_input,
            "expected_behavior": expected_behavior,
            "scoring_criteria": "All test assertions must pass",
            "pass_threshold": "100% of assertions",
            "actual_result": "PASS" if passed else ("SKIPPED" if skipped else "FAIL"),
            "passed": passed,
            "duration_seconds": float(case.attrib.get("time", 0)),
        })

    counts = Counter(r["category"] for r in records)
    passed_counts = Counter(r["category"] for r in records if r["passed"])

    summary = {
        "total_tests": len(records),
        "passed": sum(r["passed"] for r in records),
        "failed_or_skipped": sum(not r["passed"] for r in records),
        "pass_rate": sum(r["passed"] for r in records) / len(records) if records else 0,
        "categories": {
            category: {
                "total": total,
                "passed": passed_counts[category],
                "pass_rate": passed_counts[category] / total,
            }
            for category, total in counts.items()
        },
        "source": str(XML_PATH),
    }

    SUITE_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== EVALUATION SUITE ===")
    print("Total:", summary["total_tests"])
    print("Passed:", summary["passed"])
    print("Pass rate:", f'{summary["pass_rate"]:.1%}')
    for category, stats in summary["categories"].items():
        print(f'{category}: {stats["passed"]}/{stats["total"]}')
    print("Suite:", SUITE_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()

"""Read-only evaluation dashboard for IDS 706 Part 3."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

st.set_page_config(
    page_title="Multi-Agent Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)


def load_json(filename):
    path = RESULTS / filename
    return json.loads(path.read_text(encoding="utf-8"))


def show_image(filename, caption):
    path = FIGURES / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing image: {path.name}")


part1 = load_json("part1_baseline.json")
part2 = load_json("part2_baseline.json")
part3 = load_json("part3_baseline.json")
summary = load_json("evaluation_summary.json")

suite_path = ROOT / "evaluation" / "evaluation_suite.json"
suite = json.loads(suite_path.read_text(encoding="utf-8"))

st.title("Multi-Agent Fraud Detection & Evaluation")
st.caption(
    "IDS 706 | Part 3 Super Bonus | "
    "LangGraph Adaptive Multi-Agent Workflow"
)

overview, comparison, workflow, evaluation = st.tabs(
    [
        "Overview",
        "Baseline Comparison",
        "Adaptive Workflow",
        "Evaluation Suite",
    ]
)

with overview:
    st.subheader("Experiment Overview")

    cols = st.columns(4)
    cols[0].metric(
        "Evaluation Pass Rate",
        f"{summary['pass_rate']:.0%}",
    )
    cols[1].metric(
        "Tests Passed",
        f"{summary['passed']}/{summary['total_tests']}",
    )
    cols[2].metric(
        "Part 3 Holdout F1",
        f"{part3['holdout_metrics']['f1']:.4f}",
    )
    cols[3].metric("Specialized Agents", "7")

    st.markdown(
        """
        **Project goal:** Build and evaluate an adaptive
        multi-agent fraud detection workflow with specialized
        agents for data quality, planning, training, evaluation,
        recovery, and orchestration.

        The system uses LangGraph for dynamic routing,
        model selection, and failure recovery.
        """
    )

    st.info(
        "All model metrics are from saved experiments. "
        "This dashboard does not retrain models or modify the database."
    )

with comparison:
    st.subheader("Three-System Baseline Comparison")

    rows = []
    for name, data in [
        ("Part 1: Fixed Baseline", part1),
        ("Part 2: Simple LangGraph", part2),
        ("Part 3: Multi-Agent", part3),
    ]:
        metrics = data["holdout_metrics"]
        rows.append(
            {
                "System": name,
                "Holdout F1": metrics["f1"],
                "Holdout Accuracy": metrics["accuracy"],
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    show_image(
        "baseline_comparison.png",
        "Holdout performance across three systems",
    )

    st.caption(
        "Controlled comparison using 12,000 records, "
        "seed=42, and a 20% holdout. "
        "Part 1 reproduces the available fixed training task; "
        "its feature set and model differ from Parts 2 and 3. "
        "Performance differences should not be attributed "
        "solely to the multi-agent architecture."
    )

with workflow:
    st.subheader("Supervisor-Centric Multi-Agent Architecture")

    show_image(
        "workflow_graph.png",
        "LangGraph workflow exported from the actual graph",
    )

    st.subheader("Adaptive Model Selection")
    show_image(
        "validation_f1_trend.png",
        "Validation F1 across three model-selection cycles",
    )

    history_rows = []
    for system, data in [
        ("Part 2", part2),
        ("Part 3", part3),
    ]:
        for cycle, item in enumerate(data.get("history", []), 1):
            history_rows.append(
                {
                    "System": system,
                    "Cycle": cycle,
                    "Model": item["model_family"],
                    "Validation F1": item["f1"],
                }
            )

    st.dataframe(
        pd.DataFrame(history_rows),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        **Adaptive workflow capabilities**

        - Supervisor-controlled conditional routing
        - Data quality checks before training
        - Iterative model selection and evaluation
        - Recovery and retry handling
        - Stopping conditions based on target F1 and retry limits
        """
    )

with evaluation:
    st.subheader("Automated Evaluation Suite")

    st.metric(
        "Overall Test Pass Rate",
        f"{summary['pass_rate']:.0%}",
    )

    category_rows = []
    for name, stats in summary["categories"].items():
        category_rows.append(
            {
                "Category": name,
                "Passed": stats["passed"],
                "Total": stats["total"],
                "Pass Rate": stats["pass_rate"],
            }
        )

    category_df = pd.DataFrame(category_rows)
    st.dataframe(
        category_df,
        hide_index=True,
        use_container_width=True,
    )

    st.bar_chart(
        category_df.set_index("Category")[["Passed"]],
    )

    st.subheader("Individual Test Cases")

    test_rows = [
        {
            "Test ID": item["test_id"],
            "Category": item["category"],
            "Expected Behavior": item["expected_behavior"],
            "Result": item["actual_result"],
        }
        for item in suite
    ]

    st.dataframe(
        pd.DataFrame(test_rows),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("View Full Evaluation Specifications"):
        st.json(suite)

    st.caption(
        "Test outcomes are loaded from the saved pytest "
        "JUnit XML-derived evaluation report."
    )

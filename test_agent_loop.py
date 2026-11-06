from lab.langgraph.graph import build_graph

graph = build_graph()

result = graph.invoke({
    "dataset_path": "/repo/datasets/airflow/merged_latest.csv",
    "target_column": "is_fraud",
    "target_score": 0.15,
    "max_cycles": 3,
    "cycle": 0,
    "seed": 42,
    "holdout_frac": 0.2,
    "steps": [],
})

print("\n=== Agent Loop Results ===")
print("Cycles:", result.get("cycle"))
print("Best Validation F1:", result.get("best_val_f1"))
print("Stop Reason:", result.get("stop_reason"))
print("Best Model:", result.get("best_spec"))
print("Holdout Metrics:", result.get("holdout_metrics"))

print("\n=== Agent Execution History ===")
for step in result.get("steps", []):
    print(step)

graph = build_graph()

result = graph.invoke({
    "dataset_path": "/repo/datasets/airflow/merged_latest.csv",
    "target_column": "is_fraud",
    "target_score": 0.15,
    "max_cycles": 3,
    "cycle": 0,
    "seed": 42,
    "holdout_frac": 0.2,
    "steps": [],
})

print("\n=== Agent Loop Results ===")
print("Cycles:", result.get("cycle"))
print("Best Validation F1:", result.get("best_val_f1"))
print("Stop Reason:", result.get("stop_reason"))
print("Best Model:", result.get("best_spec"))
print("Holdout Metrics:", result.get("holdout_metrics"))

print("\n=== Agent Execution History ===")
for step in result.get("steps", []):
    print(step)
"""Visualize saved Part 1/2/3 baseline results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("lab/part3/results")
OUTPUT = RESULTS / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

files = [
    ("Part 1\nFixed Baseline", "part1_baseline.json"),
    ("Part 2\nLangGraph", "part2_baseline.json"),
    ("Part 3\nMulti-Agent", "part3_baseline.json"),
]

labels = []
f1_scores = []
accuracies = []

for label, filename in files:
    data = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
    metrics = data["holdout_metrics"]
    labels.append(label)
    f1_scores.append(metrics["f1"])
    accuracies.append(metrics["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, values, title in [
    (axes[0], f1_scores, "Holdout F1 Score"),
    (axes[1], accuracies, "Holdout Accuracy"),
]:
    bars = ax.bar(
        labels,
        values,
        color=["#94A3B8", "#60A5FA", "#2563EB"],
        width=0.58,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.4f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

fig.suptitle(
    "IEEE-CIS Fraud Detection: Baseline Comparison",
    fontsize=16,
    fontweight="bold",
)

fig.text(
    0.5, 0.01,
    "12,000 records | seed=42 | 20% holdout | "
    "Part 1 is a controlled task reproduction; model/features differ",
    ha="center",
    fontsize=9,
    color="#555555",
)

fig.tight_layout(rect=[0, 0.06, 1, 0.94])

path = OUTPUT / "baseline_comparison.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved:", path)
print("F1:", [round(x, 4) for x in f1_scores])
print("Accuracy:", [round(x, 4) for x in accuracies])

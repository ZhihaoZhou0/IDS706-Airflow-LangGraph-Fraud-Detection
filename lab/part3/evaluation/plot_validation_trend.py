"""Plot validation F1 across model-selection cycles."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("lab/part3/results")
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

part2 = json.loads(
    (ROOT / "part2_baseline.json").read_text(encoding="utf-8")
)
part3 = json.loads(
    (ROOT / "part3_baseline.json").read_text(encoding="utf-8")
)

h2 = part2["history"]
h3 = part3["history"]

assert len(h2) == len(h3), "Cycle counts differ"
assert all(
    a["model_family"] == b["model_family"]
    for a, b in zip(h2, h3)
), "Model sequences differ"

cycles = list(range(1, len(h2) + 1))
f1_part2 = [x["f1"] for x in h2]
f1_part3 = [x["f1"] for x in h3]

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(
    cycles, f1_part2,
    marker="o", linewidth=2.5, markersize=9,
    label="Part 2: Simple LangGraph",
    color="#60A5FA",
)

ax.plot(
    cycles, f1_part3,
    marker="s", linewidth=2.5, markersize=9,
    label="Part 3: Multi-Agent",
    color="#2563EB",
)

for values, offset in [
    (f1_part2, -17),
    (f1_part3, 12),
]:
    for x, y in zip(cycles, values):
        ax.annotate(
            f"{y:.4f}",
            (x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

ax.axhline(
    0.30,
    color="#16A34A",
    linestyle="--",
    linewidth=1.5,
    label="Target F1 = 0.30",
)

ax.set_xticks(cycles)
ax.set_xticklabels([
    "Cycle 1\nLogistic Regression",
    "Cycle 2\nRandom Forest",
    "Cycle 3\nHistGradientBoosting",
])

ax.set_ylim(0, 0.45)
ax.set_ylabel("Validation F1 Score")
ax.set_title(
    "Validation F1 Across Adaptive Model Selection Cycles",
    fontsize=14,
    fontweight="bold",
)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()

path = OUTPUT / "validation_f1_trend.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved:", path)
print("Part 2:", [round(v, 4) for v in f1_part2])
print("Part 3:", [round(v, 4) for v in f1_part3])

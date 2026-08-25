import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/jkchoi/project/autopaper/sandbox/paper-figures"

# order top-to-bottom as requested: Pythia family first, then Qwen family
# (plotted bottom-to-top, so reverse for barh/errorbar y positions)
rows = [
    ("Pythia-410M",   0.0565, 0.0429, 0.0701, False),  # full_layer_sweep=False -> tested-layer only
    ("Pythia-1.0B",   0.0317, 0.0230, 0.0404, True),
    ("Pythia-1.4B",   0.0452, 0.0370, 0.0540, True),
    ("Pythia-2.8B",   0.0087, 0.0050, 0.0124, False),
    ("Qwen2.5-1.5B",  0.1283, 0.1230, 0.1337, True),
    ("Qwen3-1.7B",    0.0306, 0.0211, 0.0401, True),
    ("Qwen3-4B",      0.0740, 0.0704, 0.0778, True),
]
rows = rows[::-1]  # reverse so Pythia-410M ends up at the top of the plot

labels = [r[0] for r in rows]
means = np.array([r[1] for r in rows])
ci_lo = np.array([r[2] for r in rows])
ci_hi = np.array([r[3] for r in rows])
full_sweep = np.array([r[4] for r in rows])
err_lo = means - ci_lo
err_hi = ci_hi - means
ROW_SPACING = 0.62
y = np.arange(len(rows)) * ROW_SPACING

fig, ax = plt.subplots(figsize=(7.5, 3.4))

ax.axvline(0, color="gray", linewidth=1.1, linestyle="--", zorder=1)

# full-sweep models: filled marker; tested-layer-only models: open marker
for i in range(len(rows)):
    color = "#1f4e8c" if full_sweep[i] else "#c46a1f"
    facecolor = color if full_sweep[i] else "white"
    ax.errorbar(means[i], y[i], xerr=[[err_lo[i]], [err_hi[i]]], fmt="o", color=color,
                ecolor=color, elinewidth=1.6, capsize=4, markersize=8,
                markerfacecolor=facecolor, markeredgewidth=1.6, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Causal steering margin (semantic direction − matched random-direction control)")
ax.set_ylim(-ROW_SPACING * 0.7, (len(rows) - 1) * ROW_SPACING + ROW_SPACING * 1.7)

# legend distinguishing full layer sweep vs tested-layer-only
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="#1f4e8c", markerfacecolor="#1f4e8c", markeredgewidth=1.6,
           linestyle="", markersize=8, label="full steering-specific layer sweep performed"),
    Line2D([0], [0], marker="o", color="#c46a1f", markerfacecolor="white", markeredgewidth=1.6,
           linestyle="", markersize=8, label="passive-best layer tested only (no full sweep)"),
]
leg = ax.legend(handles=legend_elems, loc="upper right", fontsize=8.5,
                 frameon=True, framealpha=1, edgecolor="black")
leg.get_frame().set_linewidth(0.8)

fig.tight_layout()
fig.savefig(f"{OUT}/figure_5_6_steering_margin_forest.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{OUT}/figure_5_6_steering_margin_forest.pdf", bbox_inches="tight")
print("saved figure_5_6_steering_margin_forest.png/.pdf")

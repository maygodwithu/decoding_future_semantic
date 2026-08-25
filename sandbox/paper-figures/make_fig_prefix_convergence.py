"""
Appendix B.3 figure: prefix-to-full semantic convergence.
Uses ONLY existing values already saved in P2's p2_results.json
(sandbox/p2-horizon-rollout-encoder/artifacts/p2_results.json) -- no
recomputation, no re-embedding, no new data.

rollout_cosine(H, m, E) in p2_results.json IS cos(E(y_1:m), E(y_1:H)) by
construction (P2 section 5's rollout score definition), so it is used
directly as the y-axis value here.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P2_RESULTS = "/home/jkchoi/project/autopaper/sandbox/p2-horizon-rollout-encoder/artifacts/p2_results.json"
OUT = "/home/jkchoi/project/autopaper/sandbox/paper-figures/figures/prefix_full_convergence.png"

results = json.load(open(P2_RESULTS))
ENCODERS = [("MiniLM", "MiniLM"), ("BGE", "BGE-base"), ("E5", "E5-base")]
HORIZONS = [16, 48, 96, 192, 256]
ROLLOUT_M = [3, 5, 10, 20]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=False)

colors = plt.cm.viridis_r([0.0, 0.22, 0.45, 0.68, 0.9])

for ax, (key, label) in zip(axes, ENCODERS):
    for H, color in zip(HORIZONS, colors):
        d = results["encoders"][key]["by_horizon"][str(H)]
        ys = [d["rollout"][str(m)]["rollout_cosine"] for m in ROLLOUT_M]
        ax.plot(ROLLOUT_M, ys, marker="o", markersize=5, linewidth=1.6, color=color, label=f"H={H}")
    ax.set_title(label)
    ax.set_xlabel(r"Rollout prefix length $m$")
    ax.set_xticks(ROLLOUT_M)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.6)

axes[0].set_ylabel(r"$\cos(E(y_{1:m}), E(y_{1:H}))$")
axes[-1].legend(loc="lower right", fontsize=8.5, frameon=True, edgecolor="black", framealpha=1)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", OUT)

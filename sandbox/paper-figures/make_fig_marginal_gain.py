"""
Appendix B.4 figure: diminishing marginal semantic return of additional
rollout computation. Uses ONLY existing values already saved in P2's
p2_results.json (marginal_gain_per_token, computed by run_p2.py exactly per
the formula G(H,m,E) = (S_rollout(H,m,E) - S_rollout(H,m_prev,E)) / (m-m_prev))
-- no recomputation, no new metrics.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P2_RESULTS = "/home/jkchoi/project/autopaper/sandbox/p2-horizon-rollout-encoder/artifacts/p2_results.json"
OUT = "/home/jkchoi/project/autopaper/sandbox/paper-figures/figures/rollout_marginal_gain.png"

results = json.load(open(P2_RESULTS))
ENCODERS = [("MiniLM", "(a) MiniLM"), ("BGE", "(b) BGE-base"), ("E5", "(c) E5-base")]
HORIZONS = [16, 48, 96, 192, 256]
INTERVALS = ["3->5", "5->10", "10->20"]
INTERVAL_LABELS = [r"3$\rightarrow$5", r"5$\rightarrow$10", r"10$\rightarrow$20"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=False)
colors = plt.cm.viridis_r([0.0, 0.22, 0.45, 0.68, 0.9])
x = range(len(INTERVALS))

for ax, (key, label) in zip(axes, ENCODERS):
    for H, color in zip(HORIZONS, colors):
        g = results["encoders"][key]["by_horizon"][str(H)]["marginal_gain_per_token"]
        ys = [g[iv] for iv in INTERVALS]
        ax.plot(x, ys, marker="o", markersize=5, linewidth=1.6, color=color, label=f"H={H}")
    ax.set_title(label)
    ax.set_xlabel("Rollout interval")
    ax.set_xticks(list(x))
    ax.set_xticklabels(INTERVAL_LABELS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linewidth=0.6)

axes[0].set_ylabel("Marginal cosine gain per additional rollout token")
axes[-1].legend(loc="upper right", fontsize=8.5, frameon=True, edgecolor="black", framealpha=1)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", OUT)

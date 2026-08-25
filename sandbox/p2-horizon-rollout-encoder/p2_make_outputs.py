import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = "/home/jkchoi/project/autopaper/sandbox/p2-horizon-rollout-encoder/artifacts"
results = json.load(open(f"{ART}/p2_results.json"))
ENCODERS = ["MiniLM", "BGE", "E5"]
HORIZONS = [16, 48, 96, 192, 256]
ROLLOUT_M = [3, 5, 10, 20]

# ---- Table A: main result by encoder ----
with open(f"{ART}/table_A_main_result.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Encoder", "H", "Probe", "Rollout-3", "Rollout-5", "Rollout-10", "Rollout-20", "Best layer"])
    for e in ENCODERS:
        for H in HORIZONS:
            d = results["encoders"][e]["by_horizon"][str(H)]
            w.writerow([e, H, f"{d['probe_cosine']:.4f}"] +
                       [f"{d['rollout'][str(m)]['rollout_cosine']:.4f}" for m in ROLLOUT_M] +
                       [d["best_layer"]])

# ---- Table B: probe-minus-rollout margins (with CI) ----
with open(f"{ART}/table_B_margins.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Encoder", "H", "m", "Delta", "CI low", "CI high", "Excludes 0?"])
    for e in ENCODERS:
        for H in HORIZONS:
            d = results["encoders"][e]["by_horizon"][str(H)]
            for m in ROLLOUT_M:
                delta = d["rollout"][str(m)]["delta"]
                w.writerow([e, H, m, f"{delta['point']:.4f}", f"{delta['ci_low']:.4f}",
                            f"{delta['ci_high']:.4f}", "Yes" if delta["excludes_zero"] else "No"])

# ---- Table C: rollout parity budget ----
with open(f"{ART}/table_C_parity_budget.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Encoder", "H", "Smallest tested m matching/beating probe"])
    for e in ENCODERS:
        for H in HORIZONS:
            d = results["encoders"][e]["by_horizon"][str(H)]
            w.writerow([e, H, d["rollout_parity_budget"]])

# ---- Figure 1: Delta(H,m,E) vs H, one line per m, per encoder ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
for ax, e in zip(axes, ENCODERS):
    for m in ROLLOUT_M:
        ys = [results["encoders"][e]["by_horizon"][str(H)]["rollout"][str(m)]["delta"]["point"] for H in HORIZONS]
        ax.plot(HORIZONS, ys, marker="o", label=f"m={m}")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title(e)
    ax.set_xlabel("Horizon H (tokens)")
    if e == ENCODERS[0]:
        ax.set_ylabel("Δ = probe − rollout cosine")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{ART}/figure1_delta_vs_horizon.png", dpi=150)
plt.close(fig)

# ---- Figure 2: prefix-to-full convergence C(H,m,E) vs m, one line per H, per encoder ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, e in zip(axes, ENCODERS):
    for H in HORIZONS:
        d = results["encoders"][e]["by_horizon"][str(H)]
        ys = [d["rollout"][str(m)]["rollout_cosine"] for m in ROLLOUT_M]
        ax.plot(ROLLOUT_M, ys, marker="o", label=f"H={H}")
        ax.axhline(d["probe_cosine"], color="gray", linewidth=0.6, linestyle=":")
    ax.set_title(e)
    ax.set_xlabel("Rollout length m (tokens)")
    if e == ENCODERS[0]:
        ax.set_ylabel("cos(prefix embedding, full-H embedding)\n(= rollout score; dotted = probe cosine per H)")
    ax.legend(fontsize=7)
fig.suptitle("Figure 2: semantic prefix-to-full convergence (same values as rollout score)")
fig.tight_layout()
fig.savefig(f"{ART}/figure2_prefix_convergence.png", dpi=150)
plt.close(fig)

# ---- Figure 3: compute trade-off (score vs. extra decode steps), per encoder, H=256 as representative ----
H_rep = 256
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
for ax, e in zip(axes, ENCODERS):
    d = results["encoders"][e]["by_horizon"][str(H_rep)]
    ax.scatter([0], [d["probe_cosine"]], color="crimson", zorder=5, label="probe (0 extra decode steps)", s=60)
    xs = ROLLOUT_M
    ys = [d["rollout"][str(m)]["rollout_cosine"] for m in xs]
    ax.plot(xs, ys, marker="o", color="steelblue", label="rollout")
    ax.set_title(f"{e} (H={H_rep})")
    ax.set_xlabel("Extra autoregressive decode steps")
    if e == ENCODERS[0]:
        ax.set_ylabel("Cosine similarity to full-H target")
    ax.legend(fontsize=8)
fig.suptitle(f"Figure 3: compute trade-off at H={H_rep} (probe requires zero additional decode steps)")
fig.tight_layout()
fig.savefig(f"{ART}/figure3_compute_tradeoff.png", dpi=150)
plt.close(fig)

print("saved Table A/B/C and Figure 1/2/3")

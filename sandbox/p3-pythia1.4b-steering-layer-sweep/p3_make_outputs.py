import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ART = "/home/jkchoi/project/autopaper/sandbox/p3-pythia1.4b-steering-layer-sweep/artifacts"
results = json.load(open(f"{ART}/p3_results.json"))
passive = {d["layer"]: d["probe_test_cosine"] for d in results["passive_table"]}
steer = {d["layer"]: d for d in results["steering_summary"]}
N_LAYERS = 24

# paired diff layer20 vs layer24 (both strong; best==20 but 24 close behind)
rows = list(csv.DictReader(open(f"{ART}/per_seed_margins.csv")))
m20 = {int(r["seed"]): float(r["margin"]) for r in rows if int(r["layer"]) == 20}
m24 = {int(r["seed"]): float(r["margin"]) for r in rows if int(r["layer"]) == 24}
seeds = sorted(m20.keys())
diffs = np.array([m20[s] - m24[s] for s in seeds])
d_mean = float(diffs.mean())
d_sem = float(diffs.std(ddof=1) / math.sqrt(len(diffs)))
tcrit = float(stats.t.ppf(0.975, df=len(diffs) - 1))
paired_20_24 = {"mean": d_mean, "ci95_low": d_mean - tcrit * d_sem, "ci95_high": d_mean + tcrit * d_sem}
paired_20_24["excludes_zero"] = paired_20_24["ci95_low"] > 0 or paired_20_24["ci95_high"] < 0

# ---- Table: passive probe reference ----
with open(f"{ART}/table_passive_probe.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Layer", "Passive probe cosine", "Normalized depth"])
    for L in sorted(passive):
        w.writerow([L, f"{passive[L]:.4f}", f"{L/N_LAYERS:.3f}"])

# ---- Table: steering summary ----
with open(f"{ART}/table_steering_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Layer", "Best k", "Mean margin", "Std", "95% CI low", "95% CI high", "CI excludes 0?", "Normalized depth"])
    for L in sorted(steer):
        s = steer[L]
        w.writerow([L, s["selected_k"], f"{s['mean_margin']:.5f}", f"{s['std_margin']:.5f}",
                    f"{s['ci95_low']:.5f}", f"{s['ci95_high']:.5f}", "Yes" if s["ci_excludes_zero"] else "No",
                    f"{L/N_LAYERS:.3f}"])

# ---- Table: depth-normalized combined (passive layer matched to its steer layer) ----
with open(f"{ART}/table_depth_normalized.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Steer layer", "Normalized depth", "Passive probe cosine (matched source layer)", "Steering margin"])
    p2s = results.get("passive_to_steer_map") or {"0": 1, "4": 4, "8": 8, "12": 12, "16": 16, "20": 20, "24": 24}
    for L_passive_str, L_steer in p2s.items():
        L_passive = int(L_passive_str)
        w.writerow([L_steer, f"{L_steer/N_LAYERS:.3f}", f"{passive[L_passive]:.4f}", f"{steer[L_steer]['mean_margin']:.5f}"])

# ---- Table: correlation ----
with open(f"{ART}/table_correlation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Statistic", "Value"])
    w.writerow(["Pearson r", f"{results['correlation']['pearson_r']:.4f}"])
    w.writerow(["Pearson p", f"{results['correlation']['pearson_p']:.4f}"])
    w.writerow(["Spearman rho", f"{results['correlation']['spearman_rho']:.4f}"])
    w.writerow(["Spearman p", f"{results['correlation']['spearman_p']:.4f}"])
    w.writerow(["n layers", results["correlation"]["n_layers"]])

with open(f"{ART}/paired_diff_layer20_vs_24.json", "w") as f:
    json.dump(paired_20_24, f, indent=2)

# ---- Figure 1: passive vs causal profile across depth (two aligned panels) ----
layers_sorted = sorted(passive)
steer_layers_sorted = sorted(steer)
fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
axes[0].plot(layers_sorted, [passive[L] for L in layers_sorted], marker="o", color="darkorange")
axes[0].axvline(20, color="gray", linestyle=":", linewidth=1)
axes[0].set_ylabel("Passive probe cosine")
axes[0].set_title("Passive semantic decodability by layer")

means = [steer[L]["mean_margin"] for L in steer_layers_sorted]
los = [steer[L]["mean_margin"] - steer[L]["ci95_low"] for L in steer_layers_sorted]
his = [steer[L]["ci95_high"] - steer[L]["mean_margin"] for L in steer_layers_sorted]
axes[1].errorbar(steer_layers_sorted, means, yerr=[los, his], marker="o", color="steelblue", capsize=4)
axes[1].axhline(0, color="gray", linewidth=1, linestyle="--")
axes[1].axvline(20, color="gray", linestyle=":", linewidth=1)
axes[1].set_ylabel("Steering margin (95% CI)")
axes[1].set_xlabel("Layer (hidden_states index; steer layer 1 substitutes passive layer 0)")
axes[1].set_title("Causal steering margin by layer")
fig.tight_layout()
fig.savefig(f"{ART}/figure1_passive_vs_causal.png", dpi=150)
plt.close(fig)

# ---- Figure 2: steering margin by layer (main P3 figure) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["crimson" if L == 20 else "steelblue" for L in steer_layers_sorted]
ax.errorbar(steer_layers_sorted, means, yerr=[los, his], fmt="o", capsize=5, ecolor="gray", color="black", zorder=3)
ax.bar(steer_layers_sorted, means, width=1.2, color=colors, alpha=0.5, zorder=1)
ax.axhline(0, color="gray", linewidth=1, linestyle="--")
ax.axvline(20, color="crimson", linestyle=":", linewidth=1.2, label="layer 20 (passive-best)")
ax.set_xlabel("Layer (hidden_states index)")
ax.set_ylabel("Mean steering margin")
ax.set_title("Figure 2: causal steering margin by layer, Pythia-1.4B (95% CI, 5 seeds)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{ART}/figure2_steering_by_layer.png", dpi=150)
plt.close(fig)

# ---- Figure 3: k pilot curves ----
k_rows = list(csv.DictReader(open(f"{ART}/k_sweep_results.csv")))
by_layer_k = {}
for r in k_rows:
    by_layer_k.setdefault(int(r["layer"]), []).append((float(r["k"]), float(r["pilot_margin_mean"])))
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for ax, L in zip(axes, steer_layers_sorted):
    pts = sorted(by_layer_k[L])
    ks = [p[0] for p in pts]
    ms = [p[1] for p in pts]
    ax.plot(ks, ms, marker="o")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_title(f"layer {L}")
    ax.set_xlabel("k")
    ax.set_ylabel("pilot margin")
axes[-1].axis("off")
fig.suptitle("Figure 3: per-layer k pilot curves (dense grid, pilot set only)")
fig.tight_layout()
fig.savefig(f"{ART}/figure3_k_pilot_curves.png", dpi=150)
plt.close(fig)

print("saved all P3 tables and figures")
print("paired diff layer20-layer24:", paired_20_24)

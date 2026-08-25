import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/jkchoi/project/autopaper/sandbox/semantic-target-robustness/artifacts"
results = json.load(open(f"{OUT}/p1_results.json"))
encoders = ["E1_MiniLM", "E2_BGE", "E3_E5"]
labels = {"E1_MiniLM": "MiniLM (E1, anchor)", "E2_BGE": "BGE-base (E2)", "E3_E5": "E5-base (E3)"}

# ---- A. summary table ----
with open(f"{OUT}/summary_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Encoder", "Best layer", "Probe", "Mean", "Random", "TF-IDF", "Best logit-lens",
                "Rollout-3", "Rollout-5", "R@1", "R@5", "Mean rank"])
    for e in encoders:
        r = results[e]
        w.writerow([e, r["best_layer"], f"{r['probe_cosine']:.4f}", f"{r['mean_cosine']:.4f}",
                    f"{r['random_cosine']:.4f}", f"{r['lexical_cosine']:.4f}",
                    f"{r['best_logitlens_cosine']:.4f} ({r['best_logitlens_variant']})",
                    f"{r['rollout_m3_cosine']:.4f}", f"{r['rollout_m5_cosine']:.4f}",
                    f"{r['retrieval']['recall_at_1']:.4f}", f"{r['retrieval']['recall_at_5']:.4f}",
                    f"{r['retrieval']['mean_rank']:.2f}"])

# layer-sweep table
with open(f"{OUT}/layer_sweep_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Encoder", "L4", "L8", "L12", "L20", "Best layer", "Best cosine"])
    for e in encoders:
        ls = results[e]["layer_sweep"]
        w.writerow([e, f"{ls['4']['test_cosine']:.4f}", f"{ls['8']['test_cosine']:.4f}",
                    f"{ls['12']['test_cosine']:.4f}", f"{ls['20']['test_cosine']:.4f}",
                    results[e]["best_layer"], f"{results[e]['probe_cosine']:.4f}"])

# ---- B. shuffled-target table ----
with open(f"{OUT}/shuffled_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Encoder", "Normal", "Shuffled", "Difference", "Perm seed"])
    for e in encoders:
        s = results[e]["shuffled_target"]
        w.writerow([e, f"{s['normal_cosine']:.4f}", f"{s['shuffled_cosine']:.4f}",
                    f"{s['normal_cosine'] - s['shuffled_cosine']:.4f}", s["perm_seed"]])

# ---- C. bootstrap table ----
with open(f"{OUT}/bootstrap_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Encoder", "Quantity", "Point", "95% CI low", "95% CI high", "Excludes 0?"])
    for e in encoders:
        for qname, q in results[e]["bootstrap"].items():
            w.writerow([e, qname, f"{q['point']:.4f}", f"{q['ci_low']:.4f}", f"{q['ci_high']:.4f}",
                        "Yes" if q["excludes_zero"] else "No"])

# ---- D. plot ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(encoders))
width = 0.2
metrics = [
    ("Probe", "probe_cosine", None),
    ("Best weak/token baseline", None, "best_weak"),
    ("Rollout m=3", "rollout_m3_cosine", None),
    ("Rollout m=5", "rollout_m5_cosine", None),
]
best_weak_vals = []
for e in encoders:
    r = results[e]
    best_weak_vals.append(max(r["mean_cosine"], r["random_cosine"], r["lexical_cosine"], r["best_logitlens_cosine"]))

series = {
    "Probe": [results[e]["probe_cosine"] for e in encoders],
    "Best weak/token baseline": best_weak_vals,
    "Rollout m=3": [results[e]["rollout_m3_cosine"] for e in encoders],
    "Rollout m=5": [results[e]["rollout_m5_cosine"] for e in encoders],
}
for i, (name, vals) in enumerate(series.items()):
    ax.bar(x + (i - 1.5) * width, vals, width, label=name)

ax.set_xticks(x)
ax.set_xticklabels([labels[e] for e in encoders])
ax.set_ylabel("Cosine similarity (within-encoder scale only)")
ax.set_title("P1: probe vs. baselines, by semantic target encoder\n(raw cosine not comparable across encoders)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/p1_encoder_comparison.png", dpi=150)
print("saved plot and tables")

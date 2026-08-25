import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = "/home/jkchoi/project/autopaper/sandbox/p4-future-decoding-comparison/artifacts"
r = json.load(open(f"{ART}/p4_results.json"))

# ---- Table: main comparison at H=16 ----
rows = [
    ("Mean embedding", "--", 0, r["mean_embedding_baseline_H16"]),
    ("Direct Logit Lens", "next-token identity", 0, r["M1_direct_logit_lens"]["test_cosine"]),
    ("Tuned Lens (layer 20)", "translated next-token identity", 0, r["M2_tuned_lens"]["20"]["test_cosine"]),
    ("Future-Lens-style, m=3", "future token identities", 0, r["M3_future_lens_semantic"]["3"]["test_cosine"]),
    ("Future-Lens-style, m=5", "future token identities", 0, r["M3_future_lens_semantic"]["5"]["test_cosine"]),
    ("Future-Lens-style, m=10", "future token identities", 0, r["M3_future_lens_semantic"]["10"]["test_cosine"]),
    ("Linear Semantic Probe", "continuation semantics", 0, r["M6_linear_probe_L20_H16"]["test_cosine"]),
    ("MLP Semantic Probe (diagnostic)", "continuation semantics", 0, r["M8_mlp_probe_L20_H16"]["test_cosine"]),
    ("Rollout-3", "generated prefix", 3, r["rollout_H16"]["3"]),
    ("Rollout-5", "generated prefix", 5, r["rollout_H16"]["5"]),
    ("Rollout-10", "generated prefix", 10, r["rollout_H16"]["10"]),
]
with open(f"{ART}/table_p4_main_comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Method", "Representation decoded", "Extra AR steps", "Semantic cosine (H=16)"])
    for row in rows:
        w.writerow([row[0], row[1], row[2], f"{row[3]:.4f}"])

# ---- Table: Tuned Lens by layer ----
with open(f"{ART}/table_p4_tuned_lens_by_layer.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Layer", "Best variant", "Val cosine", "Test cosine", "Next-token top1 acc (test)"])
    for L in ["4", "8", "12", "20"]:
        d = r["M2_tuned_lens"][L]
        w.writerow([L, d["best_variant"], f"{d['val_cosine']:.4f}", f"{d['test_cosine']:.4f}", f"{d['next_token_top1_acc_test']:.4f}"])

# ---- Table: Future Lens native validation ----
with open(f"{ART}/table_p4_future_lens_native.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Future offset", "Top-1 accuracy", "Top-5 accuracy"])
    for j in ["1", "2", "3", "5", "10"]:
        d = r["M3_future_lens_native"][j]
        w.writerow([j, f"{d['top1_acc']:.4f}", f"{d['top5_acc']:.4f}"])

# ---- Table: bootstrap ----
with open(f"{ART}/table_p4_bootstrap.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Comparison", "Mean difference (probe - other)", "95% CI low", "95% CI high", "Excludes 0?"])
    labels = {
        "probe_minus_tunedlens_L20": "Probe - Tuned Lens (layer 20)",
        "probe_minus_futurelens_m3": "Probe - Future-Lens-style m=3",
        "probe_minus_futurelens_m5": "Probe - Future-Lens-style m=5",
        "probe_minus_futurelens_m10": "Probe - Future-Lens-style m=10",
        "probe_minus_rollout_m3": "Probe - Rollout m=3",
        "probe_minus_rollout_m5": "Probe - Rollout m=5",
        "probe_minus_rollout_m10": "Probe - Rollout m=10",
    }
    for key, label in labels.items():
        d = r["bootstrap"][key]
        w.writerow([label, f"{d['point']:.4f}", f"{d['ci_low']:.4f}", f"{d['ci_high']:.4f}", "Yes" if d["excludes_zero"] else "No"])

# ---- Table: horizon comparison ----
with open(f"{ART}/table_p4_horizon_comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["H", "Semantic Probe", "Tuned Lens", "Future-Lens-3", "Future-Lens-5", "Future-Lens-10", "Rollout-3", "Rollout-5", "Rollout-10"])
    for H in ["16", "48", "96"]:
        d = r["horizon_table"][H]
        w.writerow([H, f"{d['probe']:.4f}", f"{d['tuned_lens']:.4f}",
                    f"{d['future_lens_m3']:.4f}", f"{d['future_lens_m5']:.4f}", f"{d['future_lens_m10']:.4f}",
                    f"{d['rollout_m3']:.4f}", f"{d['rollout_m5']:.4f}", f"{d['rollout_m10']:.4f}"])

# ---- Figure: zero-decode semantic comparison at H=16 ----
methods = ["Direct\nLogit Lens", "Tuned\nLens", "Future-Lens\nm=3", "Future-Lens\nm=5", "Future-Lens\nm=10", "Linear\nSemantic Probe"]
values = [r["M1_direct_logit_lens"]["test_cosine"], r["M2_tuned_lens"]["20"]["test_cosine"],
          r["M3_future_lens_semantic"]["3"]["test_cosine"], r["M3_future_lens_semantic"]["5"]["test_cosine"],
          r["M3_future_lens_semantic"]["10"]["test_cosine"], r["M6_linear_probe_L20_H16"]["test_cosine"]]
colors = ["#c46a1f"] * 5 + ["#1f4e8c"]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar(methods, values, color=colors, width=0.6)
ax.axhline(r["mean_embedding_baseline_H16"], color="gray", linestyle="--", linewidth=1, label="mean-embedding baseline")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("Semantic cosine (H=16, test)")
ax.set_ylim(0, max(values) * 1.2)
ax.legend(loc="upper left", fontsize=9, frameon=True)
fig.tight_layout()
fig.savefig(f"{ART}/figure_p4_zero_decode_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("saved all P4 tables and figure")

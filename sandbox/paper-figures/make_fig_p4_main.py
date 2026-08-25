import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/jkchoi/project/autopaper/sandbox/paper-figures/figures/future_decoding_comparison.png"

MEAN_EMBEDDING = 0.243

# (label, value, group)  group: 0=token-oriented, 1=direct semantic, 2=rollout
bars = [
    ("Direct\nLogit Lens", 0.180, 0),
    ("Tuned-Lens\nstyle", 0.144, 0),
    ("Future-Lens\nm=3", 0.218, 0),
    ("Future-Lens\nm=5", 0.229, 0),
    ("Future-Lens\nm=10", 0.253, 0),
    ("Linear\nSemantic Probe", 0.450, 1),
    ("Rollout-3", 0.406, 2),
    ("Rollout-5", 0.583, 2),
    ("Rollout-10", 0.835, 2),
]

GROUP_COLOR = {0: "#b0793a", 1: "#1f4e8c", 2: "#3f8f6f"}
GROUP_NAME = {0: "Token-oriented latent decoding", 1: "Direct semantic\nreadout", 2: "Behavioral rollout\n(additional AR steps)"}

UNIT = 1.35
GAP = 1.15
x = []
cur = 0.0
prev_group = None
group_spans = {}
for label, value, group in bars:
    if prev_group is not None and group != prev_group:
        cur += GAP
    x.append(cur)
    group_spans.setdefault(group, []).append(cur)
    cur += UNIT
    prev_group = group

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(9.2, 4.6))

for (label, value, group), xi in zip(bars, x):
    is_probe = group == 1
    ax.bar(xi, value, width=1.0,
           color=GROUP_COLOR[group],
           edgecolor="black" if is_probe else "none",
           linewidth=1.6 if is_probe else 0,
           zorder=3)
    ax.text(xi, value + 0.015, f"{value:.3f}", ha="center", va="bottom",
            fontsize=10.5 if not is_probe else 11.5,
            fontweight="bold" if is_probe else "normal", zorder=4)

# mean-embedding baseline as a horizontal dashed line
ax.axhline(MEAN_EMBEDDING, color="gray", linestyle="--", linewidth=1.2, zorder=2)
ax.text(x[0] - 0.55, MEAN_EMBEDDING + 0.018, f"mean embedding ({MEAN_EMBEDDING:.3f})",
        ha="left", va="bottom", fontsize=9.5, color="dimgray")

# vertical separators between families
for g in (0, 1):
    sep_x = (max(group_spans[g]) + min(group_spans[g + 1])) / 2
    ax.axvline(sep_x, color="lightgray", linewidth=1.0, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars], fontsize=8.8)
ax.set_ylabel("Semantic cosine (H=16, test)", fontsize=11.5)
ax.set_ylim(0, 0.95)
ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)

# family labels beneath the x-axis
for g, xs in group_spans.items():
    ax.text(np.mean(xs), -0.135, GROUP_NAME[g], ha="center", va="top",
            fontsize=9.3, color="black", transform=ax.get_xaxis_transform())

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", length=0)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", OUT)

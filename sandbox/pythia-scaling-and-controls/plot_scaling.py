"""
Two scaling plots from artifacts/scaling_summary.csv:
  artifacts/scaling_probe.png    (best_test_cosine & mean_baseline_cosine vs params)
  artifacts/scaling_steering.png (steering_effect_semantic vs steering_effect_random vs params)
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a2e"
BLUE = "#3b6dc7"
GREEN = "#2f9e6e"
GRAY = "#8a8fa3"
RED = "#c75b3b"


def main():
    rows = list(csv.DictReader(open("artifacts/scaling_summary.csv")))
    rows.sort(key=lambda r: int(r["params"]))
    params = [int(r["params"]) / 1e6 for r in rows]  # millions
    labels = [r["model"] for r in rows]
    best_cos = [float(r["best_test_cosine"]) for r in rows]
    mean_base = [float(r["mean_baseline_cosine"]) for r in rows]
    rand_match = [float(r["random_match_cosine"]) for r in rows]
    recall5 = [float(r["recall_at_5"]) if r["recall_at_5"] not in ("", None) else None for r in rows]

    xs = list(range(len(rows)))  # evenly-spaced categorical positions (params are irregular; log-scale
                                  # auto-ticks previously collided visually with the custom size labels)
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax1.plot(xs, best_cos, marker="o", color=BLUE, linewidth=2, label="Semantic probe (best layer) test cosine")
    ax1.plot(xs, mean_base, marker="s", color=GRAY, linewidth=2, linestyle="--", label="Mean-embedding baseline")
    ax1.plot(xs, rand_match, marker="^", color=RED, linewidth=1.5, linestyle=":", label="Random-match baseline")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{l}\n({p:.0f}M params)" for l, p in zip(labels, params)])
    ax1.set_xlabel("Model size")
    ax1.set_ylabel("Test cosine similarity")
    ax1.set_title("Semantic probe recoverability vs. Pythia model scale")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    ax2 = ax1.twinx()
    if all(r is not None for r in recall5):
        ax2.plot(xs, recall5, marker="d", color=GREEN, linewidth=1.5, alpha=0.7, label="Recall@5")
        ax2.set_ylabel("Recall@5", color=GREEN)
        ax2.tick_params(axis="y", labelcolor=GREEN)

    fig.tight_layout()
    fig.savefig("artifacts/scaling_probe.png")
    print("[plot] saved artifacts/scaling_probe.png")

    fig2, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    eff_sem = [float(r["steering_effect_semantic"]) if r["steering_effect_semantic"] not in ("", None) else None for r in rows]
    eff_rand = [float(r["steering_effect_random"]) if r["steering_effect_random"] not in ("", None) else None for r in rows]
    width = 0.35
    xs = list(range(len(rows)))
    ax.bar([x - width / 2 for x in xs], eff_sem, width=width, color=BLUE, label="Probe-derived (semantic) direction")
    ax.bar([x + width / 2 for x in xs], eff_rand, width=width, color=GRAY, label="Matched-norm random direction")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean sentiment-axis shift vs. baseline")
    ax.set_title("Causal steering effect size vs. Pythia model scale")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    fig2.savefig("artifacts/scaling_steering.png")
    print("[plot] saved artifacts/scaling_steering.png")


if __name__ == "__main__":
    main()

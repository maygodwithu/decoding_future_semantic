"""
Step 4: for each horizon H, compute the short greedy-rollout control test
cosine for m in {3, 5, 10, 20} (embedding of the first m generated tokens vs
embedding of the true H-token continuation target, on the fixed test split),
combine with the probe test cosine from train_probes.py, compute
delta_best_rollout = probe_cosine - max(rollout_m3..m20) per horizon,
determine the crossover horizon (if any), add a back-of-the-envelope
compute-normalized comparison, save the summary CSV/JSON, and produce the
horizon-vs-cosine plot.

Outputs (artifacts/):
  summary.csv / summary.json
  horizon_crossover_plot.png
"""
import json

import numpy as np
import pandas as pd

from common import cosine_rows

HORIZONS = [16, 48, 96]
ROLLOUT_MS = [3, 5, 10, 20]


def main():
    split = json.load(open("artifacts/split.json"))
    idx_test = split["test"]
    n_test = len(idx_test)

    rollout_emb = {m: np.load(f"artifacts/rollout_embeddings_m{m}.npy") for m in ROLLOUT_MS}
    horizon_stats = json.load(open("artifacts/horizon_stats.json"))

    rows = []
    for H in HORIZONS:
        target_emb = np.load(f"artifacts/continuation_embeddings_H{H}.npy")
        y_test = target_emb[idx_test]

        probe_results = json.load(open(f"artifacts/probe_results_H{H}.json"))
        probe_cos = probe_results["best_layer_test_cosine"]
        probe_layer = probe_results["best_layer_by_val"]

        row = {
            "horizon": H,
            "realized_mean_len": horizon_stats[str(H)]["mean_used_len"],
            "frac_shorter_than_H": horizon_stats[str(H)]["frac_shorter_than_H_due_to_eos_or_short_gen"],
            "test_n": n_test,
            "probe_cosine": probe_cos,
            "probe_best_layer": probe_layer,
        }

        rollout_cos = {}
        for m in ROLLOUT_MS:
            emb_m_test = rollout_emb[m][idx_test]
            cos = float(np.mean(cosine_rows(emb_m_test, y_test)))
            rollout_cos[m] = cos
            row[f"rollout_m{m}"] = cos

        best_rollout_m = max(rollout_cos, key=rollout_cos.get)
        row["best_rollout"] = rollout_cos[best_rollout_m]
        row["best_rollout_m"] = best_rollout_m
        row["delta_best_rollout"] = probe_cos - row["best_rollout"]

        # ---- back-of-envelope compute-normalized comparison ----
        # probe: 1 prompt forward pass + negligible linear map -> 0 EXTRA decode steps
        # rollout-m: 1 prompt forward pass + m extra autoregressive decode steps
        row["probe_extra_decode_steps"] = 0
        for m in ROLLOUT_MS:
            row[f"rollout_m{m}_cosine_per_extra_decode_step"] = rollout_cos[m] / m
        row["best_rollout_cosine_per_extra_decode_step"] = rollout_cos[best_rollout_m] / best_rollout_m
        # probe cosine achieved at ~0 extra decode steps vs. best rollout's steps
        row["probe_vs_best_rollout_at_best_rollout_compute"] = probe_cos - row["best_rollout"]

        rows.append(row)
        print(f"[summary] H={H}: probe={probe_cos:.4f} (layer {probe_layer}) "
              f"rollouts={ {m: round(v,4) for m,v in rollout_cos.items()} } "
              f"best_rollout={row['best_rollout']:.4f} (m={best_rollout_m}) "
              f"delta={row['delta_best_rollout']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("artifacts/summary.csv", index=False)
    with open("artifacts/summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ---- crossover determination ----
    deltas = {r["horizon"]: r["delta_best_rollout"] for r in rows}
    short_h = min(HORIZONS)
    crossover_h = None
    for H in sorted(HORIZONS):
        if H == short_h:
            continue
        if deltas[H] > 0 and deltas[short_h] < 0:
            crossover_h = H
            break
    any_positive = any(d > 0 for d in deltas.values())
    crossover_exists = crossover_h is not None

    gap_narrowed = None
    hs_sorted = sorted(HORIZONS)
    if len(hs_sorted) >= 2:
        gap_narrowed = deltas[hs_sorted[0]] < deltas[hs_sorted[-1]]

    crossover_summary = {
        "deltas_by_horizon": deltas,
        "crossover_exists": crossover_exists,
        "crossover_horizon": crossover_h,
        "any_horizon_with_positive_delta": any_positive,
        "gap_narrowed_from_shortest_to_longest": gap_narrowed,
        "shortest_horizon": short_h,
        "longest_horizon": max(HORIZONS),
    }
    with open("artifacts/crossover_summary.json", "w") as f:
        json.dump(crossover_summary, f, indent=2)
    print(f"[summary] crossover_summary = {crossover_summary}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {
        "probe": "#4C72B0",
        3: "#DD8452", 5: "#55A868", 10: "#C44E52", 20: "#8172B2",
    }
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    hs = [r["horizon"] for r in rows]
    ax.plot(hs, [r["probe_cosine"] for r in rows], marker="o", linewidth=2.5,
            color=palette["probe"], label="Linear probe (last-prompt hidden state)", zorder=5)
    for m in ROLLOUT_MS:
        ax.plot(hs, [r[f"rollout_m{m}"] for r in rows], marker="s", linewidth=1.5,
                linestyle="--", color=palette[m], label=f"Greedy rollout m={m}")
    ax.set_xlabel("Target continuation horizon H (tokens)")
    ax.set_ylabel("Test cosine similarity to full H-token continuation embedding")
    ax.set_title("Probe vs. short-rollout controls across continuation horizons\n(EleutherAI/pythia-1.4b, greedy decoding)")
    ax.set_xticks(hs)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("artifacts/horizon_crossover_plot.png")
    print("[summary] saved artifacts/horizon_crossover_plot.png")


if __name__ == "__main__":
    main()

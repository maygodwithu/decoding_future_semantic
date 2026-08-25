"""
Aggregates per-model / per-condition artifacts into:
  artifacts/scaling_summary.csv
  artifacts/pythia-1.4b/decoding_comparison.csv
  artifacts/final_summary.json
"""
import csv
import json
import os

SCALING_MODELS = [
    ("EleutherAI/pythia-410m", "pythia-410m"),
    ("EleutherAI/pythia-1b", "pythia-1.0b"),
    ("EleutherAI/pythia-1.4b", "pythia-1.4b"),
    ("EleutherAI/pythia-2.8b", "pythia-2.8b"),
]


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def build_scaling_summary():
    rows = []
    for model_name, tag in SCALING_MODELS:
        out_dir = f"artifacts/{tag}/greedy"
        meta = load(f"{out_dir}/metadata.json")
        probe = load(f"{out_dir}/probe_results.json")
        retrieval = load(f"{out_dir}/retrieval_results.json")
        steer = load(f"{out_dir}/steering_summary.json")
        if meta is None or probe is None:
            print(f"[build_summaries] SKIP {tag}: missing artifacts")
            continue
        best_layer = probe["best_layer_by_val"]
        best_test_cos = probe["layers"][str(best_layer)]["test_cosine"]
        mean_baseline = probe["baselines"]["mean_embedding"]["test_cosine"]
        row = {
            "model": tag,
            "model_name": model_name,
            "params": meta["num_parameters"],
            "n_layers": meta["n_layers"],
            "best_layer": best_layer,
            "best_test_cosine": best_test_cos,
            "mean_baseline_cosine": mean_baseline,
            "cosine_gain_over_mean": best_test_cos - mean_baseline,
            "random_match_cosine": probe["baselines"]["random_match"]["test_cosine"],
            "recall_at_5": retrieval["probe"]["recall_at_5"] if retrieval else "",
            "chance_recall_at_5": retrieval.get("chance_recall_at_5", "") if retrieval else "",
            "steering_effect_semantic": steer["steering_effect_semantic"] if steer else "",
            "steering_effect_random": steer["steering_effect_random"] if steer else "",
            "steering_margin": steer["steering_margin"] if steer else "",
            "steering_alpha": steer["alpha"] if steer else "",
        }
        rows.append(row)

    fieldnames = ["model", "model_name", "params", "n_layers", "best_layer", "best_test_cosine",
                  "mean_baseline_cosine", "cosine_gain_over_mean", "random_match_cosine", "recall_at_5",
                  "chance_recall_at_5", "steering_effect_semantic", "steering_effect_random",
                  "steering_margin", "steering_alpha"]
    with open("artifacts/scaling_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[build_summaries] wrote artifacts/scaling_summary.csv ({len(rows)} models)")
    return rows


def build_decoding_comparison():
    rows = []
    for decode_tag in ["greedy", "sampled"]:
        out_dir = f"artifacts/pythia-1.4b/{decode_tag}"
        probe = load(f"{out_dir}/probe_results.json")
        retrieval = load(f"{out_dir}/retrieval_results.json")
        controls_path = f"{out_dir}/token_control_results.csv"
        controls = {}
        if os.path.exists(controls_path):
            with open(controls_path) as f:
                for r in csv.DictReader(f):
                    controls[r["method"]] = r
        if probe is None:
            print(f"[build_summaries] SKIP decoding_comparison for {decode_tag}: missing probe results")
            continue
        best_layer = probe["best_layer_by_val"]
        row = {
            "decode_tag": decode_tag,
            "best_layer": best_layer,
            "test_cosine": probe["layers"][str(best_layer)]["test_cosine"],
            "mean_baseline_cosine": probe["baselines"]["mean_embedding"]["test_cosine"],
            "recall_at_5": retrieval["probe"]["recall_at_5"] if retrieval else "",
        }
        for method in ["topk_concat_k10", "topk_weighted_k10", "short_rollout_m1", "short_rollout_m3", "short_rollout_m5"]:
            if method in controls:
                row[f"{method}_cosine"] = controls[method]["test_cosine"]
                row[f"{method}_margin"] = controls[method]["margin_vs_probe"]
                row[f"{method}_recall_at_5"] = controls[method]["recall_at_5"]
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        os.makedirs("artifacts/pythia-1.4b", exist_ok=True)
        with open("artifacts/pythia-1.4b/decoding_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("[build_summaries] wrote artifacts/pythia-1.4b/decoding_comparison.csv")
    return rows


def build_final_summary(scaling_rows, decoding_rows):
    p1_controls = {}
    controls_path = "artifacts/pythia-1.4b/greedy/token_control_results.csv"
    if os.path.exists(controls_path):
        with open(controls_path) as f:
            for r in csv.DictReader(f):
                p1_controls[r["method"]] = {"test_cosine": float(r["test_cosine"]), "recall_at_5": float(r["recall_at_5"])}

    summary = {
        "priority1_token_identity_control": p1_controls,
        "priority2_scaling": scaling_rows,
        "priority3_decoding_comparison": decoding_rows,
    }
    with open("artifacts/final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[build_summaries] wrote artifacts/final_summary.json")


def main():
    scaling_rows = build_scaling_summary()
    decoding_rows = build_decoding_comparison()
    build_final_summary(scaling_rows, decoding_rows)


if __name__ == "__main__":
    main()

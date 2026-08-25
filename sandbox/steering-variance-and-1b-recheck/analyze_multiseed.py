"""
analyze_multiseed.py

Reads artifacts/norm_relative_multiseed_raw.csv and produces:
  - artifacts/norm_relative_multiseed_summary.csv
  - artifacts/norm_relative_multiseed_pairwise.csv

95% CI is a t-interval for the mean (ddof=1 sample std) when n_seeds >= 4.
"""
import csv
import json
import math
import os

import numpy as np
from scipy import stats

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(OUT_DIR, "artifacts")

MODEL_ORDER = ["pythia-410m", "pythia-1.0b", "pythia-1.4b", "pythia-2.8b"]


def load_raw():
    rows = []
    with open(f"{ART_DIR}/norm_relative_multiseed_raw.csv") as f:
        for r in csv.DictReader(f):
            r["margin"] = float(r["margin"])
            r["best_k"] = float(r["best_k"])
            rows.append(r)
    return rows


def summarize(rows):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    summary = []
    for tag in MODEL_ORDER:
        if tag not in by_model:
            continue
        rs = by_model[tag]
        margins = np.array([r["margin"] for r in rs], dtype=float)
        n = len(margins)
        mean = float(np.mean(margins))
        std = float(np.std(margins, ddof=1)) if n >= 2 else float("nan")
        sem = std / math.sqrt(n) if n >= 2 else float("nan")
        if n >= 4:
            tcrit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low = mean - tcrit * sem
            ci_high = mean + tcrit * sem
        else:
            ci_low = "NA"
            ci_high = "NA"
        summary.append({
            "model": tag,
            "n_seeds": n,
            "best_k": rs[0]["best_k"],
            "mean_margin": mean,
            "std_margin": std,
            "sem_margin": sem,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "min_margin": float(np.min(margins)),
            "max_margin": float(np.max(margins)),
        })
    return summary


def ci_overlap(a_low, a_high, b_low, b_high):
    if a_low == "NA" or b_low == "NA":
        return "NA (insufficient seeds for CI)"
    return bool(a_low <= b_high and b_low <= a_high)


def pairwise(summary):
    by_model = {s["model"]: s for s in summary}
    pairs = [
        ("pythia-1.0b", "pythia-410m"),
        ("pythia-1.0b", "pythia-1.4b"),
        ("pythia-2.8b", "pythia-410m"),
        ("pythia-2.8b", "pythia-1.4b"),
    ]
    out = []
    for focal, comp in pairs:
        if focal not in by_model or comp not in by_model:
            continue
        f, c = by_model[focal], by_model[comp]
        diff = f["mean_margin"] - c["mean_margin"]
        overlap = ci_overlap(f["ci95_low"], f["ci95_high"], c["ci95_low"], c["ci95_high"])
        out.append({
            "comparison": f"{focal.replace('pythia-', '').replace('.', 'p')}_minus_"
                          f"{comp.replace('pythia-', '').replace('.', 'p')}",
            "focal_model": focal,
            "comparator_model": comp,
            "mean_diff": diff,
            "focal_mean": f["mean_margin"],
            "comparator_mean": c["mean_margin"],
            "focal_ci95_low": f["ci95_low"],
            "focal_ci95_high": f["ci95_high"],
            "comparator_ci95_low": c["ci95_low"],
            "comparator_ci95_high": c["ci95_high"],
            "ci_overlap": overlap,
        })
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    rows = load_raw()
    summary = summarize(rows)
    write_csv(f"{ART_DIR}/norm_relative_multiseed_summary.csv", summary,
              ["model", "n_seeds", "best_k", "mean_margin", "std_margin", "sem_margin",
               "ci95_low", "ci95_high", "min_margin", "max_margin"])

    pw = pairwise(summary)
    write_csv(f"{ART_DIR}/norm_relative_multiseed_pairwise.csv", pw,
              ["comparison", "focal_model", "comparator_model", "mean_diff",
               "focal_mean", "comparator_mean", "focal_ci95_low", "focal_ci95_high",
               "comparator_ci95_low", "comparator_ci95_high", "ci_overlap"])

    print(json.dumps({"summary": summary, "pairwise": pw}, indent=2, default=str))


if __name__ == "__main__":
    main()

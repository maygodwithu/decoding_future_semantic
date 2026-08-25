"""
run_experiment.py

Denser k-sweep for pythia-1.0b + multi-seed repeated norm-relative steering
evaluation for all four Pythia sizes. Reuses the accepted norm-relative
steering pipeline unchanged except for (a) a wider k grid for pythia-1.0b's
pilot tuning, and (b) a seed argument controlling test-prompt subsampling and
the per-prompt random control direction for repeated evaluation. See
reused_pipeline_reference.txt for the full list of what is reused verbatim.

Two stages, run separately so a crash in one doesn't lose the other:
  --stage dense_k_sweep   : pilot k-sweep for pythia-1.0b only (val split)
  --stage multiseed       : repeated test-set evaluation for chosen models,
                             using best_k_1b_dense for 1.0b and the accepted
                             best_k_pilot from norm_relative_scaling_summary.csv
                             for the other three models.
"""
import argparse
import csv
import gc
import json
import os
import random
import shutil
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

REUSE_DIR = "/home/jkchoi/project/autopaper/sandbox/pythia-scaling-and-controls"
NORM_REL_DIR = "/home/jkchoi/project/autopaper/sandbox/steering-scale-alpha-normalization"
sys.path.insert(0, REUSE_DIR)

from common import load_model_and_tokenizer, read_jsonl  # noqa: E402
from embed_continuations import embed_texts, get_embedder  # noqa: E402
from steering_utils_generic import BatchedInjectionHook  # noqa: E402
import steering_generic as SG  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(OUT_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

TUNE_SEED = 42          # unchanged from the accepted run: pilot tuning split seed
N_PILOT = 40             # unchanged
N_TEST = 60              # unchanged
BATCH_SIZE = 20          # unchanged
SEEDS = [101, 202, 303, 404, 505]

DENSE_K_GRID_1B = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6, 2.4, 3.2, 4.8, 6.4, 8.0]

MODEL_NAME_BY_TAG = {
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.0b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
}
PRIORITY_ORDER = ["pythia-1.0b", "pythia-2.8b", "pythia-410m", "pythia-1.4b"]


def log(msg):
    print(f"[run_experiment] {msg}", flush=True)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_old_rows():
    old_rows = {}
    with open(f"{REUSE_DIR}/artifacts/scaling_summary.csv") as f:
        for r in csv.DictReader(f):
            old_rows[r["model"]] = r
    return old_rows


def load_norm_relative_summary():
    rows = {}
    with open(f"{NORM_REL_DIR}/artifacts/norm_relative_scaling_summary.csv") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


def load_sentiment_axis():
    embedder = get_embedder()
    pos_emb = embedder.encode(SG.POSITIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    neg_emb = embedder.encode(SG.NEGATIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    pos_mean, neg_mean = pos_emb.mean(axis=0), neg_emb.mean(axis=0)
    v_sem = pos_mean - neg_mean
    v_sem = v_sem / (np.linalg.norm(v_sem) + 1e-12)
    return pos_mean, neg_mean, v_sem


def select_eligible(idx_pool, continuations, seed, cap):
    eligible = [i for i in idx_pool if len(continuations[i].get("continuation_token_ids") or []) >= SG.MIN_CONT_TOKENS]
    if not eligible:
        eligible = [i for i in idx_pool if len(continuations[i]["continuation_text"].split()) >= 5]
    rng = np.random.RandomState(seed)
    eligible = list(eligible)
    rng.shuffle(eligible)
    return eligible[:cap]


def build_prompt_ids(eligible, continuations, tokenizer):
    prompt_ids_list = [continuations[i].get("prompt_token_ids") for i in eligible]
    if any(p is None for p in prompt_ids_list):
        prompt_ids_list = [tokenizer.encode(continuations[i]["prompt_text"]) for i in eligible]
    return prompt_ids_list


def build_rand_dirs(eligible, continuations, hidden_dim, seed_offset=0):
    """seed_offset=0 reproduces the ORIGINAL accepted-run formula exactly
    (base seed 200000+prompt_id). For multi-seed repeats, seed_offset is set
    to 1_000_003*seed so the control direction also varies per repeat."""
    d_rand_list = []
    for i in eligible:
        prompt_rng = np.random.RandomState(200000 + seed_offset + int(continuations[i]["prompt_id"]))
        d = prompt_rng.normal(size=hidden_dim)
        d = d / np.linalg.norm(d)
        d_rand_list.append(d)
    return np.stack(d_rand_list)


def generate_all(model, tokenizer, hook, prompt_ids_list, delta_arr, device, eos_id, batch_size=BATCH_SIZE):
    texts = []
    for b in range(0, len(prompt_ids_list), batch_size):
        chunk = prompt_ids_list[b:b + batch_size]
        d_chunk = None if delta_arr is None else delta_arr[b:b + batch_size]
        texts += SG.batched_generate(model, tokenizer, chunk, hook, d_chunk, device, eos_id)
    return texts


def clean_model_cache(model_name):
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    safe = "models--" + model_name.replace("/", "--")
    p = os.path.join(cache_root, safe)
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
        log(f"cleaned HF cache for {model_name}")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"wrote {path} ({len(rows)} rows)")


def get_layer_and_norm(tag, old_rows, split, force_recompute=True):
    layer = int(old_rows[tag]["best_layer"])
    mdir = f"{REUSE_DIR}/artifacts/{tag}/greedy"
    npz = np.load(f"{mdir}/hidden_last_token.npz")
    hs_val = npz[f"layer_{layer}"][split["val"]]
    mean_norm = float(np.linalg.norm(hs_val, axis=1).mean())
    return layer, mean_norm


# ----------------------------------------------------------------------------
# Stage 1: dense k-sweep for pythia-1.0b (pilot/val split only, TUNE_SEED=42,
# identical procedure to norm_relative_scaling.py's pilot sweep, just a wider
# k grid).
# ----------------------------------------------------------------------------
def run_dense_k_sweep():
    tag = "pythia-1.0b"
    model_name = MODEL_NAME_BY_TAG[tag]
    old_rows = load_old_rows()
    split = json.load(open(f"{REUSE_DIR}/artifacts/split.json"))
    idx_val = split["val"]
    usable = read_jsonl(f"{REUSE_DIR}/artifacts/usable_prompts.jsonl")

    layer, mean_norm = get_layer_and_norm(tag, old_rows, split)
    log(f"{tag}: layer={layer} mean_hidden_norm(val)={mean_norm:.4f}")

    mdir = f"{REUSE_DIR}/artifacts/{tag}/greedy"
    probe_npz = np.load(f"{mdir}/best_probe_trainval_layer_{layer}.npz")
    W = probe_npz["W"]
    hidden_dim = W.shape[1]
    continuations = read_jsonl(f"{mdir}/continuations.jsonl")

    pos_mean, neg_mean, v_sem = load_sentiment_axis()
    d_sem = W.T @ v_sem
    d_sem = d_sem / (np.linalg.norm(d_sem) + 1e-12)

    eligible_pilot = select_eligible(idx_val, continuations, TUNE_SEED, N_PILOT)
    log(f"{tag}: eligible pilot(val) = {len(eligible_pilot)} prompts (cap {N_PILOT})")

    set_all_seeds(TUNE_SEED)
    device = "cuda"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    eos_id = tokenizer.eos_token_id
    hook = BatchedInjectionHook(model, layer)

    pilot_ids_list = build_prompt_ids(eligible_pilot, continuations, tokenizer)
    d_rand_pilot = build_rand_dirs(eligible_pilot, continuations, hidden_dim, seed_offset=0)

    base_pilot_texts = generate_all(model, tokenizer, hook, pilot_ids_list, None, device, eos_id)
    score_base_pilot = SG.axis_scores(base_pilot_texts, pos_mean, neg_mean)

    rows = []
    for k in DENSE_K_GRID_1B:
        alpha = k * mean_norm
        sem_arr = torch.tensor(np.tile(d_sem, (len(eligible_pilot), 1)), dtype=torch.float32) * alpha
        rand_arr = torch.tensor(d_rand_pilot, dtype=torch.float32) * alpha
        sem_texts = generate_all(model, tokenizer, hook, pilot_ids_list, sem_arr, device, eos_id)
        rand_texts = generate_all(model, tokenizer, hook, pilot_ids_list, rand_arr, device, eos_id)
        score_sem = SG.axis_scores(sem_texts, pos_mean, neg_mean)
        score_rand = SG.axis_scores(rand_texts, pos_mean, neg_mean)
        effect_sem = float(np.mean(score_sem - score_base_pilot))
        effect_rand = float(np.mean(score_rand - score_base_pilot))
        margin = effect_sem - effect_rand
        rows.append({
            "model": tag, "k": k, "seed_or_split_id": f"pilot_val_seed{TUNE_SEED}",
            "margin": margin, "semantic_score": effect_sem, "random_score": effect_rand,
            "raw_alpha": alpha, "num_prompts": len(eligible_pilot),
        })
        log(f"  k={k:<6} alpha={alpha:9.3f} sem={effect_sem:+.5f} rand={effect_rand:+.5f} margin={margin:+.5f}")

    hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    clean_model_cache(model_name)

    write_csv(f"{ART_DIR}/pythia_1b_dense_k_sweep.csv", rows,
              ["model", "k", "seed_or_split_id", "margin", "semantic_score", "random_score", "raw_alpha", "num_prompts"])

    ranked = sorted(rows, key=lambda r: r["margin"], reverse=True)
    best = ranked[0]
    top5 = ranked[:5]
    old_best_k = 0.4
    old_row = next((r for r in rows if abs(r["k"] - old_best_k) < 1e-9), None)
    selection = {
        "model": tag,
        "layer": layer,
        "mean_hidden_norm_val": mean_norm,
        "k_grid": DENSE_K_GRID_1B,
        "best_k_1b_dense": best["k"],
        "best_margin": best["margin"],
        "top5_by_margin": [{"k": r["k"], "margin": r["margin"], "raw_alpha": r["raw_alpha"]} for r in top5],
        "old_sparse_grid_best_k": old_best_k,
        "old_sparse_grid_margin_on_this_dense_sweep": old_row["margin"] if old_row else None,
        "k_0p4_present_in_dense_grid": old_row is not None,
        "k_0p4_rank_by_margin": (ranked.index(old_row) + 1) if old_row else None,
    }
    with open(f"{ART_DIR}/pythia_1b_dense_k_selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    log(f"wrote {ART_DIR}/pythia_1b_dense_k_selection.json -> best_k_1b_dense={best['k']} (margin={best['margin']:+.5f})")
    return selection


# ----------------------------------------------------------------------------
# Stage 2: multi-seed repeated evaluation on the held-out test split.
# ----------------------------------------------------------------------------
def get_best_k_map():
    norm_rel = load_norm_relative_summary()
    best_k = {
        "pythia-410m": float(norm_rel["pythia-410m"]["best_k_pilot"]),
        "pythia-1.4b": float(norm_rel["pythia-1.4b"]["best_k_pilot"]),
        "pythia-2.8b": float(norm_rel["pythia-2.8b"]["best_k_pilot"]),
    }
    sel_path = f"{ART_DIR}/pythia_1b_dense_k_selection.json"
    if not os.path.exists(sel_path):
        raise RuntimeError("Run --stage dense_k_sweep first (missing pythia_1b_dense_k_selection.json)")
    sel = json.load(open(sel_path))
    best_k["pythia-1.0b"] = float(sel["best_k_1b_dense"])

    with open(f"{ART_DIR}/best_k_reused_from_prior.json", "w") as f:
        json.dump({
            "source": f"{NORM_REL_DIR}/artifacts/norm_relative_scaling_summary.csv (column best_k_pilot)",
            "metadata_missing": False,
            "pythia-410m_best_k": best_k["pythia-410m"],
            "pythia-1.4b_best_k": best_k["pythia-1.4b"],
            "pythia-2.8b_best_k": best_k["pythia-2.8b"],
            "pythia-1.0b_best_k_note": "NOT reused; retuned in this run via dense k-sweep (see pythia_1b_dense_k_selection.json)",
            "pythia-1.0b_best_k_dense": best_k["pythia-1.0b"],
        }, f, indent=2)
    return best_k


def run_multiseed(models_to_run):
    old_rows = load_old_rows()
    split = json.load(open(f"{REUSE_DIR}/artifacts/split.json"))
    idx_test = split["test"]
    best_k_map = get_best_k_map()
    pos_mean, neg_mean, v_sem = load_sentiment_axis()

    raw_path = f"{ART_DIR}/norm_relative_multiseed_raw.csv"
    existing_rows = []
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            existing_rows = list(csv.DictReader(f))
    done_keys = {(r["model"], r["seed"]) for r in existing_rows}

    fieldnames = ["model", "seed", "best_k", "layer", "mean_hidden_norm", "raw_alpha",
                  "semantic_score", "random_score", "margin", "num_prompts"]
    all_rows = list(existing_rows)

    for tag in models_to_run:
        model_name = MODEL_NAME_BY_TAG[tag]
        layer, mean_norm = get_layer_and_norm(tag, old_rows, split)
        best_k = best_k_map[tag]
        alpha = best_k * mean_norm
        log(f"=== {tag} ({model_name}) === layer={layer} mean_hidden_norm={mean_norm:.3f} "
            f"best_k={best_k} alpha={alpha:.3f}")

        seeds_needed = [s for s in SEEDS if (tag, str(s)) not in done_keys]
        if not seeds_needed:
            log(f"{tag}: all seeds already present in {raw_path}, skipping")
            continue

        mdir = f"{REUSE_DIR}/artifacts/{tag}/greedy"
        probe_npz = np.load(f"{mdir}/best_probe_trainval_layer_{layer}.npz")
        W = probe_npz["W"]
        hidden_dim = W.shape[1]
        continuations = read_jsonl(f"{mdir}/continuations.jsonl")
        d_sem = W.T @ v_sem
        d_sem = d_sem / (np.linalg.norm(d_sem) + 1e-12)

        device = "cuda"
        t_load0 = time.time()
        model, tokenizer = load_model_and_tokenizer(model_name, device=device)
        eos_id = tokenizer.eos_token_id
        hook = BatchedInjectionHook(model, layer)
        log(f"{tag}: model loaded in {time.time() - t_load0:.1f}s")

        for seed in seeds_needed:
            t0 = time.time()
            set_all_seeds(seed)
            eligible_test = select_eligible(idx_test, continuations, seed, N_TEST)
            test_ids_list = build_prompt_ids(eligible_test, continuations, tokenizer)
            d_rand_test = build_rand_dirs(eligible_test, continuations, hidden_dim, seed_offset=1_000_003 * seed)

            base_texts = generate_all(model, tokenizer, hook, test_ids_list, None, device, eos_id)
            sem_arr = torch.tensor(np.tile(d_sem, (len(eligible_test), 1)), dtype=torch.float32) * alpha
            rand_arr = torch.tensor(d_rand_test, dtype=torch.float32) * alpha
            sem_texts = generate_all(model, tokenizer, hook, test_ids_list, sem_arr, device, eos_id)
            rand_texts = generate_all(model, tokenizer, hook, test_ids_list, rand_arr, device, eos_id)

            score_base = SG.axis_scores(base_texts, pos_mean, neg_mean)
            score_sem = SG.axis_scores(sem_texts, pos_mean, neg_mean)
            score_rand = SG.axis_scores(rand_texts, pos_mean, neg_mean)
            effect_sem = float(np.mean(score_sem - score_base))
            effect_rand = float(np.mean(score_rand - score_base))
            margin = effect_sem - effect_rand

            row = {
                "model": tag, "seed": seed, "best_k": best_k, "layer": layer,
                "mean_hidden_norm": mean_norm, "raw_alpha": alpha,
                "semantic_score": effect_sem, "random_score": effect_rand, "margin": margin,
                "num_prompts": len(eligible_test),
            }
            all_rows.append(row)
            write_csv(raw_path, all_rows, fieldnames)  # checkpoint after every seed
            log(f"  seed={seed} n={len(eligible_test)} sem={effect_sem:+.5f} rand={effect_rand:+.5f} "
                f"margin={margin:+.5f} ({time.time() - t0:.1f}s)")

        hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if tag != "pythia-1.4b":  # 1.4b was already HF-cached before this run; leave it cached
            clean_model_cache(model_name)
        log(f"=== {tag} done ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dense_k_sweep", "multiseed"])
    ap.add_argument("--models", default=",".join(PRIORITY_ORDER),
                     help="comma-separated model tags for --stage multiseed, in run order")
    args = ap.parse_args()

    if args.stage == "dense_k_sweep":
        run_dense_k_sweep()
    else:
        models_to_run = [m for m in args.models.split(",") if m]
        run_multiseed(models_to_run)


if __name__ == "__main__":
    main()

"""
run_layer_sweep.py

Hypothesis test: is pythia-1.0b's null causal-steering result (accepted run in
steering-variance-and-1b-recheck/: mean margin -0.00002, 95% CI
[-0.00307, 0.00303] at layer=13, the layer chosen for PASSIVE probing) a
layer-selection artifact, or a real model-level lack of steerability?

Reuses the exact accepted norm-relative steering pipeline (same injection
hook, same sentiment-axis direction construction from a trained probe, same
random-control-direction recipe, same margin definition, same eligibility
filter / pilot-test split / k-sweep-then-multiseed structure, same CI method)
UNCHANGED, restricted to EleutherAI/pythia-1b on GPU 0, and swept across 7
candidate layers spanning the network instead of the single passive-probing
best layer.

Reused verbatim from ../pythia-scaling-and-controls/:
  common.py (load_model_and_tokenizer, read_jsonl, find_sentence_cut)
  embed_continuations.py (embed_texts, get_embedder -- all-MiniLM-L6-v2)
  steering_utils_generic.py (BatchedInjectionHook)
  steering_generic.py (batched_generate, axis_scores, POSITIVE_ANCHORS,
    NEGATIVE_ANCHORS, MIN_CONT_TOKENS=8, MAX_NEW_TOKENS=24)
  artifacts/split.json (seed-42 train/val/test split of the 549 usable
    prompts; val=83 used as pilot pool, test=83 used as final eval pool)
  artifacts/pythia-1.0b/greedy/continuations.jsonl (fixed unsteered greedy
    continuations + prompt_token_ids, model-and-prompt-set dependent only,
    NOT layer dependent -- reused unchanged)
  artifacts/pythia-1.0b/greedy/continuation_embeddings.npy (sentence
    embeddings of those continuations, the probe-training regression target,
    layer independent -- reused unchanged)

New in this run (the only allowed deviation, per protocol):
  - hidden states extracted at 7 candidate layers spanning the network
    (instead of only the one passive-probing best layer)
  - a ridge probe trained at each candidate layer (identical procedure /
    alpha grid to probe_train_generic.py) to build that layer's semantic
    steering direction d_sem = normalize(W.T @ v_sem)
  - per-layer pilot k-sweep (DENSE_K_GRID_1B, reused VERBATIM from the
    accepted steering-variance-and-1b-recheck run, which is the most recent
    accepted k-grid used specifically for pythia-1.0b) and per-layer
    norm-relative alpha (alpha = k * mean_hidden_norm(layer), same rule as
    the accepted norm-relative pipeline)
  - per-layer multi-seed (5 seeds, same SEEDS list, same seed-handling) test
    evaluation and 95% CI (same t-interval method as analyze_multiseed.py)

Layer indexing note: the protocol's layer-selection formula
  [0, round(.17*(n-1)), round(.33*(n-1)), round(.50*(n-1)),
   round(.67*(n-1)), round(.83*(n-1)), n-1]
with n=n_layers produces values in [0, n_layers-1], i.e. 0-indexed
TRANSFORMER BLOCK numbers (block 0 = first block, block n_layers-1 = last
block) -- this is the only interpretation under which layer 0 is a steerable
block. The prior pipeline's BatchedInjectionHook(model, layer_idx) hooks
gpt_neox.layers[layer_idx-1], i.e. hidden_states[layer_idx]. So block b is
addressed internally as layer_idx = b+1 (hidden_states index). All CSV/JSON
outputs report the protocol's block number b as "layer".

Two stages (run sequentially; each stage checkpoints to disk so a crash in
one doesn't lose the other):
  --stage prep       : extract hidden states at the 7 candidate layers, train
                        per-layer ridge probes, build d_sem + mean_hidden_norm
                        per layer -> artifacts/pythia_1b_layer_sweep/layer_directions.npz
  --stage k_sweep     : per-layer pilot k-sweep -> k_sweep_results.csv, selected_alphas.csv
  --stage multiseed   : per-layer multi-seed test eval -> per_seed_margins.csv
  --stage aggregate   : layer_summary.csv + report.md
"""
import argparse
import csv
import gc
import json
import math
import os
import shutil
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge

REUSE_DIR = "/home/jkchoi/project/autopaper/sandbox/pythia-scaling-and-controls"
sys.path.insert(0, REUSE_DIR)

from common import load_model_and_tokenizer, read_jsonl, find_sentence_cut  # noqa: E402
from embed_continuations import embed_texts, get_embedder  # noqa: E402
from steering_utils_generic import BatchedInjectionHook  # noqa: E402
import steering_generic as SG  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(OUT_DIR, "artifacts", "pythia_1b_layer_sweep")
os.makedirs(ART_DIR, exist_ok=True)

MODEL_NAME = "EleutherAI/pythia-1b"
MDIR = f"{REUSE_DIR}/artifacts/pythia-1.0b/greedy"
SPLIT_PATH = f"{REUSE_DIR}/artifacts/split.json"

BLOCKS = [0, 3, 5, 8, 10, 12, 15]     # 0-indexed transformer block numbers (protocol formula, n_layers=16)
HS_INDEX = {b: b + 1 for b in BLOCKS}  # hidden_states index = block + 1 (BatchedInjectionHook convention)

ALPHA_GRID_PROBE = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]  # verbatim probe_train_generic.py
DENSE_K_GRID_1B = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6, 2.4, 3.2, 4.8, 6.4, 8.0]  # verbatim steering-variance-and-1b-recheck

TUNE_SEED = 42
N_PILOT = 40
N_TEST = 60
BATCH_SIZE = 20
EXTRACT_BATCH_SIZE = 24
SEEDS = [101, 202, 303, 404, 505]
PROBE_SEED = 42


def log(msg):
    print(f"[run_layer_sweep] {msg}", flush=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"wrote {path} ({len(rows)} rows)")


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


def load_sentiment_axis():
    embedder = get_embedder()
    pos_emb = embedder.encode(SG.POSITIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    neg_emb = embedder.encode(SG.NEGATIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    pos_mean, neg_mean = pos_emb.mean(axis=0), neg_emb.mean(axis=0)
    v_sem = pos_mean - neg_mean
    v_sem = v_sem / (np.linalg.norm(v_sem) + 1e-12)
    return pos_mean, neg_mean, v_sem


# ----------------------------------------------------------------------------
# Stage: prep -- extract hidden states at the 7 candidate layers (single
# forward pass over all 549 prompts, exactly mirroring the hidden-state
# extraction block of extract_hidden_and_generations.py, just parameterized
# by our layer list instead of the depth-fraction-matched list used for
# passive probing), then train a ridge probe per layer (identical procedure
# to probe_train_generic.py) and build d_sem + mean_hidden_norm(val) per layer.
# ----------------------------------------------------------------------------
def extract_hidden_states(model, tokenizer, continuations, device):
    hs_indices = sorted(set(HS_INDEX.values()))
    n = len(continuations)
    hidden_dim = model.config.hidden_size
    out_tensors = {L: np.zeros((n, hidden_dim), dtype=np.float32) for L in hs_indices}

    tokenizer.padding_side = "right"
    order = sorted(range(n), key=lambda i: len(continuations[i]["prompt_token_ids"]))
    n_batches = (n + EXTRACT_BATCH_SIZE - 1) // EXTRACT_BATCH_SIZE
    t0 = time.time()
    with torch.no_grad():
        for b in range(n_batches):
            idxs = order[b * EXTRACT_BATCH_SIZE:(b + 1) * EXTRACT_BATCH_SIZE]
            batch_ids = [continuations[i]["prompt_token_ids"] for i in idxs]
            lengths = [len(x) for x in batch_ids]
            maxlen = max(lengths)
            input_ids = torch.full((len(batch_ids), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
            for r, ids in enumerate(batch_ids):
                input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[r, :len(ids)] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            hs = out.hidden_states
            for r, orig_idx in enumerate(idxs):
                last_pos = lengths[r] - 1
                for L in hs_indices:
                    out_tensors[L][orig_idx] = hs[L][r, last_pos, :].float().cpu().numpy()
            if b % 5 == 0 or b == n_batches - 1:
                log(f"  hidden extract batch {b + 1}/{n_batches} ({time.time() - t0:.1f}s)")
    return out_tensors


def train_probe_at_layer(X, y, idx_train, idx_val, idx_test):
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    def cosine_rows(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return np.sum(a * b, axis=1)

    best_alpha, best_val_cos, best_model = None, -2.0, None
    for a in ALPHA_GRID_PROBE:
        m = Ridge(alpha=a, random_state=PROBE_SEED)
        m.fit(X_train, y_train)
        val_cos = float(np.mean(cosine_rows(m.predict(X_val), y_val)))
        if val_cos > best_val_cos:
            best_val_cos, best_alpha, best_model = val_cos, a, m

    test_cos = float(np.mean(cosine_rows(best_model.predict(X_test), y_test)))

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    final_model = Ridge(alpha=best_alpha, random_state=PROBE_SEED)
    final_model.fit(X_trainval, y_trainval)
    return final_model.coef_, {"best_alpha": best_alpha, "val_cosine": best_val_cos, "test_cosine": test_cos}


def run_prep():
    split = json.load(open(SPLIT_PATH))
    idx_train, idx_val, idx_test = split["train"], split["val"], split["test"]
    continuations = read_jsonl(f"{MDIR}/continuations.jsonl")
    y = np.load(f"{MDIR}/continuation_embeddings.npy")
    assert y.shape[0] == len(continuations)

    device = "cuda"
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)
    log(f"model loaded; n_layers={model.config.num_hidden_layers} hidden_size={model.config.hidden_size}")

    hidden_by_hsidx = extract_hidden_states(model, tokenizer, continuations, device)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    pos_mean, neg_mean, v_sem = load_sentiment_axis()

    diagnostics = []
    layer_data = {}
    for b in BLOCKS:
        L = HS_INDEX[b]
        X = hidden_by_hsidx[L]
        W, diag = train_probe_at_layer(X, y, idx_train, idx_val, idx_test)
        d_sem = W.T @ v_sem
        d_sem = d_sem / (np.linalg.norm(d_sem) + 1e-12)
        hs_val = X[idx_val]
        mean_norm = float(np.linalg.norm(hs_val, axis=1).mean())
        layer_data[b] = {"d_sem": d_sem, "mean_hidden_norm": mean_norm, "hs_index": L}
        diagnostics.append({
            "layer": b, "hs_index": L, "probe_val_cosine": diag["val_cosine"],
            "probe_test_cosine": diag["test_cosine"], "probe_best_alpha": diag["best_alpha"],
            "mean_hidden_norm_val": mean_norm,
        })
        log(f"layer(block)={b} hs_index={L} probe_val_cos={diag['val_cosine']:.4f} "
            f"probe_test_cos={diag['test_cosine']:.4f} mean_hidden_norm(val)={mean_norm:.3f}")

    np.savez(f"{ART_DIR}/layer_directions.npz",
              **{f"d_sem_layer{b}": layer_data[b]["d_sem"] for b in BLOCKS})
    with open(f"{ART_DIR}/layer_prep_meta.json", "w") as f:
        json.dump({
            "blocks": BLOCKS,
            "hs_index_map": HS_INDEX,
            "mean_hidden_norm": {str(b): layer_data[b]["mean_hidden_norm"] for b in BLOCKS},
            "probe_diagnostics": diagnostics,
        }, f, indent=2)
    write_csv(f"{ART_DIR}/layer_probe_diagnostics.csv", diagnostics,
              ["layer", "hs_index", "probe_val_cosine", "probe_test_cosine", "probe_best_alpha", "mean_hidden_norm_val"])
    log("prep stage done")


def load_layer_data():
    npz = np.load(f"{ART_DIR}/layer_directions.npz")
    meta = json.load(open(f"{ART_DIR}/layer_prep_meta.json"))
    layer_data = {}
    for b in BLOCKS:
        layer_data[b] = {
            "d_sem": npz[f"d_sem_layer{b}"],
            "mean_hidden_norm": meta["mean_hidden_norm"][str(b)],
            "hs_index": HS_INDEX[b],
        }
    return layer_data


# ----------------------------------------------------------------------------
# Stage: k_sweep -- per-layer pilot k-sweep on the val (pilot) split, DENSE_K_GRID_1B,
# same procedure as steering-variance-and-1b-recheck/run_experiment.py::run_dense_k_sweep,
# just repeated per candidate layer instead of only layer=13.
# ----------------------------------------------------------------------------
def run_k_sweep():
    split = json.load(open(SPLIT_PATH))
    idx_val = split["val"]
    continuations = read_jsonl(f"{MDIR}/continuations.jsonl")
    layer_data = load_layer_data()
    pos_mean, neg_mean, v_sem = load_sentiment_axis()

    device = "cuda"
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)
    eos_id = tokenizer.eos_token_id

    all_rows = []
    selection = {}
    for b in BLOCKS:
        L = layer_data[b]["hs_index"]
        d_sem = layer_data[b]["d_sem"]
        mean_norm = layer_data[b]["mean_hidden_norm"]
        hidden_dim = d_sem.shape[0]
        log(f"=== layer(block)={b} (hs_index={L}) mean_hidden_norm(val)={mean_norm:.3f} ===")

        eligible_pilot = select_eligible(idx_val, continuations, TUNE_SEED, N_PILOT)
        pilot_ids_list = build_prompt_ids(eligible_pilot, continuations, tokenizer)
        d_rand_pilot = build_rand_dirs(eligible_pilot, continuations, hidden_dim, seed_offset=0)

        hook = BatchedInjectionHook(model, L)
        base_pilot_texts = generate_all(model, tokenizer, hook, pilot_ids_list, None, device, eos_id)
        score_base_pilot = SG.axis_scores(base_pilot_texts, pos_mean, neg_mean)

        layer_rows = []
        for k in DENSE_K_GRID_1B:
            alpha = k * mean_norm
            sem_arr = torch.tensor(np.tile(d_sem, (len(eligible_pilot), 1)), dtype=torch.float32) * alpha
            rand_arr = torch.tensor(d_rand_pilot, dtype=torch.float32) * alpha
            sem_texts = generate_all(model, tokenizer, hook, pilot_ids_list, sem_arr, device, eos_id)
            rand_texts = generate_all(model, tokenizer, hook, pilot_ids_list, rand_arr, device, eos_id)
            score_sem = SG.axis_scores(sem_texts, pos_mean, neg_mean)
            score_rand = SG.axis_scores(rand_texts, pos_mean, neg_mean)
            per_prompt_margin = (score_sem - score_base_pilot) - (score_rand - score_base_pilot)
            margin_mean = float(np.mean(per_prompt_margin))
            margin_std = float(np.std(per_prompt_margin, ddof=1))
            layer_rows.append({
                "layer": b, "k": k,
                "alpha_mean_or_formula_inputs": f"k={k}*mean_hidden_norm={mean_norm:.4f}=alpha={alpha:.4f}",
                "pilot_margin_mean": margin_mean, "pilot_margin_std": margin_std,
                "n_examples_or_trials": len(eligible_pilot), "selected_for_layer": False,
            })
            log(f"  k={k:<6} alpha={alpha:9.3f} margin_mean={margin_mean:+.5f} margin_std={margin_std:.5f}")

        hook.remove()

        best_row = max(layer_rows, key=lambda r: r["pilot_margin_mean"])
        best_row["selected_for_layer"] = True
        selection[b] = {"selected_k": best_row["k"], "mean_hidden_norm": mean_norm,
                         "final_alpha": best_row["k"] * mean_norm, "pilot_margin_at_best_k": best_row["pilot_margin_mean"]}
        all_rows.extend(layer_rows)
        log(f"layer(block)={b}: selected_k={best_row['k']} pilot_margin={best_row['pilot_margin_mean']:+.5f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    write_csv(f"{ART_DIR}/k_sweep_results.csv", all_rows,
              ["layer", "k", "alpha_mean_or_formula_inputs", "pilot_margin_mean", "pilot_margin_std",
               "n_examples_or_trials", "selected_for_layer"])

    alpha_rows = [{
        "layer": b, "selected_k": selection[b]["selected_k"],
        "hidden_state_norm_stat_used": selection[b]["mean_hidden_norm"],
        "final_alpha": selection[b]["final_alpha"],
    } for b in BLOCKS]
    write_csv(f"{ART_DIR}/selected_alphas.csv", alpha_rows,
              ["layer", "selected_k", "hidden_state_norm_stat_used", "final_alpha"])
    with open(f"{ART_DIR}/k_sweep_selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    log("k_sweep stage done")


# ----------------------------------------------------------------------------
# Stage: multiseed -- per-layer multi-seed test evaluation, same seed-handling
# and eval sample sizes as steering-variance-and-1b-recheck/run_experiment.py::run_multiseed.
# ----------------------------------------------------------------------------
def run_multiseed():
    split = json.load(open(SPLIT_PATH))
    idx_test = split["test"]
    continuations = read_jsonl(f"{MDIR}/continuations.jsonl")
    layer_data = load_layer_data()
    selection = json.load(open(f"{ART_DIR}/k_sweep_selection.json"))
    pos_mean, neg_mean, v_sem = load_sentiment_axis()

    device = "cuda"
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)
    eos_id = tokenizer.eos_token_id

    raw_path = f"{ART_DIR}/per_seed_margins.csv"
    existing_rows = []
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            existing_rows = list(csv.DictReader(f))
    done_keys = {(int(r["layer"]), int(r["seed"])) for r in existing_rows}
    fieldnames = ["layer", "seed", "margin", "semantic_score", "random_score", "selected_k", "final_alpha", "num_prompts"]
    all_rows = list(existing_rows)

    for b in BLOCKS:
        L = layer_data[b]["hs_index"]
        d_sem = layer_data[b]["d_sem"]
        hidden_dim = d_sem.shape[0]
        best_k = selection[str(b)]["selected_k"]
        alpha = selection[str(b)]["final_alpha"]
        log(f"=== layer(block)={b} (hs_index={L}) best_k={best_k} alpha={alpha:.3f} ===")

        seeds_needed = [s for s in SEEDS if (b, s) not in done_keys]
        if not seeds_needed:
            log(f"layer {b}: all seeds already present, skipping")
            continue

        hook = BatchedInjectionHook(model, L)
        for seed in seeds_needed:
            t0 = time.time()
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

            row = {"layer": b, "seed": seed, "margin": margin, "semantic_score": effect_sem,
                   "random_score": effect_rand, "selected_k": best_k, "final_alpha": alpha,
                   "num_prompts": len(eligible_test)}
            all_rows.append(row)
            write_csv(raw_path, all_rows, fieldnames)
            log(f"  seed={seed} n={len(eligible_test)} margin={margin:+.5f} ({time.time() - t0:.1f}s)")
        hook.remove()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("multiseed stage done")


# ----------------------------------------------------------------------------
# Stage: aggregate -- layer_summary.csv + report.md
# ----------------------------------------------------------------------------
PRIOR_MARGINS = {
    "pythia-410m": {"mean": 0.0565, "ci": (0.043, 0.070)},
    "pythia-1.4b": {"mean": 0.0404, "ci": (0.033, 0.047)},
    "pythia-2.8b": {"mean": 0.0087, "ci": (0.0050, 0.0124)},
}
PRIOR_1B_OLD = {"mean": -0.00002, "ci": (-0.00307, 0.00303), "layer": 13, "note": "passive-probing-optimal layer, from steering-variance-and-1b-recheck"}


def run_aggregate():
    raw_path = f"{ART_DIR}/per_seed_margins.csv"
    rows = []
    with open(raw_path) as f:
        for r in csv.DictReader(f):
            r["layer"] = int(r["layer"])
            r["margin"] = float(r["margin"])
            rows.append(r)
    selection = json.load(open(f"{ART_DIR}/k_sweep_selection.json"))
    model_config = json.load(open(f"{ART_DIR}/model_config.json"))

    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)

    summary = []
    for b in BLOCKS:
        rs = by_layer.get(b, [])
        margins = np.array([r["margin"] for r in rs], dtype=float)
        n = len(margins)
        mean = float(np.mean(margins)) if n else float("nan")
        std = float(np.std(margins, ddof=1)) if n >= 2 else float("nan")
        if n >= 4:
            sem = std / math.sqrt(n)
            tcrit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low, ci_high = mean - tcrit * sem, mean + tcrit * sem
        else:
            ci_low, ci_high = "NA", "NA"
        sel = selection[str(b)]
        summary.append({
            "layer": b, "n_seeds": n, "mean_margin": mean, "std_margin": std,
            "ci95_low": ci_low, "ci95_high": ci_high,
            "selected_k": sel["selected_k"], "final_alpha": sel["final_alpha"],
        })

    write_csv(f"{ART_DIR}/layer_summary.csv", summary,
              ["layer", "n_seeds", "mean_margin", "std_margin", "ci95_low", "ci95_high", "selected_k", "final_alpha"])

    numeric_summary = [s for s in summary if s["n_seeds"] >= 4]
    best = max(numeric_summary, key=lambda s: s["mean_margin"]) if numeric_summary else None
    any_positive_ci = any(isinstance(s["ci95_low"], float) and s["ci95_low"] > 0 for s in numeric_summary)
    verdict = ("layer-selection artifact: at least one pythia-1.0b layer has CI excluding zero" if any_positive_ci
               else "real model-level anomaly: no tested pythia-1.0b layer has CI excluding zero")

    write_report(summary, model_config, best, any_positive_ci, verdict)
    with open(f"{ART_DIR}/verdict.json", "w") as f:
        json.dump({"any_positive_ci_excludes_zero": any_positive_ci, "verdict": verdict,
                    "best_layer": best["layer"] if best else None,
                    "best_layer_mean_margin": best["mean_margin"] if best else None}, f, indent=2)
    log(f"AGGREGATE DONE. verdict={verdict}")
    print(json.dumps(summary, indent=2, default=str))
    return summary, verdict, best


def write_report(summary, model_config, best, any_positive_ci, verdict):
    lines = []
    lines.append("# Pythia-1.0b causal steering layer sweep\n\n")
    lines.append("## Hypothesis\n\n")
    lines.append(
        "The accepted multi-seed run (`steering-variance-and-1b-recheck/`) found pythia-1.0b's causal "
        "steering margin to be effectively zero (mean -0.00002, 95% CI [-0.00307, 0.00303]) at layer=13 "
        "(hidden_states index), the layer chosen because it was BEST for PASSIVE semantic probing, not "
        "necessarily for causal steering. This run tests whether that null result is a layer-selection "
        "artifact (some other layer IS steerable) or a real model-level property (no layer works).\n\n"
    )
    lines.append("## Model config\n\n")
    lines.append(f"- `model_name`: {model_config['model_name']}\n")
    lines.append(f"- `n_layers` (transformer blocks, programmatically read from config): {model_config['n_layers']}\n")
    lines.append(f"- `hidden_size`: {model_config['hidden_size']}\n\n")
    lines.append(
        f"Candidate layers (0-indexed transformer block number, per the protocol formula with "
        f"n_layers={model_config['n_layers']}): **{BLOCKS}** "
        f"(7 distinct layers spanning early/mid/late; block *b* is steered internally at "
        f"hidden_states index *b+1*, i.e. the output of `gpt_neox.layers[b]`).\n\n"
    )
    lines.append(
        "Pipeline reused unchanged from the accepted norm-relative steering run: probe-derived sentiment "
        "direction `d_sem = normalize(W.T @ v_sem)` (a ridge probe trained per-layer here, identical "
        "procedure/alpha-grid to the original passive-probing probe trainer), per-prompt random-direction "
        "control (`RandomState(200000+prompt_id)`), injection at the last real prompt token on the prefill "
        "step only, greedy decoding (`MAX_NEW_TOKENS=24`), the sentiment-axis scoring metric "
        "(`score(text)=cos(emb,pos)-cos(emb,neg)`), the seed-42 train/val/test split (val=83 as pilot, "
        "test=83 as final eval), the dense k-grid `DENSE_K_GRID_1B` reused verbatim from "
        "`steering-variance-and-1b-recheck` (`[0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6, 2.4, 3.2, 4.8, "
        "6.4, 8.0]`), the norm-relative alpha rule `alpha = k * mean_hidden_norm(layer, val split)`, 5 "
        "seeds per layer (`[101, 202, 303, 404, 505]`), and the t-interval 95% CI method.\n\n"
    )

    lines.append("## Per-layer k-sweep winner\n\n")
    lines.append("| layer (block) | selected_k | final_alpha |\n|---:|---:|---:|\n")
    for s in summary:
        lines.append(f"| {s['layer']} | {s['selected_k']} | {s['final_alpha']:.4f} |\n")
    lines.append("\n")

    lines.append("## Per-layer steering margin (mean +/- 95% CI, n seeds)\n\n")
    lines.append("| layer (block) | n_seeds | mean_margin | std_margin | 95% CI |\n|---:|---:|---:|---:|---:|\n")
    for s in summary:
        ci = f"[{s['ci95_low']:.5f}, {s['ci95_high']:.5f}]" if isinstance(s["ci95_low"], float) else "NA"
        lines.append(f"| {s['layer']} | {s['n_seeds']} | {s['mean_margin']:+.5f} | {s['std_margin']:.5f} | {ci} |\n")
    lines.append("\n")

    if best is not None:
        lines.append(f"**Best layer by mean margin: block {best['layer']}**, mean_margin={best['mean_margin']:+.5f}, "
                      f"95% CI=[{best['ci95_low']:.5f}, {best['ci95_high']:.5f}] "
                      f"({'CI excludes zero' if best['ci95_low'] > 0 else 'CI includes zero'}).\n\n")
    else:
        lines.append("No layer had enough seeds (n>=4) to compute a CI.\n\n")

    lines.append("## Comparison to prior model-size results and the old 1.0b layer-13 result\n\n")
    lines.append("| model / layer | mean margin | 95% CI |\n|---|---:|---:|\n")
    lines.append(f"| pythia-410m | {PRIOR_MARGINS['pythia-410m']['mean']} | "
                  f"[{PRIOR_MARGINS['pythia-410m']['ci'][0]}, {PRIOR_MARGINS['pythia-410m']['ci'][1]}] |\n")
    lines.append(f"| pythia-1.4b | {PRIOR_MARGINS['pythia-1.4b']['mean']} | "
                  f"[{PRIOR_MARGINS['pythia-1.4b']['ci'][0]}, {PRIOR_MARGINS['pythia-1.4b']['ci'][1]}] |\n")
    lines.append(f"| pythia-2.8b | {PRIOR_MARGINS['pythia-2.8b']['mean']} | "
                  f"[{PRIOR_MARGINS['pythia-2.8b']['ci'][0]}, {PRIOR_MARGINS['pythia-2.8b']['ci'][1]}] |\n")
    lines.append(f"| pythia-1.0b (OLD, layer=13, passive-probing-optimal) | {PRIOR_1B_OLD['mean']} | "
                  f"[{PRIOR_1B_OLD['ci'][0]}, {PRIOR_1B_OLD['ci'][1]}] |\n")
    if best is not None:
        lines.append(f"| pythia-1.0b (THIS RUN, best layer={best['layer']}) | {best['mean_margin']:+.5f} | "
                      f"[{best['ci95_low']:.5f}, {best['ci95_high']:.5f}] |\n")
    lines.append("\n")

    if best is not None and best["ci95_low"] > 0:
        ratio_410m = best["mean_margin"] / PRIOR_MARGINS["pythia-410m"]["mean"]
        ratio_14b = best["mean_margin"] / PRIOR_MARGINS["pythia-1.4b"]["mean"]
        ratio_28b = best["mean_margin"] / PRIOR_MARGINS["pythia-2.8b"]["mean"]
        lines.append(
            f"The best pythia-1.0b layer's mean margin ({best['mean_margin']:+.5f}) is "
            f"{ratio_410m:.2f}x the 410m margin (0.0565), {ratio_14b:.2f}x the 1.4b margin (0.0404), and "
            f"{ratio_28b:.2f}x the 2.8b margin (0.0087). "
            f"{'This is comparable in order of magnitude to the other three model sizes.' if 0.2 <= ratio_410m <= 5 or 0.2 <= ratio_14b <= 5 else 'This is smaller in magnitude than 410m/1.4b, closer to (or below) the already-weak 2.8b margin, so even though the anomaly is a layer-selection artifact in the strict CI sense, the RECOVERED effect size is still modest relative to 410m/1.4b.'}\n\n"
        )
    else:
        lines.append(
            "No tested layer recovered a positive margin whose 95% CI excludes zero, so no magnitude "
            "comparison to the other model sizes' positive margins (410m=0.0565 CI[0.043,0.070], "
            "1.4b=0.0404 CI[0.033,0.047], 2.8b=0.0087 CI[0.0050,0.0124]) is applicable: pythia-1.0b shows no "
            "evidence of the causal steerability that all three other tested sizes show, at any of the 7 "
            "layers tested here.\n\n"
        )

    lines.append("## Final verdict\n\n")
    lines.append(f"**{verdict}**\n")

    with open(f"{ART_DIR}/report.md", "w") as f:
        f.writelines(lines)
    log(f"wrote {ART_DIR}/report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "k_sweep", "multiseed", "aggregate"])
    args = ap.parse_args()
    if args.stage == "prep":
        run_prep()
    elif args.stage == "k_sweep":
        run_k_sweep()
    elif args.stage == "multiseed":
        run_multiseed()
    else:
        run_aggregate()


if __name__ == "__main__":
    main()

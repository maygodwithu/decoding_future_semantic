"""
P3 - Steering-specific layer sweep on Pythia-1.4B (the paper's anchor model).

Reuses the EXACT finalized norm-relative steering pipeline unchanged (same
injection hook, same sentiment-axis direction construction from a trained
probe, same random-control-direction recipe, same margin definition, same
pilot-then-multiseed structure, same t-interval CI method) from
pythia-scaling-and-controls/ + steering-variance-and-1b-recheck/, restricted
to EleutherAI/pythia-1.4b, swept across 7 candidate layers spanning the
network instead of only the passive-probing-best layer (20).

Layer convention: "layer" = raw hidden_states index (0=embeddings,
n_layers=final block output), matching the convention already used
everywhere else for Pythia-1.4B (hidden_states_layer_20.pt, the accepted
steering run's "layer": 20 field) -- NOT the block+1 offset convention used
internally by the pythia-1b-layer-sweep script. BatchedInjectionHook hooks
gpt_neox.layers[layer_idx-1], so layer_idx must be >=1; layer 0 (raw
embeddings, pre-first-block) has no injectable analogue and is SUBSTITUTED
with layer 1 (the earliest steerable location, output of the first
transformer block) for the steering sweep only -- layer 0 is still reported
in the passive-probe table (Section 3), since passive probing reads
hidden_states[0] directly with no injection needed. This substitution is
documented per the spec's explicit instruction to document any such mapping
deviation. Layer 24 IS a valid injection point for a 24-layer model (hooks
gpt_neox.layers[23], the last block) -- no substitution needed there.

Reused verbatim from ../pythia-scaling-and-controls/:
  common.py, embed_continuations.py, steering_utils_generic.py,
  steering_generic.py, artifacts/split.json (val=83 pilot pool, test=83 final
  eval pool), artifacts/pythia-1.4b/greedy/continuations.jsonl,
  artifacts/pythia-1.4b/greedy/continuation_embeddings.npy.

New in this run (the only allowed deviation, per protocol):
  - hidden states extracted at 7 candidate layers (0,4,8,12,16,20,24) instead
    of only the passive-probing-best layer
  - a ridge probe trained independently at each candidate layer (identical
    procedure/alpha grid to probe_train_generic.py) -> per-layer d_sem
  - per-layer pilot k-sweep over DENSE_K_GRID (reused verbatim from
    steering-variance-and-1b-recheck, the densest grid used anywhere in this
    project) and per-layer norm-relative alpha
  - per-layer 5-seed multiseed test evaluation and 95% CI (t-interval,
    matching analyze_multiseed.py / run_layer_sweep.py::run_aggregate)

Stages (run sequentially; each checkpoints to disk):
  --stage prep      : extract hidden states at 7 layers, train per-layer
                       probes, build d_sem + mean_hidden_norm(val) per layer
  --stage k_sweep    : per-layer pilot k-sweep -> selected k per STEERING layer
  --stage multiseed  : per-layer 5-seed test eval -> per_seed_margins.csv
  --stage aggregate  : layer_summary.csv, correlation stats, report.md
"""
import argparse
import csv
import gc
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge

REUSE_DIR = "/home/jkchoi/project/autopaper/sandbox/pythia-scaling-and-controls"
sys.path.insert(0, REUSE_DIR)

from common import load_model_and_tokenizer, read_jsonl  # noqa: E402
from embed_continuations import embed_texts, get_embedder  # noqa: E402
from steering_utils_generic import BatchedInjectionHook  # noqa: E402
import steering_generic as SG  # noqa: E402

OUT_DIR = "/home/jkchoi/project/autopaper/sandbox/p3-pythia1.4b-steering-layer-sweep"
ART_DIR = os.path.join(OUT_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

MODEL_NAME = "EleutherAI/pythia-1.4b"
MDIR = f"{REUSE_DIR}/artifacts/pythia-1.4b/greedy"
SPLIT_PATH = f"{REUSE_DIR}/artifacts/split.json"

LAYERS_PASSIVE = [0, 4, 8, 12, 16, 20, 24]          # raw hidden_states index; for the passive-probe table
LAYERS_STEER = [1, 4, 8, 12, 16, 20, 24]            # layer 0 -> 1 substitution for injection (documented above)
PASSIVE_TO_STEER = {0: 1, 4: 4, 8: 8, 12: 12, 16: 16, 20: 20, 24: 24}

ALPHA_GRID_PROBE = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
DENSE_K_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6, 2.4, 3.2, 4.8, 6.4, 8.0]

TUNE_SEED = 42
N_PILOT = 40
N_TEST = 60
BATCH_SIZE = 20
EXTRACT_BATCH_SIZE = 24
SEEDS = [101, 202, 303, 404, 505]
PROBE_SEED = 42

REFERENCE_LAYER20 = {"mean_margin": 0.0403575612232089, "ci95_low": 0.03348164875060359,
                      "ci95_high": 0.04723347369581422, "best_k": 6.4}


def log(msg):
    print(f"[p3] {msg}", flush=True)


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


def load_sentiment_axis():
    embedder = get_embedder()
    pos_emb = embedder.encode(SG.POSITIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    neg_emb = embedder.encode(SG.NEGATIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    pos_mean, neg_mean = pos_emb.mean(axis=0), neg_emb.mean(axis=0)
    v_sem = pos_mean - neg_mean
    v_sem = v_sem / (np.linalg.norm(v_sem) + 1e-12)
    return pos_mean, neg_mean, v_sem


def cosine_rows(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def train_probe_at_layer(X, y, idx_train, idx_val, idx_test):
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

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


# ---------------------------------------------------------------------------
def extract_hidden_states(model, tokenizer, continuations, device, layers):
    n = len(continuations)
    hidden_dim = model.config.hidden_size
    out_tensors = {L: np.zeros((n, hidden_dim), dtype=np.float32) for L in layers}

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
                for L in layers:
                    out_tensors[L][orig_idx] = hs[L][r, last_pos, :].float().cpu().numpy()
            if b % 5 == 0 or b == n_batches - 1:
                log(f"  hidden extract batch {b + 1}/{n_batches} ({time.time() - t0:.1f}s)")
    return out_tensors


def run_prep():
    split = json.load(open(SPLIT_PATH))
    idx_train, idx_val, idx_test = split["train"], split["val"], split["test"]
    continuations = read_jsonl(f"{MDIR}/continuations.jsonl")
    y = np.load(f"{MDIR}/continuation_embeddings.npy")
    assert y.shape[0] == len(continuations)

    device = "cuda"
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    log(f"model loaded; n_layers={n_layers} hidden_size={hidden_size}")
    assert 24 in LAYERS_PASSIVE and n_layers == 24, f"expected n_layers=24, got {n_layers}"

    hidden_by_layer = extract_hidden_states(model, tokenizer, continuations, device, LAYERS_PASSIVE)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    pos_mean, neg_mean, v_sem = load_sentiment_axis()

    diagnostics = []
    layer_data = {}
    for L in LAYERS_PASSIVE:
        X = hidden_by_layer[L]
        W, diag = train_probe_at_layer(X, y, idx_train, idx_val, idx_test)
        d_sem = W.T @ v_sem
        d_sem = d_sem / (np.linalg.norm(d_sem) + 1e-12)
        hs_val = X[idx_val]
        mean_norm = float(np.linalg.norm(hs_val, axis=1).mean())
        layer_data[L] = {"d_sem": d_sem, "mean_hidden_norm": mean_norm}
        diagnostics.append({
            "layer": L, "probe_val_cosine": diag["val_cosine"], "probe_test_cosine": diag["test_cosine"],
            "probe_best_alpha": diag["best_alpha"], "mean_hidden_norm_val": mean_norm,
        })
        log(f"layer={L} probe_val_cos={diag['val_cosine']:.4f} probe_test_cos={diag['test_cosine']:.4f} "
            f"mean_hidden_norm(val)={mean_norm:.3f}")

    np.savez(f"{ART_DIR}/layer_directions.npz", **{f"d_sem_layer{L}": layer_data[L]["d_sem"] for L in LAYERS_PASSIVE})
    with open(f"{ART_DIR}/layer_prep_meta.json", "w") as f:
        json.dump({
            "layers_passive": LAYERS_PASSIVE, "layers_steer": LAYERS_STEER,
            "passive_to_steer_map": PASSIVE_TO_STEER,
            "mean_hidden_norm": {str(L): layer_data[L]["mean_hidden_norm"] for L in LAYERS_PASSIVE},
            "probe_diagnostics": diagnostics,
        }, f, indent=2)
    with open(f"{ART_DIR}/model_config.json", "w") as f:
        json.dump({"model_name": MODEL_NAME, "n_layers": n_layers, "hidden_size": hidden_size}, f, indent=2)
    write_csv(f"{ART_DIR}/passive_probe_table.csv", diagnostics,
              ["layer", "probe_val_cosine", "probe_test_cosine", "probe_best_alpha", "mean_hidden_norm_val"])
    log("prep stage done")


def load_layer_data():
    npz = np.load(f"{ART_DIR}/layer_directions.npz")
    meta = json.load(open(f"{ART_DIR}/layer_prep_meta.json"))
    layer_data = {}
    for L in LAYERS_PASSIVE:
        layer_data[L] = {"d_sem": npz[f"d_sem_layer{L}"], "mean_hidden_norm": meta["mean_hidden_norm"][str(L)]}
    return layer_data


# ---------------------------------------------------------------------------
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
    for L_steer in LAYERS_STEER:
        # steering direction/mean-norm come from the PASSIVE-layer key that maps to this steer layer
        L_passive = [p for p, s in PASSIVE_TO_STEER.items() if s == L_steer][0]
        d_sem = layer_data[L_passive]["d_sem"]
        mean_norm = layer_data[L_passive]["mean_hidden_norm"]
        hidden_dim = d_sem.shape[0]
        log(f"=== steer layer={L_steer} (passive source layer={L_passive}) mean_hidden_norm(val)={mean_norm:.3f} ===")

        eligible_pilot = select_eligible(idx_val, continuations, TUNE_SEED, N_PILOT)
        pilot_ids_list = build_prompt_ids(eligible_pilot, continuations, tokenizer)
        d_rand_pilot = build_rand_dirs(eligible_pilot, continuations, hidden_dim, seed_offset=0)

        hook = BatchedInjectionHook(model, L_steer)
        base_pilot_texts = generate_all(model, tokenizer, hook, pilot_ids_list, None, device, eos_id)
        score_base_pilot = SG.axis_scores(base_pilot_texts, pos_mean, neg_mean)

        layer_rows = []
        for k in DENSE_K_GRID:
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
                "layer": L_steer, "k": k, "alpha": alpha,
                "pilot_margin_mean": margin_mean, "pilot_margin_std": margin_std,
                "n_examples": len(eligible_pilot), "selected_for_layer": False,
            })
            log(f"  k={k:<6} alpha={alpha:9.3f} margin_mean={margin_mean:+.5f} margin_std={margin_std:.5f}")

        hook.remove()

        best_row = max(layer_rows, key=lambda r: r["pilot_margin_mean"])
        best_row["selected_for_layer"] = True
        selection[L_steer] = {"selected_k": best_row["k"], "mean_hidden_norm": mean_norm,
                               "final_alpha": best_row["k"] * mean_norm, "pilot_margin_at_best_k": best_row["pilot_margin_mean"]}
        all_rows.extend(layer_rows)
        log(f"steer layer={L_steer}: selected_k={best_row['k']} pilot_margin={best_row['pilot_margin_mean']:+.5f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    write_csv(f"{ART_DIR}/k_sweep_results.csv", all_rows,
              ["layer", "k", "alpha", "pilot_margin_mean", "pilot_margin_std", "n_examples", "selected_for_layer"])
    alpha_rows = [{"layer": L, "selected_k": selection[L]["selected_k"],
                   "mean_hidden_norm_val": selection[L]["mean_hidden_norm"], "final_alpha": selection[L]["final_alpha"]}
                  for L in LAYERS_STEER]
    write_csv(f"{ART_DIR}/selected_alphas.csv", alpha_rows, ["layer", "selected_k", "mean_hidden_norm_val", "final_alpha"])
    with open(f"{ART_DIR}/k_sweep_selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    log("k_sweep stage done")


# ---------------------------------------------------------------------------
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

    for L_steer in LAYERS_STEER:
        L_passive = [p for p, s in PASSIVE_TO_STEER.items() if s == L_steer][0]
        d_sem = layer_data[L_passive]["d_sem"]
        hidden_dim = d_sem.shape[0]
        sel = selection[str(L_steer)]
        best_k, alpha = sel["selected_k"], sel["final_alpha"]
        log(f"=== steer layer={L_steer} best_k={best_k} alpha={alpha:.3f} ===")

        seeds_needed = [s for s in SEEDS if (L_steer, s) not in done_keys]
        if not seeds_needed:
            log(f"layer {L_steer}: all seeds already present, skipping")
            continue

        hook = BatchedInjectionHook(model, L_steer)
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

            row = {"layer": L_steer, "seed": seed, "margin": margin, "semantic_score": effect_sem,
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


# ---------------------------------------------------------------------------
def run_aggregate():
    raw_path = f"{ART_DIR}/per_seed_margins.csv"
    rows = []
    with open(raw_path) as f:
        for r in csv.DictReader(f):
            r["layer"] = int(r["layer"])
            r["margin"] = float(r["margin"])
            rows.append(r)
    selection = json.load(open(f"{ART_DIR}/k_sweep_selection.json"))
    passive_meta = json.load(open(f"{ART_DIR}/layer_prep_meta.json"))

    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)

    summary = []
    for L in LAYERS_STEER:
        rs = by_layer.get(L, [])
        margins = np.array([r["margin"] for r in rs], dtype=float)
        n = len(margins)
        mean = float(np.mean(margins)) if n else float("nan")
        std = float(np.std(margins, ddof=1)) if n >= 2 else float("nan")
        if n >= 4:
            sem = std / math.sqrt(n)
            tcrit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low, ci_high = mean - tcrit * sem, mean + tcrit * sem
        else:
            ci_low, ci_high = float("nan"), float("nan")
        sel = selection[str(L)]
        summary.append({
            "layer": L, "n_seeds": n, "mean_margin": mean, "std_margin": std,
            "ci95_low": ci_low, "ci95_high": ci_high,
            "selected_k": sel["selected_k"], "final_alpha": sel["final_alpha"],
            "ci_excludes_zero": bool(n >= 4 and (ci_low > 0 or ci_high < 0)),
        })

    write_csv(f"{ART_DIR}/layer_summary.csv", summary,
              ["layer", "n_seeds", "mean_margin", "std_margin", "ci95_low", "ci95_high",
               "selected_k", "final_alpha", "ci_excludes_zero"])

    # passive probe cosine per LAYERS_PASSIVE (includes layer 0, not in steering set)
    passive_table = {d["layer"]: d for d in passive_meta["probe_diagnostics"]}

    # correlation: passive cosine vs steering margin, matched by steer layer's passive source
    matched_passive_cos, matched_margin = [], []
    for L in LAYERS_STEER:
        L_passive = [p for p, s in PASSIVE_TO_STEER.items() if s == L][0]
        s = next(x for x in summary if x["layer"] == L)
        if s["n_seeds"] >= 4:
            matched_passive_cos.append(passive_table[L_passive]["probe_test_cosine"])
            matched_margin.append(s["mean_margin"])
    pearson_r, pearson_p = stats.pearsonr(matched_passive_cos, matched_margin)
    spearman_rho, spearman_p = stats.spearmanr(matched_passive_cos, matched_margin)

    best = max(summary, key=lambda s: s["mean_margin"] if s["n_seeds"] >= 4 else -999)
    layer20 = next(s for s in summary if s["layer"] == 20)

    # paired seed-level difference: best vs layer 20 (same seeds by construction)
    paired_diff = None
    if best["layer"] != 20 and best["n_seeds"] == 5 and layer20["n_seeds"] == 5:
        m_best = {r["seed"]: r["margin"] for r in by_layer[best["layer"]]}
        m_20 = {r["seed"]: r["margin"] for r in by_layer[20]}
        diffs = np.array([m_best[s] - m_20[s] for s in SEEDS])
        d_mean = float(np.mean(diffs))
        d_std = float(np.std(diffs, ddof=1))
        d_sem = d_std / math.sqrt(len(diffs))
        tcrit = float(stats.t.ppf(0.975, df=len(diffs) - 1))
        paired_diff = {"mean": d_mean, "ci95_low": d_mean - tcrit * d_sem, "ci95_high": d_mean + tcrit * d_sem,
                        "excludes_zero": bool((d_mean - tcrit * d_sem) > 0 or (d_mean + tcrit * d_sem) < 0)}

    out = {
        "passive_table": [{"layer": L, "probe_test_cosine": passive_table[L]["probe_test_cosine"]} for L in LAYERS_PASSIVE],
        "steering_summary": summary,
        "layer0_to_1_substitution": "layer 0 has no injectable analogue (BatchedInjectionHook requires layer_idx>=1); steering sweep uses layer 1 in its place, passive table still reports true layer 0",
        "passive_best_layer": max(LAYERS_PASSIVE, key=lambda L: passive_table[L]["probe_test_cosine"]),
        "steering_best_layer": best["layer"],
        "reference_layer20_prior": REFERENCE_LAYER20,
        "layer20_this_run": layer20,
        "correlation": {"pearson_r": pearson_r, "pearson_p": pearson_p,
                         "spearman_rho": spearman_rho, "spearman_p": spearman_p,
                         "n_layers": len(matched_margin)},
        "paired_diff_best_vs_layer20": paired_diff,
    }
    with open(f"{ART_DIR}/p3_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log("AGGREGATE DONE.")
    print(json.dumps(out, indent=2, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "k_sweep", "multiseed", "aggregate"])
    args = ap.parse_args()
    {"prep": run_prep, "k_sweep": run_k_sweep, "multiseed": run_multiseed, "aggregate": run_aggregate}[args.stage]()


if __name__ == "__main__":
    main()

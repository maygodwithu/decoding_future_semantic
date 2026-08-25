"""
P2 - Horizon x Rollout Budget x Semantic Encoder Analysis.

Reuses artifacts/generations.jsonl + hidden_last_token.npz produced by
p2_generate.py (single 256-token greedy generation pass, no sentence-cut,
same 600-prompt corpus as every prior project). Primary analysis uses the
"common subset" of prompts whose realized continuation reaches every tested
horizon (realized_len >= 256, n=577/600) so horizon comparisons are on a
matched sample, per spec section 2's stated preference. Per-horizon n for
the full (non-common) filter is reported separately from metadata.json.

Probe target for horizon H = embedding of the first H generated tokens.
Rollout baseline for budget m = embedding of the first m generated tokens
(SAME generated trajectory as the probe target -- paired at the example
level, no separate stochastic rollout, per spec section 5).
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "/home/jkchoi/project/autopaper/sandbox/p2-horizon-rollout-encoder")
from p2_bootstrap import paired_bootstrap_ci

ROOT = "/home/jkchoi/project/autopaper/sandbox/p2-horizon-rollout-encoder"
ART = f"{ROOT}/artifacts"
LAYERS = [4, 8, 12, 20]
HORIZONS = [16, 48, 96, 192, 256]
ROLLOUT_M = [3, 5, 10, 20]
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
SEED = 42
N_BOOT = 10000
PYTHIA_MODEL = "EleutherAI/pythia-1.4b"
COMMON_HORIZON = 256  # subset floor: realized_len >= this for ALL horizons


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cosine_rows(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def fit_ridge_select_alpha(X_train, y_train, X_val, y_val, X_test, y_test):
    best_alpha, best_val_cos, best_model = None, -2.0, None
    for a in ALPHA_GRID:
        m = Ridge(alpha=a, random_state=SEED)
        m.fit(X_train, y_train)
        val_cos = float(np.mean(cosine_rows(m.predict(X_val), y_val)))
        if val_cos > best_val_cos:
            best_val_cos, best_alpha, best_model = val_cos, a, m
    pred_test = best_model.predict(X_test)
    test_cos_per_example = cosine_rows(pred_test, y_test)
    return {"best_alpha": best_alpha, "val_cosine": best_val_cos,
            "test_cosine": float(np.mean(test_cos_per_example)),
            "test_cosine_per_example": test_cos_per_example}


def main():
    t0 = time.time()
    log = []

    def p(msg):
        print(msg)
        log.append(msg)

    records = read_jsonl(f"{ART}/generations.jsonl")
    n_all = len(records)
    lens = np.array([r["realized_len"] for r in records])
    hidden_npz = np.load(f"{ART}/hidden_last_token.npz")
    hidden_by_layer_all = {L: hidden_npz[f"layer_{L}"] for L in LAYERS}
    p(f"[p2] loaded {n_all} records; n at each horizon: " +
      ", ".join(f"H={H}:{int((lens >= H).sum())}" for H in HORIZONS))

    common_idx = np.where(lens >= COMMON_HORIZON)[0]
    n_common = len(common_idx)
    p(f"[p2] common subset (realized_len >= {COMMON_HORIZON}): n={n_common}/{n_all}")

    common_records = [records[i] for i in common_idx]
    hidden_by_layer = {L: hidden_by_layer_all[L][common_idx] for L in LAYERS}

    from sklearn.model_selection import train_test_split
    idx_all = np.arange(n_common)
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.15, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15 / 0.85, random_state=SEED)
    p(f"[p2] split on common subset: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} seed={SEED}")

    tok = AutoTokenizer.from_pretrained(PYTHIA_MODEL)

    def decode_prefix(ids, k):
        return tok.decode(ids[:k], skip_special_tokens=True)

    texts_by_H = {H: [decode_prefix(r["gen_token_ids"], H) for r in common_records] for H in HORIZONS}
    texts_by_m = {m: [decode_prefix(r["gen_token_ids"], m) for r in common_records] for m in ROLLOUT_M}
    p(f"[p2] decoded {len(HORIZONS)} horizon-prefix text sets and {len(ROLLOUT_M)} rollout-prefix text sets")

    from sentence_transformers import SentenceTransformer

    def embed(model, texts, prefix=""):
        if prefix:
            texts = [prefix + t for t in texts]
        return model.encode(texts, batch_size=64, show_progress_bar=False,
                             normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    encoder_specs = {
        "MiniLM": {"hf_name": "sentence-transformers/all-MiniLM-L6-v2", "prefix": ""},
        "BGE": {"hf_name": "BAAI/bge-base-en-v1.5", "prefix": ""},
        "E5": {"hf_name": "intfloat/e5-base-v2", "prefix": "passage: "},
    }

    results = {"n_common": n_common, "n_at_horizon_full": {str(H): int((lens >= H).sum()) for H in HORIZONS},
               "split": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test), "seed": SEED},
               "encoders": {}}

    for ename, spec in encoder_specs.items():
        p(f"\n=== {ename} ({spec['hf_name']}, prefix={spec['prefix']!r}) ===")
        model = SentenceTransformer(spec["hf_name"], device="cuda")

        y_by_H = {H: embed(model, texts_by_H[H], spec["prefix"]) for H in HORIZONS}
        rollout_by_m = {m: embed(model, texts_by_m[m], spec["prefix"]) for m in ROLLOUT_M}
        p(f"[p2] {ename}: encoded all horizon targets + rollout prefixes")

        enc_result = {"by_horizon": {}}

        for H in HORIZONS:
            y = y_by_H[H]
            y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

            layer_fits = {}
            best_layer, best_val_cos = None, -2.0
            for L in LAYERS:
                X = hidden_by_layer[L]
                fit = fit_ridge_select_alpha(X[idx_train], y_train, X[idx_val], y_val, X[idx_test], y_test)
                layer_fits[L] = fit
                if fit["val_cosine"] > best_val_cos:
                    best_val_cos, best_layer = fit["val_cosine"], L
            probe_test_per_ex = layer_fits[best_layer]["test_cosine_per_example"]
            probe_cos = float(np.mean(probe_test_per_ex))
            p(f"[p2] {ename} H={H}: best_layer={best_layer} probe_cos={probe_cos:.4f} "
              f"(layer sweep: " + ", ".join(f"L{L}={layer_fits[L]['test_cosine']:.4f}" for L in LAYERS) + ")")

            rollout_stats = {}
            for m in ROLLOUT_M:
                r_test = rollout_by_m[m][idx_test]
                y_test_H = y_test
                s_per_ex = cosine_rows(r_test, y_test_H)
                s_mean = float(np.mean(s_per_ex))
                delta_ci = paired_bootstrap_ci(probe_test_per_ex, s_per_ex, n_boot=N_BOOT, seed=SEED)
                rollout_stats[m] = {
                    "rollout_cosine": s_mean,
                    "rollout_cosine_per_example": s_per_ex,
                    "delta": delta_ci,
                    "efficiency_per_token": s_mean / m,
                }
                p(f"[p2] {ename} H={H} m={m}: rollout_cos={s_mean:.4f} "
                  f"delta(probe-rollout)={delta_ci['point']:.4f} 95%CI=[{delta_ci['ci_low']:.4f},{delta_ci['ci_high']:.4f}] "
                  f"excl0={delta_ci['excludes_zero']}")

            # marginal gain per added token between consecutive rollout budgets
            marginal = {}
            for m_prev, m_cur in zip(ROLLOUT_M[:-1], ROLLOUT_M[1:]):
                g = (rollout_stats[m_cur]["rollout_cosine"] - rollout_stats[m_prev]["rollout_cosine"]) / (m_cur - m_prev)
                marginal[f"{m_prev}->{m_cur}"] = g

            # rollout parity budget: smallest tested m with rollout_cosine >= probe_cosine
            parity = None
            for m in ROLLOUT_M:
                if rollout_stats[m]["rollout_cosine"] >= probe_cos:
                    parity = m
                    break
            parity_label = str(parity) if parity is not None else ">20"

            enc_result["by_horizon"][H] = {
                "best_layer": best_layer,
                "probe_cosine": probe_cos,
                "layer_sweep": {L: layer_fits[L]["test_cosine"] for L in LAYERS},
                "rollout": {m: {"rollout_cosine": rollout_stats[m]["rollout_cosine"],
                                 "delta": rollout_stats[m]["delta"],
                                 "efficiency_per_token": rollout_stats[m]["efficiency_per_token"]}
                            for m in ROLLOUT_M},
                "marginal_gain_per_token": marginal,
                "rollout_parity_budget": parity_label,
                # keep per-example arrays only in-memory summary form (not json-serialized) for size
            }
            p(f"[p2] {ename} H={H}: rollout parity budget = {parity_label}, marginal gains = " +
              ", ".join(f"{k}:{v:.5f}" for k, v in marginal.items()))

        results["encoders"][ename] = enc_result
        del model
        torch.cuda.empty_cache()

    with open(f"{ART}/p2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{ART}/run_log.txt", "w") as f:
        f.write("\n".join(log))

    p(f"\n[p2] done in {time.time() - t0:.1f}s. Saved {ART}/p2_results.json")


if __name__ == "__main__":
    main()

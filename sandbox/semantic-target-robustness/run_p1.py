"""
P1 - Semantic Target Robustness.

Reuses (read-only) the accepted hidden-state-semantic-lookahead run's
artifacts: generated.jsonl, split_indices.json, hidden_states_layer_{4,8,12,20}.pt,
continuation_embeddings.npy (MiniLM, E1 anchor). No new Pythia generation or
hidden-state extraction is performed. Tests whether semantic recoverability
holds when the target embedding space is BGE-base or E5-base instead of MiniLM.
"""
import json
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, "/home/jkchoi/project/autopaper/sandbox/semantic-target-robustness")
from p1_bootstrap import paired_bootstrap_ci

ANCHOR = "/home/jkchoi/project/autopaper/sandbox/hidden-state-semantic-lookahead/artifacts"
OUT = "/home/jkchoi/project/autopaper/sandbox/semantic-target-robustness/artifacts"
LAYERS = [4, 8, 12, 20]
LOGIT_LENS_LAYER = 20  # matches 4.2's control layer (probe's best layer for MiniLM)
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
SEED = 42
PYTHIA_MODEL = "EleutherAI/pythia-1.4b"
TOPK_VALUES = [1, 5, 10]
N_BOOT = 10000

import os
os.makedirs(OUT, exist_ok=True)


def cosine_rows(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def retrieval_metrics(query_emb, gallery_emb):
    q = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
    g = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-12)
    sim = q @ g.T
    n_q = sim.shape[0]
    ranks, r1, r5 = [], 0, 0
    for i in range(n_q):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        r1 += rank == 1
        r5 += rank <= 5
    return {"recall_at_1": r1 / n_q, "recall_at_5": r5 / n_q, "mean_rank": float(np.mean(ranks)), "n_query": n_q}


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
    return {
        "best_alpha": best_alpha, "val_cosine": best_val_cos,
        "test_cosine": float(np.mean(test_cos_per_example)),
        "test_cosine_per_example": test_cos_per_example,
        "pred_test": pred_test,
    }


def build_topk_texts(tokenizer, logits_row, k):
    probs = torch.softmax(logits_row, dim=-1)
    topp, topi = torch.topk(probs, k)
    toks = [tokenizer.decode([tid]) for tid in topi.tolist()]
    concat_text = "".join(toks).strip() or " ".join(t.strip() for t in toks)
    weighted_parts = []
    for tok, p in zip(toks, topp.tolist()):
        reps = max(1, round(10 * p))
        weighted_parts.extend([tok.strip() or tok] * reps)
    weighted_text = " ".join(weighted_parts)
    return concat_text if concat_text.strip() else "(empty)", weighted_text if weighted_text.strip() else "(empty)"


def main():
    t0 = time.time()
    log = []

    def p(msg):
        print(msg)
        log.append(msg)

    # ---------------- Step 1: load anchor data ----------------
    records = [json.loads(l) for l in open(f"{ANCHOR}/generated.jsonl")]
    n = len(records)
    split = json.load(open(f"{ANCHOR}/split_indices.json"))
    idx_train, idx_val, idx_test = split["train"], split["val"], split["test"]
    p(f"[p1] loaded {n} records, split train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    prompts = [r["prompt"] for r in records]
    continuation_texts = [r["continuation_text"] for r in records]

    hidden_by_layer = {L: torch.load(f"{ANCHOR}/hidden_states_layer_{L}.pt").numpy() for L in LAYERS}
    for L in LAYERS:
        assert hidden_by_layer[L].shape[0] == n

    # rollout texts (decode existing continuation_token_ids prefixes, no new generation)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PYTHIA_MODEL)
    rollout_m3, rollout_m5 = [], []
    for r in records:
        ids = r["continuation_token_ids"]
        rollout_m3.append(tok.decode(ids[:3], skip_special_tokens=True))
        rollout_m5.append(tok.decode(ids[:5], skip_special_tokens=True))
    p(f"[p1] built rollout m=3/m=5 texts from existing continuation_token_ids")

    # ---------------- logit-lens top-k texts (encoder-independent) ----------------
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(PYTHIA_MODEL, dtype=torch.float16).to("cuda")
    lm.eval()
    hidden_test_L20 = torch.tensor(hidden_by_layer[LOGIT_LENS_LAYER][idx_test], dtype=torch.float16, device="cuda")
    with torch.no_grad():
        normed = lm.gpt_neox.final_layer_norm(hidden_test_L20)
        logits = lm.get_output_embeddings()(normed).float().cpu()
    logit_lens_variants = {}
    for k in TOPK_VALUES:
        concat_texts, weighted_texts = [], []
        for row in range(logits.shape[0]):
            c, w = build_topk_texts(tok, logits[row], k)
            concat_texts.append(c)
            weighted_texts.append(w)
        logit_lens_variants[f"topk_concat_k{k}"] = concat_texts
        logit_lens_variants[f"topk_weighted_k{k}"] = weighted_texts
    del lm
    torch.cuda.empty_cache()
    p(f"[p1] built 6 logit-lens variant text sets (layer {LOGIT_LENS_LAYER}, {len(idx_test)} test examples)")

    # ---------------- Step 2: encoders ----------------
    from sentence_transformers import SentenceTransformer

    def embed(model, texts, prefix=""):
        if prefix:
            texts = [prefix + t for t in texts]
        return model.encode(texts, batch_size=64, show_progress_bar=False,
                             normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    encoder_specs = {
        "E1_MiniLM": {"hf_name": "sentence-transformers/all-MiniLM-L6-v2", "prefix": ""},
        "E2_BGE": {"hf_name": "BAAI/bge-base-en-v1.5", "prefix": ""},
        "E3_E5": {"hf_name": "intfloat/e5-base-v2", "prefix": "passage: "},
    }

    results = {}
    for ename, spec in encoder_specs.items():
        p(f"\n=== {ename} ({spec['hf_name']}, prefix={spec['prefix']!r}) ===")
        if ename == "E1_MiniLM":
            y = np.load(f"{ANCHOR}/continuation_embeddings.npy")
            p(f"[p1] {ename}: reused existing continuation_embeddings.npy shape={y.shape}")
            model = None
        else:
            model = SentenceTransformer(spec["hf_name"], device="cuda")
            y = embed(model, continuation_texts, spec["prefix"])
            p(f"[p1] {ename}: encoded {len(continuation_texts)} continuations, shape={y.shape}")
            np.save(f"{OUT}/continuation_embeddings_{ename}.npy", y)

        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

        # ---- Step 4: layer sweep ----
        layer_results = {}
        best_layer, best_val_cos = None, -2.0
        for L in LAYERS:
            X = hidden_by_layer[L]
            fit = fit_ridge_select_alpha(X[idx_train], y_train, X[idx_val], y_val, X[idx_test], y_test)
            layer_results[L] = fit
            p(f"[p1] {ename} layer {L}: alpha={fit['best_alpha']} val_cos={fit['val_cosine']:.4f} test_cos={fit['test_cosine']:.4f}")
            if fit["val_cosine"] > best_val_cos:
                best_val_cos, best_layer = fit["val_cosine"], L
        p(f"[p1] {ename} best layer by val cosine: {best_layer} (test_cos={layer_results[best_layer]['test_cosine']:.4f})")

        probe_test_cos_per_ex = layer_results[best_layer]["test_cosine_per_example"]
        probe_cos = float(np.mean(probe_test_cos_per_ex))

        # ---- Step 5: weak baselines ----
        mean_train = y_train.mean(axis=0, keepdims=True)
        mean_pred = np.repeat(mean_train, len(idx_test), axis=0)
        mean_cos_per_ex = cosine_rows(mean_pred, y_test)

        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(idx_test))
        for i in range(len(perm)):
            if perm[i] == i:
                j = (i + 1) % len(perm)
                perm[i], perm[j] = perm[j], perm[i]
        random_cos_per_ex = cosine_rows(y_test, y_test[perm])

        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        Xtf_train = tfidf.fit_transform([prompts[i] for i in idx_train]).toarray()
        Xtf_val = tfidf.transform([prompts[i] for i in idx_val]).toarray()
        Xtf_test = tfidf.transform([prompts[i] for i in idx_test]).toarray()
        tfidf_fit = fit_ridge_select_alpha(Xtf_train, y_train, Xtf_val, y_val, Xtf_test, y_test)
        lexical_cos_per_ex = tfidf_fit["test_cosine_per_example"]

        p(f"[p1] {ename} mean_cos={np.mean(mean_cos_per_ex):.4f} random_cos={np.mean(random_cos_per_ex):.4f} "
          f"lexical_cos={np.mean(lexical_cos_per_ex):.4f}")

        # ---- logit-lens baseline (embed under this encoder) ----
        y_test_only = y_test
        best_logitlens_name, best_logitlens_cos = None, -2.0
        best_logitlens_per_ex = None
        for vname, vtexts in logit_lens_variants.items():
            if ename == "E1_MiniLM":
                emb_model = SentenceTransformer(encoder_specs["E1_MiniLM"]["hf_name"], device="cuda")
            else:
                emb_model = model
            v_emb = embed(emb_model, vtexts, spec["prefix"])
            v_cos_per_ex = cosine_rows(v_emb, y_test_only)
            v_cos = float(np.mean(v_cos_per_ex))
            if v_cos > best_logitlens_cos:
                best_logitlens_cos, best_logitlens_name = v_cos, vname
                best_logitlens_per_ex = v_cos_per_ex
            if ename == "E1_MiniLM":
                del emb_model
        p(f"[p1] {ename} best logit-lens variant: {best_logitlens_name} cos={best_logitlens_cos:.4f}")

        # ---- rollout baselines ----
        if ename == "E1_MiniLM":
            embA = SentenceTransformer(encoder_specs["E1_MiniLM"]["hf_name"], device="cuda")
        else:
            embA = model
        rollout3_emb = embed(embA, [rollout_m3[i] for i in idx_test], spec["prefix"])
        rollout5_emb = embed(embA, [rollout_m5[i] for i in idx_test], spec["prefix"])
        rollout3_cos_per_ex = cosine_rows(rollout3_emb, y_test_only)
        rollout5_cos_per_ex = cosine_rows(rollout5_emb, y_test_only)
        p(f"[p1] {ename} rollout m=3 cos={np.mean(rollout3_cos_per_ex):.4f} m=5 cos={np.mean(rollout5_cos_per_ex):.4f}")
        if ename == "E1_MiniLM":
            del embA
            torch.cuda.empty_cache()

        # ---- Step 6: retrieval ----
        retrieval = retrieval_metrics(layer_results[best_layer]["pred_test"], y_test)
        p(f"[p1] {ename} retrieval R@1={retrieval['recall_at_1']:.4f} R@5={retrieval['recall_at_5']:.4f} "
          f"mean_rank={retrieval['mean_rank']:.2f} (chance R@5={min(5/len(idx_test),1.0):.4f})")

        # ---- Step 7: shuffled-target control (E1, E2 minimum) ----
        shuffled_result = None
        if True:  # run shuffled-target control for all three encoders (cheap)
            perm_seed = SEED
            rng2 = np.random.RandomState(perm_seed)
            shuf_perm = rng2.permutation(len(idx_train))
            X_best = hidden_by_layer[best_layer]
            y_train_shuf = y_train[shuf_perm]
            shuf_fit = fit_ridge_select_alpha(X_best[idx_train], y_train_shuf, X_best[idx_val], y_val, X_best[idx_test], y_test)
            shuffled_result = {"perm_seed": perm_seed, "normal_cosine": probe_cos, "shuffled_cosine": shuf_fit["test_cosine"]}
            p(f"[p1] {ename} shuffled-target control: normal={probe_cos:.4f} shuffled={shuf_fit['test_cosine']:.4f} "
              f"diff={probe_cos - shuf_fit['test_cosine']:.4f} (perm_seed={perm_seed})")

        # ---- Step 8: bootstrap CIs ----
        best_weak_or_token_per_ex = mean_cos_per_ex
        best_weak_or_token_name = "mean_embedding"
        for name, arr in [("random_match", random_cos_per_ex), ("lexical_tfidf", lexical_cos_per_ex),
                           ("logit_lens", best_logitlens_per_ex)]:
            if np.mean(arr) > np.mean(best_weak_or_token_per_ex):
                best_weak_or_token_per_ex, best_weak_or_token_name = arr, name

        boot = {
            "probe_cosine": paired_bootstrap_ci(probe_test_cos_per_ex, n_boot=N_BOOT, seed=SEED),
            f"probe_minus_{best_weak_or_token_name}": paired_bootstrap_ci(
                probe_test_cos_per_ex, best_weak_or_token_per_ex, n_boot=N_BOOT, seed=SEED),
            "probe_minus_rollout_m3": paired_bootstrap_ci(probe_test_cos_per_ex, rollout3_cos_per_ex, n_boot=N_BOOT, seed=SEED),
            "probe_minus_rollout_m5": paired_bootstrap_ci(probe_test_cos_per_ex, rollout5_cos_per_ex, n_boot=N_BOOT, seed=SEED),
        }
        p(f"[p1] {ename} bootstrap probe_cosine 95% CI = [{boot['probe_cosine']['ci_low']:.4f}, {boot['probe_cosine']['ci_high']:.4f}]")
        p(f"[p1] {ename} bootstrap probe - {best_weak_or_token_name} margin 95% CI = "
          f"[{boot[f'probe_minus_{best_weak_or_token_name}']['ci_low']:.4f}, {boot[f'probe_minus_{best_weak_or_token_name}']['ci_high']:.4f}]")

        results[ename] = {
            "hf_name": spec["hf_name"], "prefix": spec["prefix"],
            "layer_sweep": {L: {"alpha": layer_results[L]["best_alpha"], "test_cosine": layer_results[L]["test_cosine"]} for L in LAYERS},
            "best_layer": best_layer,
            "probe_cosine": probe_cos,
            "mean_cosine": float(np.mean(mean_cos_per_ex)),
            "random_cosine": float(np.mean(random_cos_per_ex)),
            "lexical_cosine": float(np.mean(lexical_cos_per_ex)),
            "best_logitlens_variant": best_logitlens_name,
            "best_logitlens_cosine": best_logitlens_cos,
            "rollout_m3_cosine": float(np.mean(rollout3_cos_per_ex)),
            "rollout_m5_cosine": float(np.mean(rollout5_cos_per_ex)),
            "retrieval": retrieval,
            "shuffled_target": shuffled_result,
            "bootstrap": boot,
        }

        if model is not None:
            del model
            torch.cuda.empty_cache()

    with open(f"{OUT}/p1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{OUT}/run_log.txt", "w") as f:
        f.write("\n".join(log))

    p(f"\n[p1] done in {time.time() - t0:.1f}s. Saved {OUT}/p1_results.json")


if __name__ == "__main__":
    main()

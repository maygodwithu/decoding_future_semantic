"""
Steps 6-7: train/val/test split, per-layer ridge probes (hidden state ->
continuation sentence embedding), baselines, and retrieval evaluation for the
best layer.

Outputs:
  artifacts/split_indices.json
  artifacts/probe_results.json
  artifacts/test_predictions_layer_{L}.npy   (for each candidate layer)
  artifacts/retrieval_results.json
  artifacts/best_probe_trainval_layer_{L}.npz  (refit on train+val, for steering)
"""
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch

GENERATED_PATH = "artifacts/generated.jsonl"
EMB_PATH = "artifacts/continuation_embeddings.npy"
LAYERS = [4, 8, 12, 20]
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
SEED = 42


def cosine_rows(a, b):
    """Row-wise cosine similarity between two [N,D] arrays."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def main():
    records = [json.loads(l) for l in open(GENERATED_PATH)]
    n = len(records)
    emb = np.load(EMB_PATH)
    assert emb.shape[0] == n

    idx_all = np.arange(n)
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.15, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15 / 0.85, random_state=SEED)

    split = {
        "train": idx_train.tolist(),
        "val": idx_val.tolist(),
        "test": idx_test.tolist(),
        "seed": SEED,
        "n_total": n,
    }
    with open("artifacts/split_indices.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"[probe] split sizes: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    y_train, y_val, y_test = emb[idx_train], emb[idx_val], emb[idx_test]

    results = {"layers": {}, "baselines": {}, "config": {"alpha_grid": ALPHA_GRID, "layers": LAYERS}}

    # ---------------- per-layer ridge probes ----------------
    best_layer, best_val_cos = None, -2.0
    for L in LAYERS:
        X = torch.load(f"artifacts/hidden_states_layer_{L}.pt").numpy()
        assert X.shape[0] == n
        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

        best_alpha, best_a_val_cos, best_model = None, -2.0, None
        for a in ALPHA_GRID:
            model = Ridge(alpha=a, random_state=SEED)
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            val_cos = float(np.mean(cosine_rows(pred_val, y_val)))
            if val_cos > best_a_val_cos:
                best_a_val_cos, best_alpha, best_model = val_cos, a, model

        pred_train = best_model.predict(X_train)
        pred_test = best_model.predict(X_test)
        train_cos = float(np.mean(cosine_rows(pred_train, y_train)))
        test_cos = float(np.mean(cosine_rows(pred_test, y_test)))
        test_cos_per_example = cosine_rows(pred_test, y_test).tolist()

        np.save(f"artifacts/test_predictions_layer_{L}.npy", pred_test.astype(np.float32))

        results["layers"][str(L)] = {
            "best_alpha": best_alpha,
            "train_cosine": train_cos,
            "val_cosine": best_a_val_cos,
            "test_cosine": test_cos,
            "test_cosine_per_example": test_cos_per_example,
        }
        print(f"[probe] layer {L}: alpha={best_alpha} train_cos={train_cos:.4f} "
              f"val_cos={best_a_val_cos:.4f} test_cos={test_cos:.4f}")

        if best_a_val_cos > best_val_cos:
            best_val_cos, best_layer = best_a_val_cos, L

    results["best_layer_by_val"] = best_layer
    print(f"[probe] best layer by validation cosine: {best_layer}")

    # ---------------- baselines (layer-independent) ----------------
    mean_train_emb = y_train.mean(axis=0, keepdims=True)
    mean_pred_test = np.repeat(mean_train_emb, len(idx_test), axis=0)
    mean_cos = cosine_rows(mean_pred_test, y_test)
    results["baselines"]["mean_embedding"] = {
        "test_cosine": float(np.mean(mean_cos)),
        "test_cosine_per_example": mean_cos.tolist(),
    }

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(idx_test))
    # avoid trivial self-matches where possible
    for i in range(len(perm)):
        if perm[i] == i:
            j = (i + 1) % len(perm)
            perm[i], perm[j] = perm[j], perm[i]
    random_match_cos = cosine_rows(y_test, y_test[perm])
    results["baselines"]["random_match"] = {
        "test_cosine": float(np.mean(random_match_cos)),
        "test_cosine_per_example": random_match_cos.tolist(),
    }

    # ---------------- optional lexical (TF-IDF) baseline ----------------
    prompts = [r["prompt"] for r in records]
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    Xtf_train = tfidf.fit_transform([prompts[i] for i in idx_train]).toarray()
    Xtf_val = tfidf.transform([prompts[i] for i in idx_val]).toarray()
    Xtf_test = tfidf.transform([prompts[i] for i in idx_test]).toarray()

    best_alpha_tf, best_val_cos_tf, best_model_tf = None, -2.0, None
    for a in ALPHA_GRID:
        model = Ridge(alpha=a, random_state=SEED)
        model.fit(Xtf_train, y_train)
        val_cos = float(np.mean(cosine_rows(model.predict(Xtf_val), y_val)))
        if val_cos > best_val_cos_tf:
            best_val_cos_tf, best_alpha_tf, best_model_tf = val_cos, a, model
    pred_test_tf = best_model_tf.predict(Xtf_test)
    tf_test_cos = cosine_rows(pred_test_tf, y_test)
    results["baselines"]["lexical_tfidf"] = {
        "best_alpha": best_alpha_tf,
        "val_cosine": best_val_cos_tf,
        "test_cosine": float(np.mean(tf_test_cos)),
        "test_cosine_per_example": tf_test_cos.tolist(),
    }
    print(f"[probe] lexical TF-IDF baseline: alpha={best_alpha_tf} test_cos={float(np.mean(tf_test_cos)):.4f}")
    print(f"[probe] mean-embedding baseline test_cos={results['baselines']['mean_embedding']['test_cosine']:.4f}")
    print(f"[probe] random-match baseline test_cos={results['baselines']['random_match']['test_cosine']:.4f}")

    with open("artifacts/probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[probe] saved artifacts/probe_results.json")

    # ---------------- retrieval evaluation for the best layer ----------------
    L = best_layer
    X = torch.load(f"artifacts/hidden_states_layer_{L}.pt").numpy()
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    alpha_best = results["layers"][str(L)]["best_alpha"]
    final_model = Ridge(alpha=alpha_best, random_state=SEED)
    final_model.fit(X_trainval, y_trainval)
    np.savez(
        f"artifacts/best_probe_trainval_layer_{L}.npz",
        W=final_model.coef_, b=final_model.intercept_, alpha=alpha_best, layer=L,
    )
    print(f"[probe] refit final probe on train+val for layer {L}, saved coefficients")

    pred_test_final = final_model.predict(X_test)

    def retrieval_metrics(query_emb, gallery_emb):
        # cosine similarity matrix [n_query, n_gallery]
        q = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
        g = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-12)
        sim = q @ g.T
        n_q = sim.shape[0]
        ranks = []
        r1 = r5 = 0
        for i in range(n_q):
            order = np.argsort(-sim[i])
            rank = int(np.where(order == i)[0][0]) + 1  # 1-indexed rank of the true match
            ranks.append(rank)
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
        return {
            "recall_at_1": r1 / n_q,
            "recall_at_5": r5 / n_q,
            "mean_rank": float(np.mean(ranks)),
            "n_query": n_q,
        }

    retrieval = {"best_layer": L, "n_test": len(idx_test)}
    retrieval["probe"] = retrieval_metrics(pred_test_final, y_test)

    mean_pred_query = np.repeat(mean_train_emb, len(idx_test), axis=0)
    retrieval["mean_embedding_baseline"] = retrieval_metrics(mean_pred_query, y_test)

    rng2 = np.random.RandomState(SEED + 1)
    random_query = rng2.normal(size=y_test.shape)
    retrieval["random_baseline"] = retrieval_metrics(random_query, y_test)
    retrieval["chance_recall_at_1"] = 1.0 / len(idx_test)
    retrieval["chance_recall_at_5"] = min(5.0 / len(idx_test), 1.0)

    with open("artifacts/retrieval_results.json", "w") as f:
        json.dump(retrieval, f, indent=2)
    print(f"[probe] retrieval (layer {L}): {retrieval['probe']}")
    print(f"[probe] retrieval mean-baseline: {retrieval['mean_embedding_baseline']}")
    print(f"[probe] retrieval random-baseline: {retrieval['random_baseline']}")
    print("[probe] saved artifacts/retrieval_results.json")


if __name__ == "__main__":
    main()

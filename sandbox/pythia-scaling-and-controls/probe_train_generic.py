"""
Generic version of the prior probe_train.py: trains per-layer ridge probes
(hidden state -> continuation sentence embedding) for a given
artifacts/{model_tag}/{decode_tag}/ directory, using the shared split.json.
Same alpha grid, same baselines (mean-embedding, random-match), same
retrieval procedure as the prior accepted run.

Outputs (written into --dir):
  probe_results.json
  test_predictions_layer_{L}.npy
  best_probe_trainval_layer_{L}.npz   (refit on train+val, for steering)
  retrieval_results.json
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import Ridge

from common import cosine_rows, retrieval_metrics, SEED

ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--split", default="artifacts/split.json")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    args = ap.parse_args()

    split = json.load(open(args.split))
    idx_train, idx_val, idx_test = split["train"], split["val"], split["test"]

    npz = np.load(f"{args.dir}/hidden_last_token.npz")
    emb = np.load(f"{args.dir}/continuation_embeddings.npy")
    n = emb.shape[0]

    layers = args.layers if args.layers else sorted(int(k.split("_")[1]) for k in npz.files)
    y_train, y_val, y_test = emb[idx_train], emb[idx_val], emb[idx_test]

    results = {"layers": {}, "baselines": {}, "config": {"alpha_grid": ALPHA_GRID, "layers": layers}}
    best_layer, best_val_cos = None, -2.0

    for L in layers:
        X = npz[f"layer_{L}"]
        assert X.shape[0] == n
        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

        best_alpha, best_a_val_cos, best_model = None, -2.0, None
        for a in ALPHA_GRID:
            model = Ridge(alpha=a, random_state=SEED)
            model.fit(X_train, y_train)
            val_cos = float(np.mean(cosine_rows(model.predict(X_val), y_val)))
            if val_cos > best_a_val_cos:
                best_a_val_cos, best_alpha, best_model = val_cos, a, model

        pred_train = best_model.predict(X_train)
        pred_test = best_model.predict(X_test)
        train_cos = float(np.mean(cosine_rows(pred_train, y_train)))
        test_cos = float(np.mean(cosine_rows(pred_test, y_test)))

        np.save(f"{args.dir}/test_predictions_layer_{L}.npy", pred_test.astype(np.float32))
        results["layers"][str(L)] = {
            "best_alpha": best_alpha, "train_cosine": train_cos,
            "val_cosine": best_a_val_cos, "test_cosine": test_cos,
        }
        print(f"[probe:{args.dir}] layer {L}: alpha={best_alpha} train_cos={train_cos:.4f} "
              f"val_cos={best_a_val_cos:.4f} test_cos={test_cos:.4f}")
        if best_a_val_cos > best_val_cos:
            best_val_cos, best_layer = best_a_val_cos, L

    results["best_layer_by_val"] = best_layer
    print(f"[probe:{args.dir}] best layer by val cosine: {best_layer}")

    # ---- baselines ----
    mean_train_emb = y_train.mean(axis=0, keepdims=True)
    mean_pred_test = np.repeat(mean_train_emb, len(idx_test), axis=0)
    mean_cos = cosine_rows(mean_pred_test, y_test)
    results["baselines"]["mean_embedding"] = {"test_cosine": float(np.mean(mean_cos))}

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(idx_test))
    for i in range(len(perm)):
        if perm[i] == i:
            j = (i + 1) % len(perm)
            perm[i], perm[j] = perm[j], perm[i]
    random_match_cos = cosine_rows(y_test, y_test[perm])
    results["baselines"]["random_match"] = {"test_cosine": float(np.mean(random_match_cos))}

    with open(f"{args.dir}/probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe:{args.dir}] mean_baseline={results['baselines']['mean_embedding']['test_cosine']:.4f} "
          f"random_match={results['baselines']['random_match']['test_cosine']:.4f}")

    # ---- refit best layer on train+val, retrieval eval ----
    L = best_layer
    X = npz[f"layer_{L}"]
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    alpha_best = results["layers"][str(L)]["best_alpha"]
    final_model = Ridge(alpha=alpha_best, random_state=SEED)
    final_model.fit(X_trainval, y_trainval)
    np.savez(f"{args.dir}/best_probe_trainval_layer_{L}.npz",
              W=final_model.coef_, b=final_model.intercept_, alpha=alpha_best, layer=L)

    pred_test_final = final_model.predict(X_test)
    retrieval = {"best_layer": L, "n_test": len(idx_test)}
    retrieval["probe"] = retrieval_metrics(pred_test_final, y_test)
    mean_pred_query = np.repeat(mean_train_emb, len(idx_test), axis=0)
    retrieval["mean_embedding_baseline"] = retrieval_metrics(mean_pred_query, y_test)
    rng2 = np.random.RandomState(SEED + 1)
    random_query = rng2.normal(size=y_test.shape)
    retrieval["random_baseline"] = retrieval_metrics(random_query, y_test)
    retrieval["chance_recall_at_5"] = min(5.0 / len(idx_test), 1.0)

    with open(f"{args.dir}/retrieval_results.json", "w") as f:
        json.dump(retrieval, f, indent=2)
    print(f"[probe:{args.dir}] retrieval (layer {L}) probe={retrieval['probe']}")


if __name__ == "__main__":
    main()

"""
Step 3: for each fixed-token horizon H in {16, 48, 96}, train the SAME probe
class as the prior accepted runs -- per-layer Ridge regression mapping the
single last-prompt hidden state (layers 4, 8, 12, 20) to the sentence
embedding of the full H-token continuation -- using the identical
train/val/test split (artifacts/split.json), the identical alpha grid, and
the identical val-cosine layer/alpha selection procedure as
probe_train_generic.py in the reused pipeline.

Outputs (artifacts/):
  probe_results_H{H}.json     per-layer alpha/train/val/test cosine + best layer
  test_predictions_H{H}.npy   compact test-set predictions at the best layer only
"""
import json

import numpy as np
from sklearn.linear_model import Ridge

from common import cosine_rows, SEED

ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
HORIZONS = [16, 48, 96]
LAYERS = [4, 8, 12, 20]


def main():
    split = json.load(open("artifacts/split.json"))
    idx_train, idx_val, idx_test = split["train"], split["val"], split["test"]
    npz = np.load("artifacts/hidden_last_token.npz")

    summary = {}
    for H in HORIZONS:
        emb = np.load(f"artifacts/continuation_embeddings_H{H}.npy")
        n = emb.shape[0]
        y_train, y_val, y_test = emb[idx_train], emb[idx_val], emb[idx_test]

        results = {"layers": {}, "config": {"alpha_grid": ALPHA_GRID, "layers": LAYERS}}
        best_layer, best_val_cos = None, -2.0
        best_test_pred = None

        for L in LAYERS:
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

            results["layers"][str(L)] = {
                "best_alpha": best_alpha, "train_cosine": train_cos,
                "val_cosine": best_a_val_cos, "test_cosine": test_cos,
            }
            print(f"[probe H={H}] layer {L}: alpha={best_alpha} train_cos={train_cos:.4f} "
                  f"val_cos={best_a_val_cos:.4f} test_cos={test_cos:.4f}")
            if best_a_val_cos > best_val_cos:
                best_val_cos, best_layer = best_a_val_cos, L
                best_test_pred = pred_test

        results["best_layer_by_val"] = best_layer
        results["best_layer_test_cosine"] = results["layers"][str(best_layer)]["test_cosine"]
        results["n_test"] = len(idx_test)
        print(f"[probe H={H}] best layer by val cosine: {best_layer} "
              f"test_cos={results['best_layer_test_cosine']:.4f}")

        with open(f"artifacts/probe_results_H{H}.json", "w") as f:
            json.dump(results, f, indent=2)
        np.save(f"artifacts/test_predictions_H{H}.npy", best_test_pred.astype(np.float32))

        summary[str(H)] = results

    with open("artifacts/probe_results_all.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[probe] done")


if __name__ == "__main__":
    main()

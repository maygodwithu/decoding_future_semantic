"""
Step 5: compute L2-normalized sentence embeddings of continuation texts using
sentence-transformers/all-MiniLM-L6-v2.

Saves artifacts/continuation_embeddings.npy: [N, 384] float32, aligned with
artifacts/generated.jsonl. Also saves artifacts/train_mean_embedding.npy once
the split is known (done in probe_train.py), so this script just produces the
raw embeddings.
"""
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

GENERATED_PATH = "artifacts/generated.jsonl"
OUT_PATH = "artifacts/continuation_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    t0 = time.time()
    records = [json.loads(l) for l in open(GENERATED_PATH)]
    texts = [r["continuation_text"] for r in records]
    print(f"[embed] encoding {len(texts)} continuations")

    model = SentenceTransformer(MODEL_NAME, device="cuda")
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb = emb.astype(np.float32)
    np.save(OUT_PATH, emb)
    print(f"[embed] saved {OUT_PATH} shape={emb.shape}")
    print(f"[embed] total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

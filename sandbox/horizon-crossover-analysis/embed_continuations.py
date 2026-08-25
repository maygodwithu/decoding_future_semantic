"""
Step 3: embed continuation texts with the same sentence embedder as the
prior run (sentence-transformers/all-MiniLM-L6-v2), normalized. Saves
continuation_embeddings.npy and the mean embedding of the TRAIN split's
continuations (for the mean-baseline).
"""
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from common import EMB_MODEL_NAME, read_jsonl

_MODEL_CACHE = {}


def get_embedder():
    if "m" not in _MODEL_CACHE:
        _MODEL_CACHE["m"] = SentenceTransformer(EMB_MODEL_NAME, device="cuda")
    return _MODEL_CACHE["m"]


def embed_texts(texts, batch_size=64):
    model = get_embedder()
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="artifacts/{model_tag}/{decode_tag}")
    ap.add_argument("--split", default="artifacts/split.json")
    args = ap.parse_args()

    records = read_jsonl(f"{args.dir}/continuations.jsonl")
    texts = [r["continuation_text"] for r in records]
    print(f"[embed] encoding {len(texts)} continuations from {args.dir}")
    emb = embed_texts(texts)
    np.save(f"{args.dir}/continuation_embeddings.npy", emb)
    print(f"[embed] saved {args.dir}/continuation_embeddings.npy shape={emb.shape}")

    split = json.load(open(args.split))
    train_idx = split["train"]
    mean_train = emb[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    np.save(f"{args.dir}/train_mean_embedding.npy", mean_train)
    print(f"[embed] saved {args.dir}/train_mean_embedding.npy")


if __name__ == "__main__":
    main()

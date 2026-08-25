"""
Step 2: from the single 96-token greedy generation (artifacts/generations.jsonl),
derive fixed-token-horizon continuation targets for H in {16, 48, 96} by
truncating each sample's generated token ids to the first H tokens (or the
realized shorter length if the model's generation naturally ended before H
tokens -- no EOS was hit inside the 96-token generation for a sample unless
recorded in generations.jsonl). No sentence-boundary cuts are applied here
(unlike the prior accepted runs) since the point of this experiment is
controlled fixed-token horizons.

Also derives short greedy-rollout PREFIXES for m in {3, 5, 10, 20}: the first
m generated tokens of the SAME generation (independent of horizon H). If
m > realized_len for a sample, the prefix is truncated to the available
generated tokens (documented rule, consistent with the horizon-target rule).

Both target texts and rollout-prefix texts are embedded with the same
sentence embedder as the prior accepted runs
(sentence-transformers/all-MiniLM-L6-v2, normalized), so
continuation_embeddings_H{H}.npy and rollout_embeddings_m{m}.npy are all in
the same embedding space and directly comparable via cosine similarity.

Outputs (artifacts/):
  targets/horizon_{H}.jsonl        {prompt_id, text, used_len}
  rollouts/rollout_m{m}.jsonl      {prompt_id, text, used_len}
  continuation_embeddings_H{H}.npy [N, 384] float32
  rollout_embeddings_m{m}.npy      [N, 384] float32
  horizon_stats.json               descriptive stats per horizon
"""
import json
import os

import numpy as np
from transformers import AutoTokenizer

from common import read_jsonl, write_jsonl
from embed_continuations import embed_texts

MODEL_NAME = "EleutherAI/pythia-1.4b"
HORIZONS = [16, 48, 96]
ROLLOUT_MS = [3, 5, 10, 20]


def decode_prefix(tokenizer, ids, k):
    sub = ids[:k]
    if not sub:
        return "(empty)", 0
    txt = tokenizer.decode(sub, skip_special_tokens=True).strip()
    return (txt if txt else "(empty)"), len(sub)


def main():
    os.makedirs("artifacts/targets", exist_ok=True)
    os.makedirs("artifacts/rollouts", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    records = read_jsonl("artifacts/generations.jsonl")
    n = len(records)
    print(f"[targets] {n} usable generations loaded")

    stats = {}
    for H in HORIZONS:
        texts, used_lens, char_lens, frac_short = [], [], [], []
        rows = []
        for r in records:
            ids = r["gen_token_ids"]
            txt, used = decode_prefix(tokenizer, ids, H)
            texts.append(txt)
            used_lens.append(used)
            char_lens.append(len(txt))
            frac_short.append(1 if used < H else 0)
            rows.append({"prompt_id": r["prompt_id"], "text": txt, "used_len": used})
        write_jsonl(f"artifacts/targets/horizon_{H}.jsonl", rows)
        print(f"[targets] horizon {H}: encoding {len(texts)} target texts")
        emb = embed_texts(texts)
        np.save(f"artifacts/continuation_embeddings_H{H}.npy", emb)
        stats[str(H)] = {
            "mean_used_len": float(np.mean(used_lens)),
            "median_used_len": float(np.median(used_lens)),
            "frac_shorter_than_H_due_to_eos_or_short_gen": float(np.mean(frac_short)),
            "char_len_mean": float(np.mean(char_lens)),
            "char_len_median": float(np.median(char_lens)),
            "n": n,
        }
        print(f"[targets] horizon {H} stats: {stats[str(H)]}")

    for m in ROLLOUT_MS:
        texts, used_lens = [], []
        rows = []
        for r in records:
            ids = r["gen_token_ids"]
            txt, used = decode_prefix(tokenizer, ids, m)
            texts.append(txt)
            used_lens.append(used)
            rows.append({"prompt_id": r["prompt_id"], "text": txt, "used_len": used})
        write_jsonl(f"artifacts/rollouts/rollout_m{m}.jsonl", rows)
        print(f"[targets] rollout m={m}: encoding {len(texts)} prefix texts "
              f"(mean used_len={np.mean(used_lens):.2f})")
        emb = embed_texts(texts)
        np.save(f"artifacts/rollout_embeddings_m{m}.npy", emb)

    with open("artifacts/horizon_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("[targets] done")


if __name__ == "__main__":
    main()

"""
Step 1 (horizon-crossover-analysis): single greedy generation pass on
EleutherAI/pythia-1.4b at max_new_tokens=96 (the longest tested horizon) for
all 600 reused prompts (artifacts/prompts.jsonl, same set as
hidden-state-semantic-lookahead / pythia-scaling-and-controls). Shorter
horizons (16, 48) are DERIVED by truncating this single 96-token generation
(nested-prefix property), per the protocol, to avoid duplicate generation
passes.

Also records, in the SAME prompt-only forward pass as the prior accepted
runs, the hidden state at the last PROMPT token (no generated tokens in
context) at layers 4, 8, 12, 20 (out of pythia-1.4b's 24 transformer blocks
-- exact prior candidate layers, best layer in the prior run was 20).

Usable-prompt filter (documented rule): keep prompts whose greedy
continuation (EOS-truncated if applicable) has realized length >= 16 tokens
(the short horizon), so every one of the three horizon conditions is backed
by a real (non-truncated-to-empty) target for every kept prompt. This is
applied ONCE and the resulting fixed prompt set + split is reused for all
horizons, matching the reuse protocol.

Outputs (artifacts/):
  usable_prompts.jsonl
  split.json                 (train/val/test indices, seed=42)
  generations.jsonl          {prompt_id, prompt_text, gen_token_ids (<=96),
                               realized_len, hit_eos}
  hidden_last_token.npz      layer_{L} -> [N, hidden_dim] float32
  metadata.json
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from common import SEED, load_model_and_tokenizer, read_jsonl, write_jsonl

MODEL_NAME = "EleutherAI/pythia-1.4b"
PROMPTS_PATH = "artifacts/prompts.jsonl"
BATCH_SIZE = 32
LAYERS = [4, 8, 12, 20]
MAX_NEW_TOKENS = 96          # longest horizon; shorter horizons are prefixes
MIN_GEN_TOKENS = 16          # usable filter = short-horizon length


def main():
    t0 = time.time()
    device = "cuda"
    torch.manual_seed(SEED)
    records = read_jsonl(PROMPTS_PATH)
    print(f"[generate] {len(records)} raw prompts loaded")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)
    tokenizer.padding_side = "left"
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    for L in LAYERS:
        assert 0 <= L <= n_layers

    for r in records:
        r["_prompt_ids"] = tokenizer.encode(r["prompt"])
    order = sorted(range(len(records)), key=lambda i: len(records[i]["_prompt_ids"]), reverse=True)
    eos_id = tokenizer.eos_token_id

    gen_results = [None] * len(records)
    n_batches = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.no_grad():
        for b in range(n_batches):
            idxs = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch_records = [records[i] for i in idxs]
            prompts_text = [r["prompt"] for r in batch_records]
            enc = tokenizer(prompts_text, return_tensors="pt", padding=True).to(device)
            max_prompt_len = enc["input_ids"].shape[1]
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                pad_token_id=eos_id, eos_token_id=eos_id,
            )
            gen_part = out[:, max_prompt_len:]
            for row, rec in enumerate(batch_records):
                gen_ids_full = gen_part[row].tolist()
                hit_eos = eos_id in gen_ids_full
                if hit_eos:
                    gen_ids_full = gen_ids_full[:gen_ids_full.index(eos_id)]
                gen_results[idxs[row]] = {
                    "id": rec["id"], "prompt": rec["prompt"], "domain": rec.get("domain"),
                    "source": rec.get("source"), "prompt_token_ids": rec["_prompt_ids"],
                    "gen_token_ids": gen_ids_full, "realized_len": len(gen_ids_full),
                    "hit_eos": hit_eos,
                }
            if b % 5 == 0 or b == n_batches - 1:
                print(f"[generate] gen batch {b + 1}/{n_batches} ({time.time() - t0:.1f}s)")

    kept = [r for r in gen_results if r is not None and r["realized_len"] >= MIN_GEN_TOKENS]
    print(f"[generate] usable (realized_len >= {MIN_GEN_TOKENS}) = {len(kept)} / {len(gen_results)}")

    os.makedirs("artifacts", exist_ok=True)
    usable_prompts = [
        {"id": r["id"], "prompt": r["prompt"], "domain": r["domain"], "source": r["source"]}
        for r in kept
    ]
    write_jsonl("artifacts/usable_prompts.jsonl", usable_prompts)

    n = len(kept)
    idx_all = np.arange(n)
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.15, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15 / 0.85, random_state=SEED)
    split = {
        "train": idx_train.tolist(), "val": idx_val.tolist(), "test": idx_test.tolist(),
        "seed": SEED, "n_total": n,
        "note": "indices are positions into artifacts/usable_prompts.jsonl / generations.jsonl (fixed order/N reused across all horizons)",
    }
    with open("artifacts/split.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"[generate] split sizes: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    write_jsonl("artifacts/generations.jsonl", [
        {"prompt_id": r["id"], "prompt_text": r["prompt"], "gen_token_ids": r["gen_token_ids"],
         "realized_len": r["realized_len"], "hit_eos": r["hit_eos"]}
        for r in kept
    ])

    # ---- hidden states at last PROMPT token (prompt-only forward pass) ----
    tokenizer.padding_side = "right"  # gather index = length-1
    out_tensors = {L: np.zeros((n, hidden_dim), dtype=np.float32) for L in LAYERS}
    order2 = sorted(range(n), key=lambda i: len(kept[i]["prompt_token_ids"]))
    n_batches2 = (n + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.no_grad():
        for b in range(n_batches2):
            idxs = order2[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch_ids = [kept[i]["prompt_token_ids"] for i in idxs]
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
                for L in LAYERS:
                    out_tensors[L][orig_idx] = hs[L][r, last_pos, :].float().cpu().numpy()
            if b % 5 == 0 or b == n_batches2 - 1:
                print(f"[generate] hidden batch {b + 1}/{n_batches2} ({time.time() - t0:.1f}s)")

    np.savez("artifacts/hidden_last_token.npz", **{f"layer_{L}": out_tensors[L] for L in LAYERS})

    realized_lens = [r["realized_len"] for r in kept]
    metadata = {
        "model_name": MODEL_NAME, "decode": "greedy",
        "decoding_params": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS},
        "n_usable_prompts": n, "max_new_tokens": MAX_NEW_TOKENS, "min_gen_tokens_filter": MIN_GEN_TOKENS,
        "seed": SEED, "n_layers": n_layers, "hidden_dim": hidden_dim, "layers_extracted": LAYERS,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "realized_len_mean": float(np.mean(realized_lens)),
        "realized_len_median": float(np.median(realized_lens)),
        "frac_hit_eos_before_96": float(np.mean([r["hit_eos"] for r in kept])),
    }
    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[generate] done in {time.time() - t0:.1f}s; peak GPU mem {mem:.2f}GB")
    print(f"[generate] metadata: {metadata}")


if __name__ == "__main__":
    main()

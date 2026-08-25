"""
Step 1: reconstruct the same usable-prompt subset as the prior run by running
pythia-1.4b greedy generation over the full reused artifacts/prompts.jsonl
(600 prompts) with the same filter (>=5 generated tokens, non-empty), which
in the prior run yielded 549/600 usable. This fixed usable prompt set (and
its deterministic split) is then reused, unmodified, for every other
model/decoding condition in this study so all comparisons share exactly the
same N and the same train/val/test partition.

Also extracts hidden-state features and saves the greedy pythia-1.4b
continuations directly (so this run doubles as
artifacts/pythia-1.4b/greedy/*).

Outputs:
  artifacts/usable_prompts.jsonl
  artifacts/split.json                (train/val/test indices, seed=42)
  artifacts/pythia-1.4b/greedy/continuations.jsonl
  artifacts/pythia-1.4b/greedy/hidden_last_token.npz
  artifacts/pythia-1.4b/greedy/metadata.json
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from common import (
    MAX_NEW_TOKENS, MIN_GEN_TOKENS, SEED, find_sentence_cut,
    load_model_and_tokenizer, read_jsonl, write_jsonl, depth_fraction_layers,
)

MODEL_NAME = "EleutherAI/pythia-1.4b"
MODEL_TAG = "pythia-1.4b"
PROMPTS_PATH = "artifacts/prompts.jsonl"
BATCH_SIZE = 32
LAYERS_1P4B = [4, 8, 12, 20]  # exact prior candidate layers


def main():
    t0 = time.time()
    device = "cuda"
    torch.manual_seed(SEED)
    records = read_jsonl(PROMPTS_PATH)
    print(f"[build_usable] {len(records)} raw prompts loaded")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)
    tokenizer.padding_side = "left"
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    for L in LAYERS_1P4B:
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
                if eos_id in gen_ids_full:
                    gen_ids_full = gen_ids_full[:gen_ids_full.index(eos_id)]
                cut_ids, cut_text = find_sentence_cut(gen_ids_full, tokenizer)
                gen_results[idxs[row]] = {
                    "id": rec["id"], "prompt": rec["prompt"], "domain": rec.get("domain"),
                    "source": rec.get("source"), "prompt_token_ids": rec["_prompt_ids"],
                    "continuation_text": cut_text, "continuation_token_ids": cut_ids,
                }
            if b % 5 == 0 or b == n_batches - 1:
                print(f"[build_usable] gen batch {b + 1}/{n_batches} ({time.time() - t0:.1f}s)")

    kept = []
    for r in gen_results:
        if r is None:
            continue
        if len(r["continuation_token_ids"]) < MIN_GEN_TOKENS or not r["continuation_text"].strip():
            continue
        kept.append(r)
    print(f"[build_usable] usable = {len(kept)} / {len(gen_results)}")

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
        "note": "indices are positions into artifacts/usable_prompts.jsonl / per-model continuations.jsonl (same fixed order/N reused across all models+decode conditions)",
    }
    with open("artifacts/split.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"[build_usable] split sizes: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    # ---- save pythia-1.4b greedy continuations + hidden states directly ----
    out_dir = f"artifacts/{MODEL_TAG}/greedy"
    os.makedirs(out_dir, exist_ok=True)
    continuations = [
        {"prompt_id": r["id"], "prompt_text": r["prompt"], "continuation_text": r["continuation_text"],
         "continuation_token_ids": r["continuation_token_ids"], "prompt_token_ids": r["prompt_token_ids"]}
        for r in kept
    ]
    write_jsonl(f"{out_dir}/continuations.jsonl", continuations)

    tokenizer.padding_side = "right"  # gather index = length-1
    out_tensors = {L: np.zeros((n, hidden_dim), dtype=np.float32) for L in LAYERS_1P4B}
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
                for L in LAYERS_1P4B:
                    out_tensors[L][orig_idx] = hs[L][r, last_pos, :].float().cpu().numpy()
            if b % 5 == 0 or b == n_batches2 - 1:
                print(f"[build_usable] hidden batch {b + 1}/{n_batches2} ({time.time() - t0:.1f}s)")

    np.savez(f"{out_dir}/hidden_last_token.npz", **{f"layer_{L}": out_tensors[L] for L in LAYERS_1P4B})

    metadata = {
        "model_name": MODEL_NAME, "model_tag": MODEL_TAG, "decode_tag": "greedy",
        "decoding_params": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS},
        "n_usable_prompts": n, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
        "n_layers": n_layers, "hidden_dim": hidden_dim, "layers_extracted": LAYERS_1P4B,
        "layer20_depth_fraction": 20 / n_layers,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[build_usable] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

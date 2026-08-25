"""
Step 2: generic feature extraction utility. For a given Pythia model and a
decoding config (greedy or sampled), loads the model on GPU, and for the
FIXED usable prompt set (artifacts/usable_prompts.jsonl, same N/order for
every condition):
  - generates a continuation (greedy, or sampled with temperature=0.8,
    top_p=0.95, fixed seed);
  - records the hidden state at the last PROMPT token (prompt-only forward
    pass, no generated tokens in context) at depth-fraction-matched layers
    (matching pythia-1.4b's candidate layers 4,8,12,20 out of 24 blocks).

Note: unlike the prior run's per-model MIN_GEN_TOKENS filter, this script
does NOT drop prompts -- the usable subset was already fixed once (see
build_usable_and_split.py) so every model/decode condition has identical N
and can share the same split.json for direct comparability.

Saves under artifacts/{model_tag}/{decode_tag}/:
  continuations.jsonl   {prompt_id, prompt_text, continuation_text}
  hidden_last_token.npz  keys layer_{L} -> [N, hidden_dim] float32
  metadata.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import (
    MAX_NEW_TOKENS, SEED, MODEL_TAGS, find_sentence_cut,
    load_model_and_tokenizer, read_jsonl, write_jsonl, depth_fraction_layers,
)

USABLE_PATH = "artifacts/usable_prompts.jsonl"
BATCH_SIZE = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--decode", choices=["greedy", "sampled"], default="greedy")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda"
    torch.manual_seed(args.seed)

    model_tag = MODEL_TAGS.get(args.model_name, args.model_name.split("/")[-1])
    decode_tag = args.decode
    out_dir = args.out_dir or f"artifacts/{model_tag}/{decode_tag}"
    os.makedirs(out_dir, exist_ok=True)

    records = read_jsonl(USABLE_PATH)
    n = len(records)
    print(f"[extract] {n} usable prompts, model={args.model_name} decode={decode_tag}")

    model, tokenizer = load_model_and_tokenizer(args.model_name, device=device)
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layers = depth_fraction_layers(n_layers)
    print(f"[extract] model has {n_layers} blocks; hidden_dim={hidden_dim}; extracting layers {layers}")

    tokenizer.padding_side = "left"
    for r in records:
        r["_prompt_ids"] = tokenizer.encode(r["prompt"])
    order = sorted(range(n), key=lambda i: len(records[i]["_prompt_ids"]), reverse=True)
    eos_id = tokenizer.eos_token_id

    gen_out = [None] * n
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.no_grad():
        for b in range(n_batches):
            idxs = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch_records = [records[i] for i in idxs]
            prompts_text = [r["prompt"] for r in batch_records]
            enc = tokenizer(prompts_text, return_tensors="pt", padding=True).to(device)
            max_prompt_len = enc["input_ids"].shape[1]

            if args.decode == "greedy":
                gen_kwargs = dict(do_sample=False, num_beams=1)
            else:
                # deterministic per-batch seeding for reproducibility
                torch.manual_seed(args.seed + b)
                gen_kwargs = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p)

            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, pad_token_id=eos_id,
                eos_token_id=eos_id, **gen_kwargs,
            )
            gen_part = out[:, max_prompt_len:]
            for row, rec in enumerate(batch_records):
                gen_ids_full = gen_part[row].tolist()
                if eos_id in gen_ids_full:
                    gen_ids_full = gen_ids_full[:gen_ids_full.index(eos_id)]
                cut_ids, cut_text = find_sentence_cut(gen_ids_full, tokenizer)
                if not cut_text.strip():
                    # keep at least the raw (un-cut) generation so every
                    # condition has a non-empty continuation for the fixed N
                    cut_ids, cut_text = gen_ids_full, tokenizer.decode(gen_ids_full, skip_special_tokens=True)
                if not cut_text.strip():
                    cut_text = "(empty)"
                gen_out[idxs[row]] = {
                    "prompt_id": rec["id"], "prompt_text": rec["prompt"],
                    "continuation_text": cut_text, "continuation_token_ids": cut_ids,
                    "prompt_token_ids": rec["_prompt_ids"],
                }
            if b % 5 == 0 or b == n_batches - 1:
                print(f"[extract] gen batch {b + 1}/{n_batches} ({time.time() - t0:.1f}s)")

    write_jsonl(f"{out_dir}/continuations.jsonl", [
        {"prompt_id": r["prompt_id"], "prompt_text": r["prompt_text"], "continuation_text": r["continuation_text"],
         "continuation_token_ids": r["continuation_token_ids"], "prompt_token_ids": r["prompt_token_ids"]}
        for r in gen_out
    ])

    # ---- hidden states at last PROMPT token (prompt-only forward pass) ----
    tokenizer.padding_side = "right"
    out_tensors = {L: np.zeros((n, hidden_dim), dtype=np.float32) for L in layers}
    order2 = sorted(range(n), key=lambda i: len(gen_out[i]["prompt_token_ids"]))
    n_batches2 = (n + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.no_grad():
        for b in range(n_batches2):
            idxs = order2[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch_ids = [gen_out[i]["prompt_token_ids"] for i in idxs]
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
                for L in layers:
                    out_tensors[L][orig_idx] = hs[L][r, last_pos, :].float().cpu().numpy()
            if b % 5 == 0 or b == n_batches2 - 1:
                print(f"[extract] hidden batch {b + 1}/{n_batches2} ({time.time() - t0:.1f}s)")

    np.savez(f"{out_dir}/hidden_last_token.npz", **{f"layer_{L}": out_tensors[L] for L in layers})

    metadata = {
        "model_name": args.model_name, "model_tag": model_tag, "decode_tag": decode_tag,
        "decoding_params": (
            {"do_sample": False, "num_beams": 1} if args.decode == "greedy"
            else {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
        ),
        "n_usable_prompts": n, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        "n_layers": n_layers, "hidden_dim": hidden_dim, "layers_extracted": layers,
        "layer20_equivalent_depth_fraction": 20 / 24,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[extract] saved to {out_dir}; total time {time.time() - t0:.1f}s; peak GPU mem {mem:.2f}GB")


if __name__ == "__main__":
    main()

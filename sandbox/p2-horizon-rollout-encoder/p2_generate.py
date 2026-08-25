"""
P2 generation: single greedy pass on EleutherAI/pythia-1.4b at
max_new_tokens=256 (the longest tested horizon) for the same 600 reused
prompts (artifacts/prompts.jsonl, md5-identical to every prior project's
prompt set). Shorter horizons (16/48/96/192/256) are DERIVED by truncating
this single 256-token generation (nested-prefix property) -- no sentence-cut
truncation (required: "do not stop at the first sentence for this
experiment").

Also records, in the same prompt-only forward pass, hidden states at the last
PROMPT token at layers 4, 8, 12, 20 (exact prior candidate layers).

Outputs (artifacts/):
  usable_prompts.jsonl   (realized_len >= 16, same floor as prior horizon run)
  generations.jsonl      {prompt_id, prompt_text, gen_token_ids (<=256), realized_len, hit_eos}
  hidden_last_token.npz  layer_{L} -> [N, hidden_dim] float32
  metadata.json
"""
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-1.4b"
PROMPTS_PATH = "/home/jkchoi/project/autopaper/sandbox/horizon-crossover-analysis/artifacts/prompts.jsonl"
OUT_DIR = "/home/jkchoi/project/autopaper/sandbox/p2-horizon-rollout-encoder/artifacts"
BATCH_SIZE = 32
LAYERS = [4, 8, 12, 20]
MAX_NEW_TOKENS = 256
MIN_GEN_TOKENS = 16
SEED = 42


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    t0 = time.time()
    device = "cuda"
    torch.manual_seed(SEED)
    records = read_jsonl(PROMPTS_PATH)
    print(f"[generate] {len(records)} raw prompts loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    tokenizer.padding_side = "left"
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
            if b % 3 == 0 or b == n_batches - 1:
                print(f"[generate] gen batch {b + 1}/{n_batches} ({time.time() - t0:.1f}s)")

    kept = [r for r in gen_results if r is not None and r["realized_len"] >= MIN_GEN_TOKENS]
    print(f"[generate] usable (realized_len >= {MIN_GEN_TOKENS}) = {len(kept)} / {len(gen_results)}")
    lens = np.array([r["realized_len"] for r in kept])
    for H in (16, 48, 96, 192, 256):
        print(f"[generate]   n with realized_len >= {H}: {int((lens >= H).sum())}")

    os.makedirs(OUT_DIR, exist_ok=True)
    usable_prompts = [
        {"id": r["id"], "prompt": r["prompt"], "domain": r["domain"], "source": r["source"]}
        for r in kept
    ]
    write_jsonl(f"{OUT_DIR}/usable_prompts.jsonl", usable_prompts)

    write_jsonl(f"{OUT_DIR}/generations.jsonl", [
        {"prompt_id": r["id"], "prompt_text": r["prompt"], "gen_token_ids": r["gen_token_ids"],
         "realized_len": r["realized_len"], "hit_eos": r["hit_eos"]}
        for r in kept
    ])

    # ---- hidden states at last PROMPT token (prompt-only forward pass) ----
    n = len(kept)
    tokenizer.padding_side = "right"
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

    np.savez(f"{OUT_DIR}/hidden_last_token.npz", **{f"layer_{L}": out_tensors[L] for L in LAYERS})

    metadata = {
        "model_name": MODEL_NAME, "decode": "greedy",
        "decoding_params": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS},
        "n_usable_prompts": n, "max_new_tokens": MAX_NEW_TOKENS, "min_gen_tokens_filter": MIN_GEN_TOKENS,
        "seed": SEED, "n_layers": n_layers, "hidden_dim": hidden_dim, "layers_extracted": LAYERS,
        "realized_len_mean": float(np.mean(lens)), "realized_len_median": float(np.median(lens)),
        "n_at_horizon": {str(H): int((lens >= H).sum()) for H in (16, 48, 96, 192, 256)},
        "frac_hit_eos_before_256": float(np.mean([r["hit_eos"] for r in kept])),
    }
    with open(f"{OUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[generate] done in {time.time() - t0:.1f}s; peak GPU mem {mem:.2f}GB")
    print(f"[generate] metadata: {metadata}")


if __name__ == "__main__":
    main()

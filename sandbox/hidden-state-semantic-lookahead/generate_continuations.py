"""
Step 3: generate one greedy continuation per prompt with Pythia-1.4b.

Saves artifacts/generated.jsonl with fields:
  prompt, prompt_token_ids, continuation_text, continuation_token_ids, full_text, domain, source
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-1.4b"
PROMPTS_PATH = "artifacts/prompts.jsonl"
OUT_PATH = "artifacts/generated.jsonl"
MAX_NEW_TOKENS = 24
MIN_GEN_TOKENS = 5
BATCH_SIZE = 32
SENTENCE_ENDERS = (".", "!", "?")

torch.manual_seed(42)


def find_sentence_cut(token_ids, tokenizer):
    """Return (cut_ids, cut_text): truncate at first sentence-ending
    punctuation among decoded tokens; otherwise return everything."""
    text_so_far = ""
    for k in range(1, len(token_ids) + 1):
        text_so_far = tokenizer.decode(token_ids[:k], skip_special_tokens=True)
        stripped = text_so_far.rstrip()
        if stripped and stripped[-1] in SENTENCE_ENDERS:
            return token_ids[:k], text_so_far
    return token_ids, text_so_far


def main():
    t0 = time.time()
    device = "cuda"
    print(f"[generate] loading model {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
    model.eval()

    records = [json.loads(l) for l in open(PROMPTS_PATH)]
    print(f"[generate] {len(records)} prompts loaded")

    # sort by token length (desc) to reduce padding waste, remember original order
    for r in records:
        r["_prompt_ids"] = tokenizer.encode(r["prompt"])
    order = sorted(range(len(records)), key=lambda i: len(records[i]["_prompt_ids"]), reverse=True)

    results = [None] * len(records)
    n_batches = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for b in range(n_batches):
            idxs = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch_records = [records[i] for i in idxs]
            prompts_text = [r["prompt"] for r in batch_records]
            enc = tokenizer(prompts_text, return_tensors="pt", padding=True).to(device)
            max_prompt_len = enc["input_ids"].shape[1]

            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
            )
            gen_part = out[:, max_prompt_len:]  # [batch, MAX_NEW_TOKENS] (or fewer if all stopped early)

            for row, rec in enumerate(batch_records):
                gen_ids_full = gen_part[row].tolist()
                # truncate at first eos/pad occurrence
                if eos_id in gen_ids_full:
                    first_eos = gen_ids_full.index(eos_id)
                    gen_ids_full = gen_ids_full[:first_eos]
                # truncate further at first sentence-ending punctuation
                cut_ids, cut_text = find_sentence_cut(gen_ids_full, tokenizer)
                prompt_ids = rec["_prompt_ids"]
                full_text = tokenizer.decode(prompt_ids + cut_ids, skip_special_tokens=True)
                results[idxs[row]] = {
                    "id": rec["id"],
                    "prompt": rec["prompt"],
                    "domain": rec.get("domain"),
                    "source": rec.get("source"),
                    "prompt_token_ids": prompt_ids,
                    "continuation_text": cut_text,
                    "continuation_token_ids": cut_ids,
                    "full_text": full_text,
                }
            if b % 5 == 0 or b == n_batches - 1:
                print(f"[generate] batch {b + 1}/{n_batches} done ({time.time() - t0:.1f}s elapsed)")

    # filter
    kept = []
    dropped = 0
    for r in results:
        if r is None:
            dropped += 1
            continue
        if len(r["continuation_token_ids"]) < MIN_GEN_TOKENS or not r["continuation_text"].strip():
            dropped += 1
            continue
        kept.append(r)

    print(f"[generate] kept {len(kept)} / {len(results)} (dropped {dropped})")

    os.makedirs("artifacts", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[generate] wrote {len(kept)} records to {OUT_PATH}")
    print(f"[generate] total time {time.time() - t0:.1f}s, peak GPU mem {mem:.2f} GB")


if __name__ == "__main__":
    main()

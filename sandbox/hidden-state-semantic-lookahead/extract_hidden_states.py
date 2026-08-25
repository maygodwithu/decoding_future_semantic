"""
Step 4: extract hidden states at the final prompt-token position for each
generated example, at 4 candidate layers, running the model on the prompt
alone (no generated continuation in context).

Saves artifacts/hidden_states_layer_{L}.pt: a torch tensor [N, hidden_dim]
aligned row-for-row with artifacts/generated.jsonl.
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-1.4b"
GENERATED_PATH = "artifacts/generated.jsonl"
LAYERS = [4, 8, 12, 20]  # indices into output_hidden_states tuple (0 = embeddings)
BATCH_SIZE = 32


def main():
    t0 = time.time()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # so gather index = length - 1

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"[hidden] model has {n_layers} transformer layers; extracting hidden_states indices {LAYERS}")
    for L in LAYERS:
        assert 0 <= L <= n_layers, f"layer index {L} out of range [0,{n_layers}]"

    records = [json.loads(l) for l in open(GENERATED_PATH)]
    print(f"[hidden] {len(records)} records")

    hidden_dim = model.config.hidden_size
    out_tensors = {L: torch.zeros(len(records), hidden_dim, dtype=torch.float32) for L in LAYERS}

    # sort by prompt length for efficient batching, keep track of original idx
    order = sorted(range(len(records)), key=lambda i: len(records[i]["prompt_token_ids"]))
    n_batches = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for b in range(n_batches):
            idxs = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch_ids = [records[i]["prompt_token_ids"] for i in idxs]
            lengths = [len(x) for x in batch_ids]
            maxlen = max(lengths)
            input_ids = torch.full((len(batch_ids), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
            for r, ids in enumerate(batch_ids):
                input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[r, :len(ids)] = 1
            input_ids = input_ids.to(device)
            attn = attn.to(device)

            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            hs = out.hidden_states  # tuple length n_layers+1, each [batch, seq, hidden]

            for r, orig_idx in enumerate(idxs):
                last_pos = lengths[r] - 1
                for L in LAYERS:
                    vec = hs[L][r, last_pos, :].float().cpu()
                    out_tensors[L][orig_idx] = vec

            if b % 5 == 0 or b == n_batches - 1:
                print(f"[hidden] batch {b + 1}/{n_batches} done ({time.time() - t0:.1f}s elapsed)")

    for L in LAYERS:
        path = f"artifacts/hidden_states_layer_{L}.pt"
        torch.save(out_tensors[L], path)
        print(f"[hidden] saved {path} shape={tuple(out_tensors[L].shape)}")

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[hidden] total time {time.time() - t0:.1f}s, peak GPU mem {mem:.2f} GB")


if __name__ == "__main__":
    main()

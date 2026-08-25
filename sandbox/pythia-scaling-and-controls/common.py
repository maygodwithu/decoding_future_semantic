"""
Shared utilities for the pythia-scaling-and-controls follow-up experiment.
Mirrors preprocessing / metric definitions from the prior accepted run in
hidden-state-semantic-lookahead/ (build_prompts.py, extract_hidden_states.py,
generate_continuations.py, compute_embeddings.py, probe_train.py,
steering_utils.py) so results are directly comparable.
"""
import json
import os

import numpy as np
import torch

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_ENDERS = (".", "!", "?")
MAX_NEW_TOKENS = 24   # matched to prior run
MIN_GEN_TOKENS = 5    # matched to prior run's usable-subset filter
SEED = 42

MODEL_TAGS = {
    "EleutherAI/pythia-160m": "pythia-160m",
    "EleutherAI/pythia-410m": "pythia-410m",
    "EleutherAI/pythia-1b": "pythia-1.0b",
    "EleutherAI/pythia-1.4b": "pythia-1.4b",
    "EleutherAI/pythia-2.8b": "pythia-2.8b",
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def find_sentence_cut(token_ids, tokenizer):
    """Truncate at the first sentence-ending punctuation among decoded
    tokens; otherwise return everything. Mirrors prior generate_continuations.py."""
    text_so_far = ""
    for k in range(1, len(token_ids) + 1):
        text_so_far = tokenizer.decode(token_ids[:k], skip_special_tokens=True)
        stripped = text_so_far.rstrip()
        if stripped and stripped[-1] in SENTENCE_ENDERS:
            return token_ids[:k], text_so_far
    return token_ids, text_so_far


def cosine_rows(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def retrieval_metrics(query_emb, gallery_emb):
    """Same procedure as prior probe_train.py: rank of the true diagonal match."""
    q = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
    g = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-12)
    sim = q @ g.T
    n_q = sim.shape[0]
    ranks = []
    r1 = r5 = 0
    for i in range(n_q):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        if rank == 1:
            r1 += 1
        if rank <= 5:
            r5 += 1
    return {
        "recall_at_1": r1 / n_q,
        "recall_at_5": r5 / n_q,
        "mean_rank": float(np.mean(ranks)),
        "n_query": n_q,
    }


def depth_fraction_layers(n_layers, fractions=(4 / 24, 8 / 24, 12 / 24, 20 / 24)):
    """Map the prior run's candidate layer indices (4,8,12,20 out of pythia-1.4b's
    24 transformer blocks) to matched depth-fraction layer indices for a model
    with a different number of blocks. Returns a sorted, deduped list of
    hidden_states-tuple indices (1..n_layers)."""
    idxs = sorted(set(max(1, min(n_layers, round(f * n_layers))) for f in fractions))
    return idxs


def load_model_and_tokenizer(model_name, device="cuda", dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def jaccard(text_a, text_b):
    sa = set(text_a.lower().split())
    sb = set(text_b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

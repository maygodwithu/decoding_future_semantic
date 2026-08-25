"""
Generic causal steering test, reusing the PRIOR accepted run's methodology
(steering_experiment.py) rather than a per-example-target formulation: an
early per-example-target design was piloted here and found to hit a ceiling
effect (unperturbed greedy baseline already reproduces the true continuation
almost exactly, cos~0.98, leaving only room for perturbations to hurt, not
help -- see report.md for the full explanation). The prior run's sentiment-
contrast-axis design avoids this ceiling and is reproduced here, generalized
to run on any model/layer/dir:

  - v_sem = normalize(mean(embed(POSITIVE_ANCHORS)) - mean(embed(NEGATIVE_ANCHORS)))
  - d_sem = normalize(W.T @ v_sem)  (probe weights map the embedding-space
    axis back into this model's hidden space at its best/matched layer)
  - d_rand_i: an independent per-prompt random unit direction (matched norm)
  - alpha = 4.0 * ||std_per_dim(hidden_states_val)||_2  (same pilot recipe)
  - conditions: base (delta=None), sem_plus (+alpha*d_sem, shared across
    prompts), rand_plus (+alpha*d_rand_i, fresh per prompt)
  - score(text) = cos(embed(text), pos_mean_unit) - cos(embed(text), neg_mean_unit)

Reports:
  steering_effect_semantic = mean(score(sem_plus) - score(base))
  steering_effect_random   = mean(score(rand_plus) - score(base))
  steering_margin           = steering_effect_semantic - steering_effect_random
"""
import argparse
import json

import numpy as np
import torch

from common import find_sentence_cut, load_model_and_tokenizer, read_jsonl
from embed_continuations import embed_texts, get_embedder
from steering_utils_generic import BatchedInjectionHook, compute_pilot_alpha

SEED = 42
MIN_CONT_TOKENS = 8
MAX_NEW_TOKENS = 24

POSITIVE_ANCHORS = [
    "This is wonderful.", "The outcome was excellent.", "It made everyone happy.",
    "Everything turned out great.", "What a fantastic result.",
]
NEGATIVE_ANCHORS = [
    "This is terrible.", "The outcome was awful.", "It made everyone upset.",
    "Everything turned out badly.", "What a disastrous result.",
]


def batched_generate(model, tokenizer, prompt_ids_list, hook, deltas, device, eos_id):
    tokenizer.padding_side = "left"
    maxlen = max(len(p) for p in prompt_ids_list)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(prompt_ids_list), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(prompt_ids_list), maxlen), dtype=torch.long)
    for r, ids in enumerate(prompt_ids_list):
        input_ids[r, maxlen - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attn[r, maxlen - len(ids):] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    hook.reset()
    if deltas is None:
        hook.set_delta(None)
    else:
        if deltas.dim() == 1:
            deltas = deltas.unsqueeze(0).expand(len(prompt_ids_list), -1)
        hook.set_delta(deltas)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attn, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, num_beams=1, pad_token_id=eos_id, eos_token_id=eos_id,
        )
    gen_part = out[:, maxlen:]
    texts = []
    for row in range(len(prompt_ids_list)):
        ids = gen_part[row].tolist()
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        _, text = find_sentence_cut(ids, tokenizer)
        texts.append(text.strip() or "(empty)")
    return texts


def axis_scores(texts, pos_mean, neg_mean):
    emb = embed_texts(texts)
    pn = pos_mean / (np.linalg.norm(pos_mean) + 1e-12)
    nn = neg_mean / (np.linalg.norm(neg_mean) + 1e-12)
    return emb @ pn - emb @ nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--split", default="artifacts/split.json")
    ap.add_argument("--n_prompts", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    split = json.load(open(args.split))
    idx_test, idx_val = split["test"], split["val"]

    npz = np.load(f"{args.dir}/hidden_last_token.npz")
    probe_npz = np.load(f"{args.dir}/best_probe_trainval_layer_{args.layer}.npz")
    W = probe_npz["W"]
    hidden_dim = W.shape[1]
    continuations = read_jsonl(f"{args.dir}/continuations.jsonl")

    rng = np.random.RandomState(SEED)
    eligible = [i for i in idx_test if len(continuations[i].get("continuation_token_ids") or []) >= MIN_CONT_TOKENS]
    if not eligible:
        eligible = [i for i in idx_test if len(continuations[i]["continuation_text"].split()) >= 5]
    rng.shuffle(eligible)
    eligible = eligible[:args.n_prompts]
    print(f"[steer:{args.dir}] {len(eligible)} eligible test prompts selected")

    device = "cuda"
    model, tokenizer = load_model_and_tokenizer(args.model_name, device=device)
    eos_id = tokenizer.eos_token_id
    embedder = get_embedder()

    pos_emb = embedder.encode(POSITIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    neg_emb = embedder.encode(NEGATIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    pos_mean, neg_mean = pos_emb.mean(axis=0), neg_emb.mean(axis=0)
    v_sem = pos_mean - neg_mean
    v_sem = v_sem / (np.linalg.norm(v_sem) + 1e-12)
    d_sem = W.T @ v_sem
    d_sem = d_sem / (np.linalg.norm(d_sem) + 1e-12)

    hs_val = npz[f"layer_{args.layer}"][idx_val]
    alpha, agg_std = compute_pilot_alpha(hs_val, fraction=4.0)
    print(f"[steer:{args.dir}] alpha={alpha:.4f} agg_std={agg_std:.4f}")

    prompt_ids_list = [continuations[i].get("prompt_token_ids") for i in eligible]
    if any(p is None for p in prompt_ids_list):
        prompt_ids_list = [tokenizer.encode(continuations[i]["prompt_text"]) for i in eligible]

    d_rand_list = []
    for i in eligible:
        prompt_rng = np.random.RandomState(200000 + int(continuations[i]["prompt_id"]))
        d_rand = prompt_rng.normal(size=hidden_dim)
        d_rand = d_rand / np.linalg.norm(d_rand)
        d_rand_list.append(d_rand)

    d_sem_arr = torch.tensor(np.tile(d_sem, (len(eligible), 1)), dtype=torch.float32) * alpha
    d_rand_arr = torch.tensor(np.stack(d_rand_list), dtype=torch.float32) * alpha

    hook = BatchedInjectionHook(model, args.layer)
    base_texts, sem_texts, rand_texts = [], [], []
    bs = args.batch_size
    for b in range(0, len(eligible), bs):
        chunk_prompts = prompt_ids_list[b:b + bs]
        base_texts += batched_generate(model, tokenizer, chunk_prompts, hook, None, device, eos_id)
        sem_texts += batched_generate(model, tokenizer, chunk_prompts, hook, d_sem_arr[b:b + bs], device, eos_id)
        rand_texts += batched_generate(model, tokenizer, chunk_prompts, hook, d_rand_arr[b:b + bs], device, eos_id)
        print(f"[steer:{args.dir}] batch {b // bs + 1} done")
    hook.remove()

    score_base = axis_scores(base_texts, pos_mean, neg_mean)
    score_sem = axis_scores(sem_texts, pos_mean, neg_mean)
    score_rand = axis_scores(rand_texts, pos_mean, neg_mean)

    effect_sem = float(np.mean(score_sem - score_base))
    effect_rand = float(np.mean(score_rand - score_base))
    margin = effect_sem - effect_rand

    summary = {
        "dir": args.dir, "model_name": args.model_name, "layer": args.layer,
        "alpha": alpha, "agg_std": agg_std, "n_prompts": len(eligible),
        "mean_sentiment_score_base": float(np.mean(score_base)),
        "mean_sentiment_score_semantic": float(np.mean(score_sem)),
        "mean_sentiment_score_random": float(np.mean(score_rand)),
        "steering_effect_semantic": effect_sem, "steering_effect_random": effect_rand,
        "steering_margin": margin,
        "frac_semantic_shift_positive": float(np.mean((score_sem - score_base) > 0)),
        "frac_random_shift_positive": float(np.mean((score_rand - score_base) > 0)),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

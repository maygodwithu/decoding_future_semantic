"""
Step 8: causal semantic steering test.

For the best-validation layer, we:
  1. use the ridge probe W (fit on train+val) to map an embedding-space
     sentiment contrast (positive - negative) back into hidden-state space:
     d_h = W^T v_sem, normalized to a unit vector;
  2. pick a matched-norm random control direction d_rand;
  3. choose an intervention magnitude alpha from a pilot on validation
     hidden states;
  4. for ~100 held-out (test-split) prompts with continuations >= 8 tokens,
     generate under alpha in {0, +a, -a} for the semantic direction and
     alpha in {+a, -a} for the random-direction control;
  5. score each generation's continuation embedding on the sentiment axis
     and report mean shift vs. alpha=0, plus the fraction of prompts shifting
     in the predicted direction.

Also runs a secondary "temporal" contrast (future vs. past) as additional
(non-primary) evidence.

Saves artifacts/steering_results.jsonl and artifacts/steering_summary.json.
"""
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from steering_utils import InjectionHook, compute_pilot_alpha

MODEL_NAME = "EleutherAI/pythia-1.4b"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATED_PATH = "artifacts/generated.jsonl"
SPLIT_PATH = "artifacts/split_indices.json"
PROBE_RESULTS_PATH = "artifacts/probe_results.json"
MAX_NEW_TOKENS = 24
MIN_CONT_TOKENS_FOR_STEERING = 8
TARGET_N_PROMPTS = 100
SEED = 42
SENTENCE_ENDERS = (".", "!", "?")

POSITIVE_ANCHORS = [
    "This is wonderful.",
    "The outcome was excellent.",
    "It made everyone happy.",
    "Everything turned out great.",
    "What a fantastic result.",
]
NEGATIVE_ANCHORS = [
    "This is terrible.",
    "The outcome was awful.",
    "It made everyone upset.",
    "Everything turned out badly.",
    "What a disastrous result.",
]
FUTURE_ANCHORS = [
    "This will happen tomorrow.",
    "In the future, things will change.",
    "Soon we will see the results.",
    "Next year the plan will begin.",
    "Eventually this will be finished.",
]
PAST_ANCHORS = [
    "This happened yesterday.",
    "In the past, things were different.",
    "We already saw the results.",
    "Last year the plan began.",
    "This was finished long ago.",
]


def find_sentence_cut(token_ids, tokenizer):
    text_so_far = ""
    for k in range(1, len(token_ids) + 1):
        text_so_far = tokenizer.decode(token_ids[:k], skip_special_tokens=True)
        stripped = text_so_far.rstrip()
        if stripped and stripped[-1] in SENTENCE_ENDERS:
            return token_ids[:k], text_so_far
    return token_ids, text_so_far


def generate_one(model, tokenizer, prompt_ids, hook, delta, device, eos_id):
    hook.reset()
    hook.set_delta(delta)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )
    gen_ids = out[0, len(prompt_ids):].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    cut_ids, cut_text = find_sentence_cut(gen_ids, tokenizer)
    return cut_text


def axis_score(embedder, text, pos_mean, neg_mean):
    e = embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    pn = pos_mean / (np.linalg.norm(pos_mean) + 1e-12)
    nn = neg_mean / (np.linalg.norm(neg_mean) + 1e-12)
    cos_pos = float(np.dot(e, pn))
    cos_neg = float(np.dot(e, nn))
    return cos_pos - cos_neg, e


def main():
    t0 = time.time()
    device = "cuda"
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)

    records = [json.loads(l) for l in open(GENERATED_PATH)]
    split = json.load(open(SPLIT_PATH))
    probe_results = json.load(open(PROBE_RESULTS_PATH))
    best_layer = probe_results["best_layer_by_val"]
    print(f"[steer] using best layer = {best_layer}")

    probe_npz = np.load(f"artifacts/best_probe_trainval_layer_{best_layer}.npz")
    W = probe_npz["W"]  # [384, 2048]  pred = X @ W.T + b
    hidden_dim = W.shape[1]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
    model.eval()
    eos_id = tokenizer.eos_token_id

    embedder = SentenceTransformer(EMB_MODEL_NAME, device="cuda")

    # ---- semantic contrast directions in embedding space ----
    pos_emb = embedder.encode(POSITIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    neg_emb = embedder.encode(NEGATIVE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    pos_mean = pos_emb.mean(axis=0)
    neg_mean = neg_emb.mean(axis=0)
    v_sem_sentiment = pos_mean - neg_mean
    v_sem_sentiment = v_sem_sentiment / (np.linalg.norm(v_sem_sentiment) + 1e-12)

    fut_emb = embedder.encode(FUTURE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    past_emb = embedder.encode(PAST_ANCHORS, normalize_embeddings=True, convert_to_numpy=True)
    fut_mean = fut_emb.mean(axis=0)
    past_mean = past_emb.mean(axis=0)
    v_sem_temporal = fut_mean - past_mean
    v_sem_temporal = v_sem_temporal / (np.linalg.norm(v_sem_temporal) + 1e-12)

    def to_hidden_dir(v_sem):
        d_h = W.T @ v_sem  # [2048]
        d_h = d_h / (np.linalg.norm(d_h) + 1e-12)
        return d_h

    d_sentiment = to_hidden_dir(v_sem_sentiment)
    d_temporal = to_hidden_dir(v_sem_temporal)

    print(f"[steer] cos(d_sentiment, d_temporal) = {float(np.dot(d_sentiment, d_temporal)):.3f}")

    # ---- pilot alpha from validation hidden states at this layer ----
    # A pilot sweep (see artifacts/steering_pilot_sweep.json) showed the
    # protocol's literal suggestion (fraction=0.5, i.e. alpha ~= 34) gave a
    # negligible, noisy effect that did not clearly separate from a random
    # direction; effect size grew ~monotonically with alpha up to the largest
    # multiplier tested (20x) before generations started to degrade in
    # coherence. We use fraction=4.0 (alpha ~= 271, ~8x the literal
    # suggestion) as an empirically-tuned "modest but measurable" magnitude
    # that keeps outputs mostly fluent while producing a visible effect. This
    # deviation from the literal protocol value is reported transparently.
    val_idx = split["val"]
    hs_all = torch.load(f"artifacts/hidden_states_layer_{best_layer}.pt").numpy()
    hs_val = hs_all[val_idx]
    alpha, agg_std = compute_pilot_alpha(hs_val, fraction=4.0)
    print(f"[steer] pilot aggregate std = {agg_std:.4f}, alpha = {alpha:.4f}")

    # ---- select held-out test prompts ----
    test_idx = split["test"]
    eligible = [i for i in test_idx if len(records[i]["continuation_token_ids"]) >= MIN_CONT_TOKENS_FOR_STEERING]
    rng.shuffle(eligible)
    eligible = eligible[:TARGET_N_PROMPTS]
    print(f"[steer] {len(eligible)} held-out prompts selected (target {TARGET_N_PROMPTS}, "
          f"test-split size {len(test_idx)})")

    hook = InjectionHook(model, best_layer, delta=None)

    # Random-direction control: to avoid the result hinging on one lucky/
    # unlucky draw, we sample a FRESH independent random unit direction per
    # prompt (deterministic per-prompt seed) for both the sentiment and
    # temporal random controls, and average the resulting effect over all
    # prompts. This estimates the expected effect of an arbitrary
    # matched-norm direction much more reliably than a single fixed draw.
    rows = []
    for n_done, idx in enumerate(eligible):
        rec = records[idx]
        prompt_ids = rec["prompt_token_ids"]

        prompt_rng = np.random.RandomState(100000 + idx)
        d_rand_i = prompt_rng.normal(size=hidden_dim)
        d_rand_i = d_rand_i / np.linalg.norm(d_rand_i)
        d_rand2_i = prompt_rng.normal(size=hidden_dim)
        d_rand2_i = d_rand2_i / np.linalg.norm(d_rand2_i)

        conditions = [
            ("base", 0.0, None),
            ("sem_plus", alpha, d_sentiment),
            ("sem_minus", -alpha, d_sentiment),
            ("rand_plus", alpha, d_rand_i),
            ("rand_minus", -alpha, d_rand_i),
            ("temp_plus", alpha, d_temporal),
            ("temp_minus", -alpha, d_temporal),
            ("randB_plus", alpha, d_rand2_i),
            ("randB_minus", -alpha, d_rand2_i),
        ]

        for cond_name, a, direction in conditions:
            if direction is None:
                delta = None
            else:
                delta = torch.tensor(a * direction, dtype=torch.float16)
            gen_text = generate_one(model, tokenizer, prompt_ids, hook, delta, device, eos_id)
            score_sent, emb_vec = axis_score(embedder, gen_text, pos_mean, neg_mean)
            score_temp, _ = axis_score(embedder, gen_text, fut_mean, past_mean)
            rows.append({
                "prompt_idx": idx,
                "prompt": rec["prompt"],
                "condition": cond_name,
                "alpha": a,
                "layer": best_layer,
                "generated_text": gen_text,
                "sentiment_score": score_sent,
                "temporal_score": score_temp,
            })

        if (n_done + 1) % 20 == 0 or n_done == len(eligible) - 1:
            print(f"[steer] {n_done + 1}/{len(eligible)} prompts done ({time.time() - t0:.1f}s elapsed)")

    hook.remove()

    with open("artifacts/steering_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[steer] wrote {len(rows)} rows to artifacts/steering_results.jsonl")

    # ---- summarize ----
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_idx"], {})[r["condition"]] = r

    def summarize(axis_key, plus_cond, minus_cond, base_key):
        deltas_plus, deltas_minus = [], []
        n_plus_correct = n_minus_correct = 0
        n = 0
        for pidx, conds in by_prompt.items():
            if plus_cond not in conds or minus_cond not in conds or "base" not in conds:
                continue
            base_score = conds["base"][axis_key]
            plus_score = conds[plus_cond][axis_key]
            minus_score = conds[minus_cond][axis_key]
            d_plus = plus_score - base_score
            d_minus = minus_score - base_score
            deltas_plus.append(d_plus)
            deltas_minus.append(d_minus)
            if d_plus > 0:
                n_plus_correct += 1
            if d_minus < 0:
                n_minus_correct += 1
            n += 1
        return {
            "n": n,
            "mean_delta_plus": float(np.mean(deltas_plus)) if n else None,
            "mean_delta_minus": float(np.mean(deltas_minus)) if n else None,
            "frac_plus_correct_direction": n_plus_correct / n if n else None,
            "frac_minus_correct_direction": n_minus_correct / n if n else None,
        }

    summary = {
        "best_layer": best_layer,
        "alpha": alpha,
        "pilot_aggregate_std": agg_std,
        "n_prompts": len(eligible),
        "cos_dsentiment_dtemporal": float(np.dot(d_sentiment, d_temporal)),
        "sentiment_direction": summarize("sentiment_score", "sem_plus", "sem_minus", "base"),
        "sentiment_random_control": summarize("sentiment_score", "rand_plus", "rand_minus", "base"),
        "temporal_direction": summarize("temporal_score", "temp_plus", "temp_minus", "base"),
        "temporal_random_control": summarize("temporal_score", "randB_plus", "randB_minus", "base"),
    }
    with open("artifacts/steering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[steer] total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

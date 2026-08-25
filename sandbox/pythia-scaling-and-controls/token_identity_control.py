"""
Priority 1: token-identity control baselines for a given model/decode
directory, evaluated against the true full-continuation embeddings.

Controls (all derived WITHOUT using the true continuation):
  A. Logit-lens next-token text controls at a chosen hidden layer:
     - apply final layer norm + lm_head to the hidden state to get next-token
       logits;
     - top-k predicted tokens (k in {1,5,10}), two text variants:
         concat_tokens: decoded top-k tokens concatenated in rank order
         weighted_tokens_text: each token repeated round(10*p_i) times (min 1)
  B. Short greedy/sampled rollout controls: the first m tokens (m in
     {1,3,5}) of the ALREADY GENERATED continuation for this condition. This
     is exactly what a live m-token-only generation would produce for greedy
     decoding (generation is causal and does not depend on max_new_tokens),
     and is used as a fixed, cheap, reproducible surrogate for sampled
     decoding too (documented deviation, see report.md).

Evaluates on the test split against the ground-truth continuation
embeddings: cosine similarity, Recall@5 retrieval (same procedure as probe),
probe-minus-control margins, and a secondary lexical Jaccard overlap metric.

Saves --out_csv (e.g. artifacts/pythia-1.4b/greedy/token_control_results.csv).
"""
import argparse
import csv
import json

import numpy as np
import torch

from common import cosine_rows, jaccard, load_model_and_tokenizer, read_jsonl, retrieval_metrics
from embed_continuations import embed_texts

TOPK_VALUES = [1, 5, 10]
ROLLOUT_M_VALUES = [1, 3, 5]


def build_topk_texts(tokenizer, logits_row, k):
    probs = torch.softmax(logits_row, dim=-1)
    topp, topi = torch.topk(probs, k)
    toks = [tokenizer.decode([tid]) for tid in topi.tolist()]
    concat_text = "".join(toks).strip() or " ".join(t.strip() for t in toks)
    weighted_parts = []
    for tok, p in zip(toks, topp.tolist()):
        reps = max(1, round(10 * p))
        weighted_parts.extend([tok.strip() or tok] * reps)
    weighted_text = " ".join(weighted_parts)
    return concat_text if concat_text.strip() else "(empty)", weighted_text if weighted_text.strip() else "(empty)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--layer", type=int, required=True, help="hidden_states index used for logit-lens + as 'layer 20' control layer")
    ap.add_argument("--probe_layer", type=int, default=None, help="best probe layer (for margin reporting); defaults to --layer")
    ap.add_argument("--split", default="artifacts/split.json")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    probe_layer = args.probe_layer if args.probe_layer is not None else args.layer

    split = json.load(open(args.split))
    idx_test = split["test"]

    npz = np.load(f"{args.dir}/hidden_last_token.npz")
    y_true = np.load(f"{args.dir}/continuation_embeddings.npy")
    records = read_jsonl(f"{args.dir}/continuations.jsonl")
    gen_records = read_jsonl("artifacts/usable_prompts.jsonl")  # for prompt text alignment sanity
    y_test = y_true[idx_test]

    # ---- probe predictions (already trained) ----
    probe_pred = np.load(f"{args.dir}/test_predictions_layer_{probe_layer}.npy")
    probe_cos = cosine_rows(probe_pred, y_test)
    probe_retr = retrieval_metrics(probe_pred, y_test)

    rows_summary = []
    rows_summary.append({
        "method": f"semantic_probe_layer{probe_layer}", "test_cosine": float(np.mean(probe_cos)),
        "recall_at_5": probe_retr["recall_at_5"], "margin_vs_probe": 0.0, "lexical_jaccard": "",
    })

    # ---- A. logit-lens top-k controls ----
    model, tokenizer = load_model_and_tokenizer(args.model_name, device="cuda")
    hidden_test = torch.tensor(npz[f"layer_{args.layer}"][idx_test], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        # apply final layer norm then lm_head (logit lens), matches GPT-NeoX-style models
        normed = model.gpt_neox.final_layer_norm(hidden_test.to(next(model.parameters()).dtype))
        logits = model.get_output_embeddings()(normed).float().cpu()  # [n_test, vocab]

    true_texts_test = [records[i]["continuation_text"] for i in idx_test]

    for k in TOPK_VALUES:
        concat_texts, weighted_texts = [], []
        for row in range(logits.shape[0]):
            c, w = build_topk_texts(tokenizer, logits[row], k)
            concat_texts.append(c)
            weighted_texts.append(w)
        for name, texts in [(f"topk_concat_k{k}", concat_texts), (f"topk_weighted_k{k}", weighted_texts)]:
            emb = embed_texts(texts)
            cos = cosine_rows(emb, y_test)
            retr = retrieval_metrics(emb, y_test)
            jac = float(np.mean([jaccard(t, gt) for t, gt in zip(texts, true_texts_test)]))
            rows_summary.append({
                "method": name, "test_cosine": float(np.mean(cos)), "recall_at_5": retr["recall_at_5"],
                "margin_vs_probe": float(np.mean(probe_cos)) - float(np.mean(cos)), "lexical_jaccard": jac,
            })

    # ---- B. short rollout controls (first m tokens of already-generated continuation) ----
    for m in ROLLOUT_M_VALUES:
        texts = []
        for i in idx_test:
            ids = records[i].get("continuation_token_ids")
            if ids is None:
                # continuations.jsonl for pythia-1.4b/greedy doesn't store token ids; re-tokenize text
                ids = tokenizer.encode(records[i]["continuation_text"])
            short_ids = ids[:m]
            txt = tokenizer.decode(short_ids, skip_special_tokens=True).strip() or "(empty)"
            texts.append(txt)
        emb = embed_texts(texts)
        cos = cosine_rows(emb, y_test)
        retr = retrieval_metrics(emb, y_test)
        jac = float(np.mean([jaccard(t, gt) for t, gt in zip(texts, true_texts_test)]))
        rows_summary.append({
            "method": f"short_rollout_m{m}", "test_cosine": float(np.mean(cos)), "recall_at_5": retr["recall_at_5"],
            "margin_vs_probe": float(np.mean(probe_cos)) - float(np.mean(cos)), "lexical_jaccard": jac,
        })

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "test_cosine", "recall_at_5", "margin_vs_probe", "lexical_jaccard"])
        writer.writeheader()
        for r in rows_summary:
            writer.writerow(r)

    print(f"[token_control] wrote {args.out_csv}")
    for r in rows_summary:
        print(r)


if __name__ == "__main__":
    main()

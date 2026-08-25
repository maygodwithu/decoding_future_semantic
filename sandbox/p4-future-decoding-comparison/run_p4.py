"""
P4 - Comparison with Existing Future-Decoding Methods.

Reuses (read-only) sandbox/p2-horizon-rollout-encoder/artifacts/generations.jsonl
and hidden_last_token.npz (Pythia-1.4B, greedy, 256-token generation pass,
prompt-only hidden states at layers 4/8/12/20, 600 records) -- no new
autoregressive generation is performed anywhere in P4. The common subset
(realized_len >= 256, n=577) and train/val/test split (seed 42) are recomputed
with EXACTLY the same procedure P2 used, so indices match P2's bit-for-bit.

Methods compared, all at ZERO additional autoregressive decode steps except
explicit rollout:
  M1 Direct Logit Lens   -- frozen model head applied to raw h_L (no training)
  M2 Tuned Lens           -- affine translator A_L,b_L (h~ = A_L h_L + b_L),
                             then frozen final_layer_norm + lm_head, trained
                             on TRAIN next-token cross-entropy, model-selected
                             on VAL, never touching TEST
  M3 Future-Lens-style    -- per-offset affine translator A_j,b_j (same
                             architecture as Tuned Lens, one per future offset
                             j=1..10), each trained independently on the
                             SAME original pre-generation hidden state to
                             predict the token at offset j (no autoregressive
                             chaining) -- this affine-translator construction
                             (rather than a raw hidden->vocab linear probe) is
                             the deliberate implementation deviation from the
                             published Future Lens method, chosen for
                             architectural consistency with M2 and documented
                             here per the spec's requirement to flag any such
                             deviation. Referred to as "Future-Lens-style"
                             throughout, never as "Future Lens".
  M6 Linear Semantic Probe -- unchanged ridge probe (reuses P2's own
                             per-horizon results directly where no new
                             per-example array is needed; recomputed at H=16
                             specifically to obtain per-example cosines for
                             bootstrap CIs, and cross-checked against P2's
                             saved summary number as a consistency check)
  M8 MLP Semantic Probe    -- optional nonlinear diagnostic, small 1-hidden-
                             layer MLP, model-selected on VAL

Rollout-m (m=3,5,10) is recomputed directly from the same generated
trajectories for per-example arrays (bootstrap needs per-example cosines,
which P2 did not persist to disk); values are a consistency check against
P2's saved summary numbers.

Primary horizon H=16 (bootstrap + main comparison table). Robustness
horizons H=48, H=96 reuse the SAME trained decoders (Tuned Lens / Future-
Lens-style outputs do not depend on target horizon) and P2's own saved
Probe/Rollout numbers -- no retraining, only re-embedding of the horizon-H
target and Tuned-Lens/Future-Lens outputs already computed.
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/jkchoi/project/autopaper/sandbox/p2-horizon-rollout-encoder")
from p2_bootstrap import paired_bootstrap_ci  # noqa: E402

P2_ART = "/home/jkchoi/project/autopaper/sandbox/p2-horizon-rollout-encoder/artifacts"
ART = "/home/jkchoi/project/autopaper/sandbox/p4-future-decoding-comparison/artifacts"
os.makedirs(ART, exist_ok=True)

MODEL_NAME = "EleutherAI/pythia-1.4b"
LAYERS_TUNED_LENS = [4, 8, 12, 20]
PRIMARY_LAYER = 20
FUTURE_OFFSETS = list(range(1, 11))          # m=1..10 for training/native validation
FUTURE_M_SEMANTIC = [3, 5, 10]                # required semantic-reconstruction budgets
NATIVE_VAL_OFFSETS = [1, 2, 3, 5, 10]         # required native-accuracy table rows
HORIZONS = [16, 48, 96]
ROLLOUT_M = [3, 5, 10]
COMMON_HORIZON = 256
SEED = 42
ALPHA_GRID_PROBE = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
TOPK_VALUES = [1, 5, 10]
N_BOOT = 10000
DEVICE = "cuda"


def log(msg):
    print(f"[p4] {msg}", flush=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cosine_rows(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


# ---------------------------------------------------------------------------
def load_data():
    records = read_jsonl(f"{P2_ART}/generations.jsonl")
    lens = np.array([r["realized_len"] for r in records])
    hidden_npz = np.load(f"{P2_ART}/hidden_last_token.npz")
    hidden_by_layer_all = {L: hidden_npz[f"layer_{L}"] for L in LAYERS_TUNED_LENS}

    common_idx = np.where(lens >= COMMON_HORIZON)[0]
    n_common = len(common_idx)
    common_records = [records[i] for i in common_idx]
    hidden_by_layer = {L: hidden_by_layer_all[L][common_idx] for L in LAYERS_TUNED_LENS}

    idx_all = np.arange(n_common)
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.15, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15 / 0.85, random_state=SEED)
    log(f"common subset n={n_common}; split train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} "
        f"(must match P2 exactly: {len(idx_train)==403 and len(idx_val)==87 and len(idx_test)==87})")
    return common_records, hidden_by_layer, idx_train, idx_val, idx_test


def decode_prefix(tok, ids, k):
    return tok.decode(ids[:k], skip_special_tokens=True)


def embed_texts(embedder, texts, batch_size=64):
    return embedder.encode(texts, batch_size=batch_size, show_progress_bar=False,
                            normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def build_topk_texts(tokenizer, logits_row, k):
    probs = torch.softmax(logits_row, dim=-1)
    topp, topi = torch.topk(probs, k)
    toks = [tokenizer.decode([tid]) for tid in topi.tolist()]
    concat_text = "".join(toks).strip() or " ".join(t.strip() for t in toks)
    weighted_parts = []
    for tok_, p in zip(toks, topp.tolist()):
        reps = max(1, round(10 * p))
        weighted_parts.extend([tok_.strip() or tok_] * reps)
    weighted_text = " ".join(weighted_parts)
    return concat_text if concat_text.strip() else "(empty)", weighted_text if weighted_text.strip() else "(empty)"


# ---------------------------------------------------------------------------
class AffineTranslator(nn.Module):
    """h~ = A h + b, per the spec's Tuned Lens formula, reused unchanged for
    the Future-Lens-style per-offset translators (documented deviation from
    the published Future Lens raw-linear-probe formulation)."""
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        nn.init.zeros_(self.lin.weight)
        with torch.no_grad():
            self.lin.weight.add_(torch.eye(dim))  # init near-identity, standard tuned-lens practice
        nn.init.zeros_(self.lin.bias)

    def forward(self, h):
        return self.lin(h)


def train_translator(model, final_norm, lm_head, X_train, y_train_ids, X_val, y_val_ids,
                      dim, max_epochs=300, patience=15, lr=1e-3, seed=SEED, tag=""):
    torch.manual_seed(seed)
    translator = AffineTranslator(dim).to(DEVICE).float()
    opt = torch.optim.Adam(translator.parameters(), lr=lr, weight_decay=1e-4)
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_train_ids, dtype=torch.long, device=DEVICE)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    yva = torch.tensor(y_val_ids, dtype=torch.long, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(max_epochs):
        translator.train()
        opt.zero_grad()
        h_tilde = translator(Xtr)
        with torch.no_grad():
            pass
        normed = final_norm(h_tilde)
        logits = lm_head(normed)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()

        translator.eval()
        with torch.no_grad():
            h_tilde_v = translator(Xva)
            normed_v = final_norm(h_tilde_v)
            logits_v = lm_head(normed_v)
            val_loss = loss_fn(logits_v, yva).item()
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in translator.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    translator.load_state_dict(best_state)
    translator.eval()
    log(f"  [{tag}] converged epoch~{epoch} best_val_loss={best_val_loss:.4f}")
    return translator, best_val_loss


def get_logits(translator, final_norm, lm_head, X):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        h_tilde = translator(Xt)
        normed = final_norm(h_tilde)
        logits = lm_head(normed)
    return logits.float().cpu()


def raw_logits(final_norm, lm_head, X):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        normed = final_norm(Xt)
        logits = lm_head(normed)
    return logits.float().cpu()


def topk_variant_texts(tokenizer, logits, ks=TOPK_VALUES):
    variants = {}
    for k in ks:
        concat_texts, weighted_texts = [], []
        for row in range(logits.shape[0]):
            c, w = build_topk_texts(tokenizer, logits[row], k)
            concat_texts.append(c)
            weighted_texts.append(w)
        variants[f"topk_concat_k{k}"] = concat_texts
        variants[f"topk_weighted_k{k}"] = weighted_texts
    return variants


def select_best_variant(embedder, variants, y_val):
    best_name, best_cos, best_val_emb = None, -2.0, None
    for name, texts in variants.items():
        emb = embed_texts(embedder, texts)
        cos = float(np.mean(cosine_rows(emb, y_val)))
        if cos > best_cos:
            best_cos, best_name = cos, name
    return best_name, best_cos


def main():
    t0 = time.time()
    common_records, hidden_by_layer, idx_train, idx_val, idx_test = load_data()
    n_train, n_val, n_test = len(idx_train), len(idx_val), len(idx_test)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    final_norm = model.gpt_neox.final_layer_norm
    lm_head = model.get_output_embeddings()
    vocab_size = model.config.vocab_size
    log(f"model loaded fp32; vocab_size={vocab_size}")

    # ---------------- target texts / embeddings at each horizon ----------------
    texts_by_H = {H: [decode_prefix(tok, r["gen_token_ids"], H) for r in common_records] for H in HORIZONS}
    y_by_H = {H: embed_texts(embedder, texts_by_H[H]) for H in HORIZONS}
    log("embedded horizon targets H=" + ",".join(map(str, HORIZONS)))

    # rollout texts/embeddings (reused trajectory, no new generation)
    texts_by_m = {m: [decode_prefix(tok, r["gen_token_ids"], m) for r in common_records] for m in ROLLOUT_M}
    rollout_emb_by_m = {m: embed_texts(embedder, texts_by_m[m]) for m in ROLLOUT_M}

    # ground-truth offset token ids (for tuned lens j=1 and future-lens j=1..10)
    offset_token_ids = {}
    for j in FUTURE_OFFSETS:
        offset_token_ids[j] = np.array([r["gen_token_ids"][j - 1] for r in common_records], dtype=np.int64)

    # ================= M1 Direct Logit Lens =================
    log("=== M1 Direct Logit Lens ===")
    raw_logits_L20_test = raw_logits(final_norm, lm_head, hidden_by_layer[PRIMARY_LAYER][idx_test])
    raw_logits_L20_val = raw_logits(final_norm, lm_head, hidden_by_layer[PRIMARY_LAYER][idx_val])
    variants_val = topk_variant_texts(tok, raw_logits_L20_val)
    y16_val = y_by_H[16][idx_val]
    m1_best_name, m1_val_cos = select_best_variant(embedder, variants_val, y16_val)
    variants_test = topk_variant_texts(tok, raw_logits_L20_test)
    m1_test_emb = embed_texts(embedder, variants_test[m1_best_name])
    m1_test_per_ex = cosine_rows(m1_test_emb, y_by_H[16][idx_test])
    log(f"M1 best variant (val-selected): {m1_best_name}, val_cos={m1_val_cos:.4f}, test_cos={np.mean(m1_test_per_ex):.4f}")

    # ================= M2 Tuned Lens =================
    log("=== M2 Tuned Lens ===")
    tuned_lens_results = {}
    for L in LAYERS_TUNED_LENS:
        translator, val_loss = train_translator(
            model, final_norm, lm_head,
            hidden_by_layer[L][idx_train], offset_token_ids[1][idx_train],
            hidden_by_layer[L][idx_val], offset_token_ids[1][idx_val],
            dim=hidden_by_layer[L].shape[1], tag=f"TunedLens-L{L}")
        tl_logits_val = get_logits(translator, final_norm, lm_head, hidden_by_layer[L][idx_val])
        tl_variants_val = topk_variant_texts(tok, tl_logits_val)
        best_name, val_cos = select_best_variant(embedder, tl_variants_val, y_by_H[16][idx_val])
        tl_logits_test = get_logits(translator, final_norm, lm_head, hidden_by_layer[L][idx_test])
        tl_variants_test = topk_variant_texts(tok, tl_logits_test)
        test_emb = embed_texts(embedder, tl_variants_test[best_name])
        test_per_ex = cosine_rows(test_emb, y_by_H[16][idx_test])
        # native next-token accuracy (val loss already gives training-objective validation)
        with torch.no_grad():
            pred_ids_test = tl_logits_test.argmax(dim=-1).numpy()
        top1_acc = float(np.mean(pred_ids_test == offset_token_ids[1][idx_test]))
        tuned_lens_results[L] = {
            "translator": translator, "best_variant": best_name, "val_cosine": val_cos,
            "test_cosine": float(np.mean(test_per_ex)), "test_per_example": test_per_ex,
            "val_ce_loss": val_loss, "next_token_top1_acc_test": top1_acc,
        }
        log(f"TunedLens L{L}: best_variant={best_name} val_cos={val_cos:.4f} test_cos={np.mean(test_per_ex):.4f} "
            f"next-token top1 acc={top1_acc:.4f}")

    # ================= M3 Future-Lens-style =================
    log("=== M3 Future-Lens-style (layer 20 only, offsets 1..10) ===")
    fl_translators = {}
    fl_native = {}
    Xtr20, Xva20, Xte20 = hidden_by_layer[PRIMARY_LAYER][idx_train], hidden_by_layer[PRIMARY_LAYER][idx_val], hidden_by_layer[PRIMARY_LAYER][idx_test]
    for j in FUTURE_OFFSETS:
        translator, val_loss = train_translator(
            model, final_norm, lm_head, Xtr20, offset_token_ids[j][idx_train],
            Xva20, offset_token_ids[j][idx_val], dim=Xtr20.shape[1], tag=f"FutureLens-j{j}")
        logits_test = get_logits(translator, final_norm, lm_head, Xte20)
        probs_test = torch.softmax(logits_test, dim=-1)
        top1_pred = probs_test.argmax(dim=-1).numpy()
        top5_pred = torch.topk(probs_test, 5, dim=-1).indices.numpy()
        true_ids = offset_token_ids[j][idx_test]
        top1_acc = float(np.mean(top1_pred == true_ids))
        top5_acc = float(np.mean([true_ids[i] in top5_pred[i] for i in range(len(true_ids))]))
        fl_translators[j] = translator
        fl_native[j] = {"top1_acc": top1_acc, "top5_acc": top5_acc, "val_ce_loss": val_loss, "top1_pred_ids": top1_pred}
        log(f"FutureLens j={j}: top1_acc={top1_acc:.4f} top5_acc={top5_acc:.4f} val_ce={val_loss:.4f}")

    # semantic reconstruction: concatenate top-1 predicted tokens for offsets 1..m
    fl_semantic = {}
    for m in FUTURE_M_SEMANTIC:
        seqs = []
        for i in range(n_test):
            ids = [int(fl_native[j]["top1_pred_ids"][i]) for j in range(1, m + 1)]
            seqs.append(tok.decode(ids, skip_special_tokens=True).strip() or "(empty)")
        emb = embed_texts(embedder, seqs)
        per_ex = cosine_rows(emb, y_by_H[16][idx_test])
        fl_semantic[m] = {"test_cosine": float(np.mean(per_ex)), "test_per_example": per_ex}
        log(f"FutureLens semantic m={m}: test_cos={np.mean(per_ex):.4f}")

    # ================= M6 Linear Semantic Probe (H=16, layer 20; cross-checked vs P2) =================
    log("=== M6 Linear Semantic Probe ===")

    def fit_ridge(X, y, idx_tr, idx_va, idx_te):
        best_alpha, best_val_cos, best_model = None, -2.0, None
        for a in ALPHA_GRID_PROBE:
            m_ = Ridge(alpha=a, random_state=SEED)
            m_.fit(X[idx_tr], y[idx_tr])
            vc = float(np.mean(cosine_rows(m_.predict(X[idx_va]), y[idx_va])))
            if vc > best_val_cos:
                best_val_cos, best_alpha, best_model = vc, a, m_
        pred_test = best_model.predict(X[idx_te])
        per_ex = cosine_rows(pred_test, y[idx_te])
        return {"best_alpha": best_alpha, "val_cosine": best_val_cos, "test_cosine": float(np.mean(per_ex)),
                "test_per_example": per_ex, "model": best_model}

    probe_L20_H16 = fit_ridge(hidden_by_layer[PRIMARY_LAYER], y_by_H[16], idx_train, idx_val, idx_test)
    log(f"Probe (L20,H16) test_cos={probe_L20_H16['test_cosine']:.4f} "
        f"(P2 saved value for cross-check: 0.4504)")

    # ================= M8 MLP Semantic Probe (optional diagnostic) =================
    log("=== M8 MLP Semantic Probe (diagnostic) ===")
    best_mlp, best_mlp_val_cos, best_alpha = None, -2.0, None
    for alpha in [1e-2, 1e-1, 1.0]:
        mlp = MLPRegressor(hidden_layer_sizes=(512,), activation="relu", alpha=alpha,
                            max_iter=800, random_state=SEED, early_stopping=False)
        mlp.fit(hidden_by_layer[PRIMARY_LAYER][idx_train], y_by_H[16][idx_train])
        val_cos = float(np.mean(cosine_rows(mlp.predict(hidden_by_layer[PRIMARY_LAYER][idx_val]), y_by_H[16][idx_val])))
        if val_cos > best_mlp_val_cos:
            best_mlp_val_cos, best_mlp, best_alpha = val_cos, mlp, alpha
    mlp_test_pred = best_mlp.predict(hidden_by_layer[PRIMARY_LAYER][idx_test])
    mlp_test_per_ex = cosine_rows(mlp_test_pred, y_by_H[16][idx_test])
    log(f"MLP probe: best_alpha={best_alpha} val_cos={best_mlp_val_cos:.4f} test_cos={np.mean(mlp_test_per_ex):.4f}")

    # ================= Rollout reference (H=16, recomputed for per-example arrays) =================
    log("=== Rollout reference (H=16) ===")
    rollout_per_ex = {}
    for m in ROLLOUT_M:
        emb_test = rollout_emb_by_m[m][idx_test]
        per_ex = cosine_rows(emb_test, y_by_H[16][idx_test])
        rollout_per_ex[m] = per_ex
        log(f"Rollout m={m}: test_cos={np.mean(per_ex):.4f}")

    # ================= Bootstrap =================
    log("=== Bootstrap (10000 resamples, H=16) ===")
    boot = {
        "probe_minus_tunedlens_L20": paired_bootstrap_ci(probe_L20_H16["test_per_example"], tuned_lens_results[PRIMARY_LAYER]["test_per_example"], n_boot=N_BOOT, seed=SEED),
    }
    for m in FUTURE_M_SEMANTIC:
        boot[f"probe_minus_futurelens_m{m}"] = paired_bootstrap_ci(probe_L20_H16["test_per_example"], fl_semantic[m]["test_per_example"], n_boot=N_BOOT, seed=SEED)
    for m in ROLLOUT_M:
        boot[f"probe_minus_rollout_m{m}"] = paired_bootstrap_ci(probe_L20_H16["test_per_example"], rollout_per_ex[m], n_boot=N_BOOT, seed=SEED)
    for k, v in boot.items():
        log(f"  {k}: point={v['point']:.4f} CI=[{v['ci_low']:.4f},{v['ci_high']:.4f}] excl0={v['excludes_zero']}")

    # ================= Horizon robustness (H=48,96): reuse decoders, only re-embed target =================
    log("=== Horizon robustness H=48,96 (reusing trained decoders) ===")
    horizon_table = {16: {
        "probe": probe_L20_H16["test_cosine"], "tuned_lens": tuned_lens_results[PRIMARY_LAYER]["test_cosine"],
        **{f"future_lens_m{m}": fl_semantic[m]["test_cosine"] for m in FUTURE_M_SEMANTIC},
        **{f"rollout_m{m}": float(np.mean(rollout_per_ex[m])) for m in ROLLOUT_M},
    }}
    for H in [48, 96]:
        yH_test = y_by_H[H][idx_test]
        # probe: retrain at this horizon (probe target changes with H) -- cheap ridge refit
        probeH = fit_ridge(hidden_by_layer[PRIMARY_LAYER], y_by_H[H], idx_train, idx_val, idx_test)
        # tuned lens: reuse fixed best-variant TEXT (decoder output doesn't depend on H), re-embed vs new target
        tl_text = topk_variant_texts(tok, get_logits(tuned_lens_results[PRIMARY_LAYER]["translator"], final_norm, lm_head, Xte20))[tuned_lens_results[PRIMARY_LAYER]["best_variant"]]
        tl_emb = embed_texts(embedder, tl_text)
        tl_cos = float(np.mean(cosine_rows(tl_emb, yH_test)))
        # future lens: reuse fixed top-1 sequences, re-embed vs new target
        fl_cos = {}
        for m in FUTURE_M_SEMANTIC:
            seqs = []
            for i in range(n_test):
                ids = [int(fl_native[j]["top1_pred_ids"][i]) for j in range(1, m + 1)]
                seqs.append(tok.decode(ids, skip_special_tokens=True).strip() or "(empty)")
            emb = embed_texts(embedder, seqs)
            fl_cos[m] = float(np.mean(cosine_rows(emb, yH_test)))
        # rollout: reuse fixed rollout text, re-embed vs new target
        roll_cos = {m: float(np.mean(cosine_rows(rollout_emb_by_m[m][idx_test], yH_test))) for m in ROLLOUT_M}
        horizon_table[H] = {"probe": probeH["test_cosine"], "tuned_lens": tl_cos,
                             **{f"future_lens_m{m}": fl_cos[m] for m in FUTURE_M_SEMANTIC},
                             **{f"rollout_m{m}": roll_cos[m] for m in ROLLOUT_M}}
        log(f"H={H}: probe={probeH['test_cosine']:.4f} tuned_lens={tl_cos:.4f} " +
            " ".join(f"FL{m}={fl_cos[m]:.4f}" for m in FUTURE_M_SEMANTIC) + " " +
            " ".join(f"R{m}={roll_cos[m]:.4f}" for m in ROLLOUT_M))

    # ================= Save everything =================
    results = {
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "M1_direct_logit_lens": {"best_variant": m1_best_name, "val_cosine": m1_val_cos, "test_cosine": float(np.mean(m1_test_per_ex))},
        "M2_tuned_lens": {str(L): {k: v for k, v in tuned_lens_results[L].items() if k not in ("translator", "test_per_example")} for L in LAYERS_TUNED_LENS},
        "M3_future_lens_native": {str(j): {k: v for k, v in fl_native[j].items() if k != "top1_pred_ids"} for j in FUTURE_OFFSETS},
        "M3_future_lens_semantic": {str(m): {"test_cosine": fl_semantic[m]["test_cosine"]} for m in FUTURE_M_SEMANTIC},
        "M6_linear_probe_L20_H16": {"val_cosine": probe_L20_H16["val_cosine"], "test_cosine": probe_L20_H16["test_cosine"], "best_alpha": probe_L20_H16["best_alpha"]},
        "M8_mlp_probe_L20_H16": {"best_alpha": best_alpha, "val_cosine": best_mlp_val_cos, "test_cosine": float(np.mean(mlp_test_per_ex))},
        "rollout_H16": {str(m): float(np.mean(rollout_per_ex[m])) for m in ROLLOUT_M},
        "bootstrap": boot,
        "horizon_table": horizon_table,
    }
    with open(f"{ART}/p4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\ndone in {time.time()-t0:.1f}s. Saved {ART}/p4_results.json")


if __name__ == "__main__":
    main()

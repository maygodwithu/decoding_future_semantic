# P1 — Semantic Target Robustness — Report

**Project**: `semantic-target-robustness` | **Anchor reused**: `hidden-state-semantic-lookahead` (Pythia-1.4B, greedy, 549 usable prompts, split seed 42, no new generation or hidden-state extraction performed) | **Runtime**: ~90s (encoder inference + ridge fits only)

Encoders tested: **E1** `sentence-transformers/all-MiniLM-L6-v2` (anchor, no prefix), **E2** `BAAI/bge-base-en-v1.5` (no prefix — BGE's instruction prefix applies to the query side of asymmetric retrieval, not to passage/document encoding, so continuations are encoded plain per BGE's documented convention), **E3** `intfloat/e5-base-v2` (`"passage: "` prefix on every continuation — required by E5's training convention).

## A. Summary table

| Encoder | Best layer | Probe | Mean | Random | TF-IDF | Best logit-lens | Rollout-3 | Rollout-5 | R@1 | R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 MiniLM (anchor) | 20 | 0.4283 | 0.2310 | 0.0616 | 0.3612 | 0.1812 | 0.4757 | 0.6371 | 0.398 | 0.735 |
| E2 BGE-base | 20 | 0.7034 | 0.6364 | 0.4080 | 0.6624 | 0.5135 | 0.6818 | 0.7699 | 0.205 | 0.651 |
| E3 E5-base | 20 | 0.8574 | 0.8283 | 0.6888 | 0.8455 | 0.7512 | 0.8209 | 0.8669 | 0.301 | 0.614 |

Layer sweep, all four candidate layers: identical qualitative ordering (4 < 8 < 12 < 20) and identical best layer (20) for all three encoders — no layer-choice divergence in this experiment, unlike the causal-steering layer dissociation reported elsewhere in the paper.

**Raw cosine is not comparable across rows** (see §11 of the spec and Finding E below) — BGE and E5 have much higher absolute cosine everywhere, including for random-match, because of a known anisotropy property of those embedding spaces, not because they encode more signal.

**Protocol note**: Table A uses the **train-only probe fit** (ridge fit on the 383-example train split, alpha selected on val, evaluated once on the 83-example test split) — this is the **canonical protocol used consistently throughout the journal manuscript**, across passive probing (§4.1/§4.4), encoder robustness (this experiment), and every cross-model comparison (Pythia-1.0B/1.4B/2.8B, Qwen2.5, Qwen3). It is not the same fitting procedure as the final trainval-refit probe originally saved for the §4.1 MiniLM anchor artifact (`best_probe_trainval_layer_20.npz`); see A.1 below for that comparison, kept separate deliberately so the two protocols are never mixed in the paper's reported numbers.

### A.1 Final-refit diagnostic (train+val refit — NOT the paper protocol)

For completeness/auditing only, the same layer (20) and already-selected alpha (10000.0, unchanged from the train-only sweep) were used to refit each encoder's ridge probe on train+val combined (466 examples) and evaluate once on the same 83-example test split, reusing all cached hidden states and continuation embeddings (no new generation or hidden-state extraction):

| Encoder | Probe cosine | R@1 | R@5 | Mean rank |
|---|---:|---:|---:|---:|
| MiniLM | 0.4401 | 0.4337 | 0.7590 | 4.77 |
| BGE-base | 0.7068 | 0.2410 | 0.6386 | 10.66 |
| E5-base | 0.8593 | 0.3253 | 0.6024 | 8.40 |

MiniLM's final-refit numbers reproduce the original §4.1 anchor artifact's retrieval values exactly (R@1=0.4337, R@5=0.7590, mean rank=4.77), confirming the two protocols are both internally reproducible and differ only in how much data the final probe is fit on. Note the refit does **not** move every encoder the same direction: probe cosine rises for all three (more training data), but Recall@5 rises for MiniLM (0.735→0.759) while it slightly *falls* for BGE (0.651→0.639) and E5 (0.614→0.602) — continuous-cosine quality and discrete retrieval ranking do not always move together, plausibly related to BGE/E5's embedding-space anisotropy (Finding E) making rankings more sensitive to small prediction changes. **This table is a diagnostic cross-check, not a source of numbers for the paper** — all reported results use the train-only protocol in Table A.

## B. Shuffled-target table

| Encoder | Normal | Shuffled | Difference |
|---|---:|---:|---:|
| E1 MiniLM | 0.4283 | 0.2316 | 0.1967 |
| E2 BGE | 0.7034 | 0.6363 | 0.0671 |
| E3 E5 | 0.8574 | 0.8282 | 0.0292 |

(Permutation seed 42, train-targets-only shuffle, evaluated against the true unshuffled test targets, all three encoders.)

## C. Bootstrap table (10,000-resample paired bootstrap, 95% CI)

| Encoder | Quantity | Point | 95% CI | Excludes 0? |
|---|---|---:|---|:---:|
| E1 | probe cosine | 0.4283 | [0.398, 0.460] | Yes |
| E1 | probe − lexical TF-IDF (strongest weak/token baseline) | 0.0671 | [0.043, 0.091] | **Yes** |
| E1 | probe − rollout m=3 | −0.0475 | [−0.097, 0.002] | No |
| E1 | probe − rollout m=5 | −0.2089 | [−0.260, −0.157] | Yes (negative) |
| E2 | probe cosine | 0.7034 | [0.688, 0.719] | Yes |
| E2 | probe − lexical TF-IDF | 0.0410 | [0.034, 0.048] | **Yes** |
| E2 | probe − rollout m=3 | 0.0217 | [−0.002, 0.046] | No |
| E2 | probe − rollout m=5 | −0.0664 | [−0.091, −0.042] | Yes (negative) |
| E3 | probe cosine | 0.8574 | [0.851, 0.863] | Yes |
| E3 | probe − lexical TF-IDF | 0.0119 | [0.008, 0.016] | **Yes, but small** |
| E3 | probe − rollout m=3 | 0.0365 | [0.025, 0.048] | **Yes (positive)** |
| E3 | probe − rollout m=5 | −0.0095 | [−0.023, 0.004] | No (parity) |

## D. Plot

`artifacts/p1_encoder_comparison.png` — probe / best weak-or-token-identity baseline / rollout-3 / rollout-5, grouped by encoder (bars are within-encoder comparisons only; axis is explicitly not a cross-encoder scale).

## E. Written verdict

**1. Is semantic recoverability robust to the choice of sentence encoder?**
**Partial support.** The core claim — that the probe beats mean/random/lexical/token-identity baselines — replicates in all three encoders with bootstrap CIs excluding zero. But the size of that advantage is not encoder-invariant: it shrinks by roughly an order of magnitude from MiniLM (probe−lexical margin 0.067) to E5 (0.012), and retrieval-based discriminability (R@5) is actually *highest* for MiniLM (0.735) and *lowest* for E5 (0.614) despite E5 having the highest raw probe cosine (0.857) of the three. Raw cosine and retrieval-discriminability diverge across encoders because BGE and E5 have markedly higher baseline (random-match) similarity — 0.408 and 0.689 respectively, versus MiniLM's 0.062 — a known anisotropy property of those embedding families, not evidence of stronger recoverability. We recommend the paper report R@5, not raw cosine margin, as the primary cross-encoder-robustness metric, and note this anisotropy explicitly rather than let a reader assume "higher cosine = more meaning recovered."

**2. Does the probe continue to beat weak and token-identity baselines?**
**Yes, in all three encoders**, with the probe−best-baseline bootstrap CI excluding zero every time (E1 [0.043,0.091], E2 [0.034,0.048], E3 [0.008,0.016]). This part of the claim is safe to state without qualification.

**3. Does the strong rollout-baseline finding remain?**
**No — this is the most important result of P1, and it does not replicate uniformly.** Under MiniLM, the probe clearly loses to rollout-m5 (CI excludes zero, negative) and is statistically tied with rollout-m3 (CI includes zero, point estimate slightly negative) — consistent with the paper's existing §4.2 framing. Under BGE, the pattern softens: the probe is statistically tied with rollout-m3 (point estimate now *positive*, +0.022, though CI still includes zero) and still loses to rollout-m5. Under E5, the pattern partially **reverses**: the probe significantly *beats* rollout-m3 (CI excludes zero, positive, [0.025, 0.048]) and is at statistical parity with rollout-m5 (CI includes zero, near-zero point estimate). The paper's central methodological argument in §4.2–4.3 — "a fair behavioral baseline beats the probe at short horizons" — is real under MiniLM but is not an encoder-invariant fact about the underlying phenomenon; it is at least partly a property of how much of the rollout's local surface information the specific target encoder rewards. We recommend explicitly scoping the §4.2/4.3 rollout-crossover claim to the MiniLM target space in the main text, and reporting this cross-encoder instability as a limitation/robustness finding rather than folding it silently into the headline claim.

**4. Does shuffled-target training destroy recoverability?**
**Yes, in all three encoders** — normal probe cosine exceeds shuffled-target cosine by a wide margin in every case (E1: 0.428→0.232; E2: 0.703→0.636; E3: 0.857→0.828), confirming the hidden state is genuinely predictive of *which* continuation occurred and not merely of the marginal distribution of continuation embeddings. Even E5's smaller absolute gap (0.029) is directionally consistent and larger than its own probe−lexical bootstrap margin, so we do not read it as evidence of a marginal-distribution artifact specifically for E5.

**5. Are there any encoder-specific anomalies worth reporting?**
Yes, two, both already noted above and worth a short paragraph in the IEEE Access robustness section rather than a footnote:
- **BGE and E5 embedding spaces are markedly anisotropic** relative to MiniLM (random-match baseline cosine 0.41 / 0.69 vs. 0.06), which compresses every margin computed in raw-cosine terms and makes retrieval (R@k) the more trustworthy cross-encoder metric.
- **The rollout-vs-probe crossover point is encoder-dependent**, shifting from "probe already loses by m=3" (MiniLM) to "probe beats m=3, ties m=5" (E5) — the single biggest qualitative difference found in this experiment, and the one most likely to draw a reviewer question if left unaddressed.

**Overall recommendation for the IEEE Access draft**: the paper's headline recoverability claim ("hidden states linearly encode continuation semantics, beating weak/token-identity baselines") is safe to state as encoder-general — it replicates cleanly. The rollout-baseline crossover claim (§4.2/4.3) should be explicitly scoped to the MiniLM target space used in the main results, with this P1 robustness check cited as showing the qualitative comparison to a behavioral baseline is target-encoder-dependent — an honest, reportable nuance rather than a threat to the paper's core contribution.

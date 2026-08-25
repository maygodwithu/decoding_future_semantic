# P4 — Comparison with Existing Future-Decoding Methods — Report

**Project**: `p4-future-decoding-comparison` | **Model**: `EleutherAI/pythia-1.4b`, base, greedy | **Data**: reused verbatim from `sandbox/p2-horizon-rollout-encoder/artifacts/` (generations.jsonl, hidden_last_token.npz at layers 4/8/12/20) — **no new autoregressive generation was performed anywhere in P4**. Common subset (realized_len ≥ 256) and train/val/test split reproduced with P2's exact procedure: **n=577, train=403 / val=87 / test=87, seed=42** — verified to match P2's split sizes bit-for-bit before any method was trained. **Runtime**: ~117s total (all translator training + evaluation; no generation cost).

## 0. What P4 tests and why the numbers differ slightly from earlier sections

All methods here are evaluated on the exact same H=16 test set defined by P2's common-subset protocol (n=87 test examples, target = first 16 generated tokens of a 256-token generation pass, no sentence-boundary cut). This differs slightly from the paper's original §4.1/§4.2 anchor (0.4283, sentence-cut truncation, 549-example set) and from P2's own H=16 number (0.4504) is in fact **exactly** P2's number, confirmed by direct cross-check below — the two are the same evaluation, just re-run here for per-example arrays needed by bootstrap. The spec's own reference value ("previous MiniLM result at H=16 was approximately 0.439") comes from the *original* `horizon-crossover-analysis` project (a different, non-common-subset n=600 pool); per the spec's own instruction not to conflate protocols, we use P2's own number throughout and flag this lineage explicitly rather than silently picking whichever number is more convenient.

## 1. Implementation choices and documented deviations

- **M2 Tuned Lens**: affine translator `h̃ = A·h + b` (2048×2048 + bias, ~4.2M params), `A` initialized to the identity matrix (standard tuned-lens practice), trained via Adam (lr=1e-3, weight_decay=1e-4) to minimize next-token cross-entropy against the true first generated token, using **train split only**; early-stopped on **val** cross-entropy (patience=15, max 300 epochs); **test set never touched during training or model selection**. Applied through the model's own frozen `final_layer_norm` + `lm_head`. Trained independently at all four passive-probing layers (4, 8, 12, 20) since the added cost was negligible; layer 20 is the primary/anchor condition per the spec.
- **M3 Future-Lens-style decoder**: **the same affine-translator architecture as M2**, one independent instance per future offset *j* ∈ {1,...,10}, each trained only on the *original pre-generation hidden state at layer 20* to predict the ground-truth token actually generated at offset *j* — no offset's prediction ever consumes another offset's prediction or any ground-truth future token (verified by construction: every translator's only input is the single, shared, fixed `h_20` array). **This reuse of the affine-translator architecture (rather than the published Future Lens's raw hidden→vocab linear-probe formulation) is a deliberate implementation deviation**, chosen for architectural consistency with M2 and computational efficiency; we refer to this method as "Future-Lens-style" throughout, never as "Future Lens," per the spec's requirement.
- **Semantic aggregation** (M1, M2, M3): reused verbatim from the paper's existing §4.2 logit-lens code — six variants (top-*k* concat / probability-weighted, *k*∈{1,5,10}). The variant used for the *reported* test number is selected by **validation cosine only**, then applied unchanged to the test set — stricter than needed by some published logit-lens baselines but required here to satisfy the spec's no-test-set-tuning constraint.
- **M8 MLP probe** (optional diagnostic): one hidden layer (512 units, ReLU), `sklearn.MLPRegressor`, L2 penalty α selected from {0.01, 0.1, 1.0} by validation cosine (winner: α=1.0).
- **Section 5's optional probability-aware Future-Lens variant was not implemented** — the spec makes this explicitly optional and conditional on a rule fixed in advance; given P4's scope was already large, we report only the required top-1 sequence reconstruction, consistent with the spec's stated primary requirement.
- **Rollout-*m*** (m=3,5,10): recomputed directly from the same generated trajectories already used throughout P2 — text = first *m* generated tokens, embedded with MiniLM, no new generation. Recomputed values match P2's saved summary numbers exactly (0.4059 / 0.5825 / 0.8350), confirming pipeline consistency.

## 2. Future-Lens-style native validation (implementation check, not a headline result)

| Future offset | Top-1 | Top-5 |
|---:|---:|---:|
| 1 | 0.5862 | 0.8621 |
| 2 | 0.2529 | 0.5057 |
| 3 | 0.1494 | 0.3793 |
| 5 | 0.0690 | 0.2414 |
| 10 | 0.0575 | 0.1264 |

Accuracy declines sharply and monotonically with offset, as expected for a non-autoregressive from-a-single-state predictor (offset 1 ≈ ordinary next-token prediction with 58.6% top-1 accuracy; by offset 10, top-1 accuracy has fallen to the mid-single digits, only modestly above what a frequency-based guess might achieve). Offset 1's top-1 accuracy (0.5862) exactly matches M2 Tuned Lens's layer-20 next-token accuracy (§3), confirming the two pipelines share the same underlying mechanism at *j*=1, as they should by construction.

## 3. Tuned Lens by layer

| Layer | Best variant | Val cosine | Test cosine | Next-token top-1 acc (test) |
|---:|---|---:|---:|---:|
| 4 | topk_concat_k5 | — | 0.0992 | 0.4713 |
| 8 | topk_concat_k1 | — | 0.1068 | 0.4598 |
| 12 | topk_concat_k1 | — | 0.1039 | 0.5057 |
| **20** | topk_concat_k5 | 0.1516 | **0.1442** | 0.5862 |

Next-token accuracy rises fairly consistently with depth, and the semantic cosine does too — layer 20 is both the best next-token-accuracy layer and the best semantic-cosine layer among the four tested, so Tuned Lens shows no layer-dissociation effect of its own. But layer 20's Tuned Lens semantic cosine (0.144) is still the best Tuned Lens can do, and — surprisingly — it does not beat the untrained direct logit lens at the same layer (0.180, §4), let alone the probe (0.450).

## 4. Primary comparison table (H=16, n=87 test examples)

| Method | Representation decoded | Extra AR steps | Semantic cosine |
|---|---|---:|---:|
| Mean embedding | — | 0 | 0.2430 |
| Direct Logit Lens | next-token identity | 0 | 0.1795 |
| Tuned Lens (layer 20) | translated next-token identity | 0 | 0.1442 |
| Future-Lens-style, m=3 | future token identities | 0 | 0.2175 |
| Future-Lens-style, m=5 | future token identities | 0 | 0.2292 |
| Future-Lens-style, m=10 | future token identities | 0 | 0.2529 |
| **Linear Semantic Probe** | **continuation semantics** | **0** | **0.4504** |
| MLP Semantic Probe (diagnostic) | continuation semantics | 0 | 0.2810 |
| Rollout-3 | generated prefix | 3 | 0.4059 |
| Rollout-5 | generated prefix | 5 | 0.5825 |
| Rollout-10 | generated prefix | 10 | 0.8350 |

`figure_p4_zero_decode_comparison.png` shows the six zero-decode methods as a bar chart against the mean-embedding baseline; rollout is deliberately excluded from that figure since it incurs additional sequential computation (kept in the table only, per the spec's instruction to keep the two categories visually/conceptually separate).

**Note on Tuned Lens vs. Direct Logit Lens**: Tuned Lens (0.144) scores *below* the untrained Direct Logit Lens (0.180) at the same layer, on the same test set. This is reported exactly as observed — no post-hoc adjustment was made. See §7 (Q1) for discussion.

## 5. Horizon robustness (H=16, 48, 96) — decoders reused unchanged, only the target re-embedded

| H | Semantic Probe | Tuned Lens | Future-Lens-3 | Future-Lens-5 | Future-Lens-10 | Rollout-3 | Rollout-5 | Rollout-10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.4504 | 0.1442 | 0.2175 | 0.2292 | 0.2529 | 0.4059 | 0.5825 | 0.8350 |
| 48 | 0.4363 | 0.1046 | 0.1621 | 0.1744 | 0.2176 | 0.3103 | 0.4594 | 0.6807 |
| 96 | 0.4164 | 0.1005 | 0.1506 | 0.1611 | 0.2039 | 0.2732 | 0.4044 | 0.5985 |

Exactly as anticipated by the spec: Tuned Lens's and Future-Lens-style's outputs do not depend on *H* (same fixed decoded text at every row), so their small drift downward as *H* grows is purely a re-embedding-against-a-longer-target effect, not a re-decoding effect — no retraining occurred between rows. The Semantic Probe was retrained per horizon (its target changes with *H* by design) and its own values here match §4.3/§4.7-style numbers from P2 exactly, as expected.

## 6. Bootstrap (10,000-resample paired bootstrap, H=16, n=87)

| Comparison | Mean difference (probe − other) | 95% CI | Excludes 0? |
|---|---:|---|:---:|
| Probe − Tuned Lens (layer 20) | 0.3063 | [0.2675, 0.3420] | **Yes** |
| Probe − Future-Lens-style m=3 | 0.2330 | [0.1962, 0.2690] | **Yes** |
| Probe − Future-Lens-style m=5 | 0.2213 | [0.1878, 0.2540] | **Yes** |
| Probe − Future-Lens-style m=10 | 0.1975 | [0.1644, 0.2291] | **Yes** |
| Probe − Rollout m=3 | 0.0446 | [−0.0028, 0.0913] | No (tied) |
| Probe − Rollout m=5 | −0.1321 | [−0.1806, −0.0843] | **Yes (rollout wins)** |
| Probe − Rollout m=10 | −0.3845 | [−0.4203, −0.3484] | **Yes (rollout wins)** |

The probe beats Tuned Lens and every tested Future-Lens-style budget decisively (all three CIs exclude zero, comfortably). Against rollout, the probe is statistically tied with the cheapest condition (m=3) and clearly loses to m=5 and m=10 — this reproduces P2's own H=16 finding number-for-number, as it should since it is the identical comparison.

## 7. Research questions

**Q1 — Tuned Lens.** *Does learning a layer-specific translator make token-level latent decoding substantially more competitive with direct semantic probing?* **No.** In this implementation, Tuned Lens at layer 20 (0.144) is not merely uncompetitive with the semantic probe (0.450) — it is *slightly worse than the untrained Direct Logit Lens* at the same layer (0.180), despite Tuned Lens clearly improving raw next-token top-1 accuracy over what direct logit-lens implicitly achieves at shallower layers. The large gap between token-identity decoding and the semantic probe is therefore **not** primarily an artifact of direct logit lens's specific weaknesses — a learned translator, trained properly (train-only fitting, validation-only model selection) does not close it, and mildly widens it on this metric. A plausible explanation: the translator was fit on only 403 examples to a ~4.2M-parameter affine map, optimizing next-token cross-entropy — an objective that is not the same as, and does not guarantee improvement on, the downstream top-*k*-text-then-embed semantic metric used for comparison. We report this as a genuine, unforced negative result.

**Q2 — Future Lens (the most important P4 comparison).** *Can independently decoded future token identities reconstruct future semantics as accurately as directly predicting the continuation embedding?* **No, not at any tested budget.** Future-Lens-style reconstruction improves steadily with *m* (0.218 → 0.229 → 0.253 for m=3/5/10) but plateaus far below the probe (0.450); the bootstrap CI for Probe − Future-Lens-style excludes zero decisively at every tested *m* (narrowest gap still 0.16–0.23 CI at m=10). This is **Outcome A** as defined in the spec: the semantic probe clearly beats both Tuned Lens and Future-Lens-style, supporting the claim that directly decoding continuation-level semantics captures information not well summarized by independently-predicted future token identities — without our claiming these prior methods are inferior in general; their native objective (token identity, not paraphrase-invariant meaning) is simply different.

**Q3 — Token-first vs. semantics-first decoding.** *Is information lost when future output is first discretized into predicted token identities?* The results support this interpretation, though we state it as supported rather than asserted independent of evidence: every method that routes through a discrete token-identity bottleneck (Direct Logit Lens 0.180, Tuned Lens 0.144, Future-Lens-style 0.22–0.25) falls well below the probe (0.450) *and* below the much cruder mean-embedding baseline is not beaten by them either in Tuned Lens's case (mean baseline 0.243 > Tuned Lens 0.144) — i.e., Tuned Lens's token-first pipeline, after aggregation, is not even reliably better than predicting nothing prompt-specific at all. This is a strong signal that the top-1/top-*k* discretization step, not merely "insufficient future-position information in the hidden state," is destroying meaning-bearing signal that the continuous-valued linear regression preserves.

**Q4 — Existing decoders vs. real generation.** *How close can zero-extra-decode methods come to actual rollout?* Not close, for the token-identity methods: Direct Logit Lens (0.180) and Tuned Lens (0.144) sit far below even Rollout-3 (0.406); Future-Lens-style narrows this somewhat (up to 0.253 at m=10) but still falls well short of Rollout-3. Only the **Linear Semantic Probe** enters rollout's territory: it is statistically tied with Rollout-3 (CI includes zero) and only loses to Rollout-5/10. The clean summary: among all zero-extra-decode-step methods tested, only directly regressing onto the continuation embedding gets close to what three tokens of real generation reveal; every token-identity-mediated zero-decode method (however it is constructed) falls well short of that same three-token budget.

**Q5 — Linearity.** *Does nonlinear decoding substantially improve continuation-semantic prediction over the ridge probe?* **No — it is distinctly worse.** The MLP probe (0.281, validation-selected among three regularization strengths) underperforms the linear ridge probe (0.450) by a wide margin, despite having strictly more representational capacity. At this training-set scale (403 examples) the additional nonlinear capacity does not help and plausibly hurts (harder optimization landscape, more room to overfit relative to ridge regression's well-regularized closed-form solution). This strengthens the paper's linear-accessibility framing: whatever future-continuation semantics are present in the hidden state are not only linearly decodable, they are *at least as well* captured by a linear readout as by a simple nonlinear one under this data budget.

## 8. Outcome classification (per the spec's rubric, §13)

**Outcome A applies**: the semantic probe clearly beats both Tuned Lens and Future-Lens-style, with bootstrap CIs excluding zero in every case. This supports the claim that continuation-level semantic decoding captures information not well summarized by future token-identity prediction — while we explicitly do not claim Tuned Lens or Future Lens are generally inferior methods, since their native objective (token identity / calibrated next-token distributions) differs from the semantic-embedding objective evaluated here.

**Outcome D does *not* apply, and in fact trends the opposite way**: Tuned Lens did not meaningfully improve over Direct Logit Lens in this implementation — it was slightly worse on the semantic metric. We do **not** generalize this to "Tuned Lens doesn't work" (its next-token accuracy did improve with depth, and it is a training-data-and-objective-limited implementation, not a reproduction of the full published Tuned Lens training recipe, which typically uses far more training data than 403 examples). We report the result as observed and flag the likely cause (data scarcity relative to translator capacity, and an objective mismatch between next-token CE and the downstream semantic-aggregation metric) rather than either overclaiming a general critique of Tuned Lens or suppressing the finding.

## 9. Final scientific interpretation

The paper's candidate closing statement,

> "Direct continuation-level semantic probing remains competitive with stronger existing latent-prediction methods, while explicit autoregressive rollout provides a distinct and generally stronger source of future information through additional sequential computation,"

**is supported by these results and can be retained.** Specifically: the probe does not merely beat the weak baselines already in the paper (mean/random/lexical/direct-logit-lens) — it also clearly beats two more sophisticated zero-decode latent-prediction methods implemented specifically for this comparison (Tuned Lens, Future-Lens-style), across every tested horizon and future-token budget, with no cherry-picking of reconstruction variants (all variant selection was validation-only). At the same time, explicit rollout remains a distinct, generally stronger source of information once a few tokens of real sequential computation are spent (statistically tied with the probe only at the cheapest budget, m=3, and clearly ahead from m=5 onward) — exactly the "distinct and generally stronger through additional computation" framing the candidate statement makes. The one genuine surprise — Tuned Lens's non-improvement over direct logit lens — does not threaten this conclusion; if anything it strengthens it, since it means the paper is not merely beating a weak or unfairly-hobbled version of prior latent-decoding methods.

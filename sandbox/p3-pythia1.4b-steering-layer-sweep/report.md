# P3 — Steering-Specific Layer Sweep on Pythia-1.4B — Report

**Project**: `p3-pythia1.4b-steering-layer-sweep` | **Model**: `EleutherAI/pythia-1.4b`, base, greedy generation, n_layers=24, hidden_size=2048 | **Pipeline**: exact finalized norm-relative steering protocol reused unchanged from `pythia-scaling-and-controls` / `steering-variance-and-1b-recheck` (same injection hook, sentiment-axis direction construction, random-direction control, margin definition, pilot-then-multiseed structure, t-interval 95% CI). **Runtime**: prep ~2s, k-sweep ~5min, multiseed ~2min.

## 0. Layer convention and the layer-0 substitution

"Layer" = raw `hidden_states` index (0=embeddings, 24=final block output) — the same convention already used for Pythia-1.4B everywhere else in this project (`hidden_states_layer_20.pt`, the accepted steering run's `"layer": 20` field), **not** the block+1 offset convention used internally by the earlier `pythia-1b-layer-sweep` script.

`BatchedInjectionHook` hooks `gpt_neox.layers[layer_idx-1]` and asserts `layer_idx >= 1`, so **layer 0 (raw embeddings, before any transformer block) has no injectable analogue**. Per the spec's instruction to document any such mapping deviation: the steering sweep substitutes **layer 1** (the output of the first transformer block, the earliest steerable location) in place of layer 0. Layer 0 is still reported in full in the passive-probe table (Section 1), since passive probing reads `hidden_states[0]` directly with no injection required. Layer 24 required no substitution — it is a valid injection point (`gpt_neox.layers[23]`, the last block).

**Reproduction check (required before interpreting the sweep)**: this run's independently-repiloted layer 20 result is mean margin **0.04517**, 95% CI **[0.0366, 0.0537]**, selected k=8.0 — close to and overlapping the prior accepted reference (mean 0.0404, CI [0.0335, 0.0472], k=6.4). The small difference (this run's pilot, using only 40 examples, happened to select k=8.0 rather than 6.4 — both are near the top of a fairly flat, noisy pilot curve at layer 20) is well within the seed/sampling variance this paper's own §4.5 methodological-lessons section already documents. **The layer-20 reference reproduces successfully.**

## 1. Passive probe reference by layer

| Layer | Passive probe cosine | Normalized depth |
|---:|---:|---:|
| 0 | 0.2775 | 0.000 |
| 4 | 0.3363 | 0.167 |
| 8 | 0.3412 | 0.333 |
| 12 | 0.3670 | 0.500 |
| 16 | 0.4066 | 0.667 |
| **20** | **0.4283** | 0.833 |
| 24 | 0.4220 | 1.000 |

Passive decodability rises monotonically from layer 0 through layer 20, then dips very slightly at the final layer (24) — layer 20 remains the clear passive optimum, consistent with every prior experiment using Pythia-1.4B.

## 2. Per-layer k-pilot winners

| Layer | Best k (dense grid) | Pilot margin |
|---:|---:|---:|
| 1 (sub. for 0) | 1.6 | +0.00611 |
| 4 | 0.1 | +0.00401 |
| 8 | 1.0 | +0.01591 |
| 12 | 0.2 | +0.01400 |
| 16 | 1.0 | +0.03386 |
| 20 | 8.0 | +0.04769 |
| 24 | 8.0 | +0.02544 |

Full pilot curves for every layer × 14 k-values are in `k_sweep_results.csv` and plotted in `figure3_k_pilot_curves.png` (all layers, weak and strong, preserved per the spec's requirement not to discard negative/weak pilot results).

## 3. Final multi-seed steering evaluation (5 seeds each)

| Layer | Best k | Mean margin | Std | 95% CI | CI excludes 0? |
|---:|---:|---:|---:|---|:---:|
| 1 (sub. for 0) | 1.6 | 0.00099 | 0.00251 | [−0.00213, 0.00411] | No |
| 4 | 0.1 | −0.00108 | 0.00255 | [−0.00425, 0.00209] | No |
| 8 | 1.0 | 0.00233 | 0.00624 | [−0.00542, 0.01007] | No |
| 12 | 0.2 | 0.00163 | 0.00396 | [−0.00329, 0.00655] | No |
| **16** | 1.0 | **−0.00742** | 0.00357 | **[−0.01185, −0.00298]** | **Yes (negative)** |
| **20** | 8.0 | **0.04517** | 0.00687 | **[0.03663, 0.05371]** | **Yes (positive)** |
| **24** | 8.0 | **0.03124** | 0.00919 | **[0.01983, 0.04264]** | **Yes (positive)** |

Every value above is from the 5-seed distribution — no single-seed result is used as a headline number, per the spec's requirement.

## 4. Primary analysis

**L*_probe = layer 20** (passive cosine 0.4283, the maximum among all 7 scanned layers).
**L*_steer = layer 20** (mean margin 0.04517, the maximum among all 7 scanned layers).

**Is the steering-optimal layer also layer 20? Yes.** This is **Outcome B — approximate alignment** as defined in the spec: the passive-best and steering-best layers coincide exactly in Pythia-1.4B.

With one important qualification: **layer 24 is close behind** (margin 0.0312, CI [0.0198, 0.0426], overlapping layer 20's CI in its upper range). A paired seed-level comparison (same 5 seeds, `paired_diff_layer20_vs_24.json`) gives Δ(20−24) = 0.0139, 95% CI **[−0.0046, 0.0325] — includes zero**, so layer 20 is *not* statistically distinguishable from layer 24 with only 5 paired seeds. This softens a pure "Outcome B" reading toward a **mild Outcome C (plateau at the deep layers, 20 and 24 both clearly positive and not significantly different from each other)** — while the four shallower/mid layers (1, 4, 8, 12) all have CIs that include zero, cleanly separating a "deep, steerable" region (20, 24) from a "shallow/mid, not detectably steerable" region.

**Bonus finding (Outcome D — sign reversal, preserved and not over-interpreted):** layer 16 shows a **significantly negative** margin (−0.0074, CI excludes zero) — steering in the nominally "positive" direction there measurably pushes output sentiment the *wrong* way, on 5/5 seeds with consistent sign. This directly parallels the unexplained layer-4 negative-steering anomaly found for Pythia-1.0B in the paper's existing §4.5 — Pythia-1.4B has an analogous reversal phenomenon at a different (much deeper, proportionally) layer. Per the spec's instruction, we report this without proposing a mechanism and flag it for future work (§7 below).

## 5. Passive–causal relationship across layers

| Statistic | Value |
|---|---:|
| Pearson r | 0.615 (p=0.141) |
| Spearman rho | 0.571 (p=0.180) |

Positive but **not statistically significant** with only 7 layers — exactly the caveat the spec anticipated ("do not overinterpret a non-significant correlation with only 7 layers"). Descriptively: stronger passive decodability trends toward stronger steering margin across this scan, but the relationship is driven substantially by the two deep layers (20, 24) being high on both axes and the negative layer-16 outlier pulling the correlation down from what it would be if steering strength tracked passive cosine monotonically — layer 16 has the *second-highest* passive cosine (0.4066) but the *most negative* steering margin, which is the single biggest reason this correlation is not stronger.

## 6. Depth-normalized results (for cross-model comparison)

| Steer layer | Normalized depth | Passive cosine (matched layer) | Steering margin |
|---:|---:|---:|---:|
| 1 | 0.042 | 0.2775 (layer 0) | 0.00099 |
| 4 | 0.167 | 0.3363 | −0.00108 |
| 8 | 0.333 | 0.3412 | 0.00233 |
| 12 | 0.500 | 0.3670 | 0.00163 |
| 16 | 0.667 | 0.4066 | −0.00742 |
| 20 | 0.833 | 0.4283 | 0.04517 |
| 24 | 1.000 | 0.4220 | 0.03124 |

## 7. Figures

- `figure1_passive_vs_causal.png` — two aligned panels sharing the layer axis: passive probe cosine (top) and steering margin with 95% CI (bottom). Visually confirms the peaks coincide at layer 20, with the layer-16 negative dip clearly visible directly below its comparatively high passive-cosine point.
- `figure2_steering_by_layer.png` — the main P3 figure: steering margin by layer with 95% CI error bars, zero line, layer 20 marked.
- `figure3_k_pilot_curves.png` — all 7 per-layer k-pilot curves (log-x), preserved for reproducibility/supplement per the spec.

## 8. Cross-model comparison

| Model | Passive-best layer | Steering-best layer | Same? | Best steering margin | 95% CI |
|---|---:|---:|---|---:|---|
| Pythia-1.0B | 13 | 6 | No | 0.0317 | [0.0230, 0.0404] |
| **Pythia-1.4B (this run)** | **20** | **20** | **Yes*** | **0.0452** | **[0.0366, 0.0537]** |
| Qwen2.5-1.5B | 24 | 28 | No | 0.1283 | [0.1230, 0.1337] |
| Qwen3-1.7B | 24 | 14 | No | 0.0306 | [0.0211, 0.0401] |
| Qwen3-4B | 32 | 31 | Approximately | 0.0740 | [0.0704, 0.0778] |

*Pythia-1.4B is the only model in this table where the nominal best-layer match is exact, though layer 24 is a statistically indistinguishable close second (§4 above) — so even the "Yes" here is closer to "yes, within a plateau" than to a sharp unique optimum.

## 9. Written verdict

**1. Which layer is best for passive semantic decoding in Pythia-1.4B?** Layer 20 (cosine 0.4283), consistent with every prior Pythia-1.4B result in this project.

**2. Which layer is best for causal semantic steering?** Layer 20 (mean margin 0.0452, 95% CI [0.0366, 0.0537]), with layer 24 a close, statistically indistinguishable second (0.0312, CI [0.0198, 0.0426]).

**3. Are those layers the same?** Yes — this is the first model in the cross-model comparison table where the nominal passive-best and steering-best layers coincide exactly.

**4. If different, is the steering-best layer significantly better than layer 20?** N/A in the strict sense (they are the same layer), but we ran the natural follow-up question anyway: is layer 20 significantly better than its closest competitor, layer 24? **No** — the paired seed-level difference (0.0139, 95% CI [−0.0046, 0.0325]) includes zero, so Pythia-1.4B shows a **plateau of steerability across its two deepest scanned layers**, not a single sharply-defined optimum.

**5. Is passive probe quality correlated with steering strength across depth?** Weakly, descriptively: Pearson r=0.615, Spearman rho=0.571, neither significant at n=7. The correlation is muted specifically because layer 16 combines high passive cosine (0.4066, second-highest of all 7) with a significantly *negative* steering margin — the clearest single piece of evidence in this experiment that passive decodability and causal steerability are not the same underlying quantity, even in the one model where their *optima* happen to coincide.

**6. Are there layers with zero or negative causal effects despite positive passive recoverability?** Yes, two kinds: layers 1, 4, 8, and 12 all have positive (or near-zero) passive cosine but steering-margin CIs that include zero (statistically null causal effect); layer 16 is the more striking case — clearly positive passive cosine (0.4066, higher than layers 0–12) but a **significantly negative** steering margin (CI excludes zero, all 5 seeds same sign).

**7. How does the Pythia-1.4B pattern compare with Pythia-1.0B and the Qwen models?** Pythia-1.4B is the outlier in the cross-model table: every other model tested so far (1.0B, Qwen2.5-1.5B, Qwen3-1.7B) shows a clear dissociation between passive-best and steering-best layers, and even Qwen3-4B's "approximate" match is at ~86–89% depth for both, not an exact index match. Pythia-1.4B alone shows the two optima landing on the literal same layer. Taken together with the layer-16 reversal and the 20-vs-24 plateau, the more precise reading is: **Pythia-1.4B's passive-best layer happens to also be a steering-best layer, but this is not because passive quality drives steering strength in this model either — it is one point of coincidence inside a still-noisy, non-monotonic passive/causal relationship** (rising passive cosine from layer 0 to 20, but a negative causal dip at layer 16 right in the middle of that rise).

**8. What exact claim about passive decodability versus causal steerability is justified after this experiment?** The paper's existing dissociation claim (established via Pythia-1.0B and generalized via the Qwen models) should be amended, not abandoned: **layer dissociation between passive semantic decoding and causal steerability is real and appears in most models tested (1.0B, Qwen2.5-1.5B, Qwen3-1.7B, and in a softer form Qwen3-4B), but it is not universal — the anchor model used throughout the main results, Pythia-1.4B, is a genuine counterexample where the two optima coincide.** At the same time, Pythia-1.4B is not simply "aligned" in a clean, monotonic sense: it has its own dissociation evidence in the form of the layer-16 negative-margin anomaly, and its steering-best layer is not uniquely identified (layers 20 and 24 are statistically tied). The most defensible paper-level statement is: *passive decodability and causal steerability are related but distinct properties whose alignment across layers is model-dependent — sometimes their global optima coincide (Pythia-1.4B), and even when they do, intermediate layers can still show clear dissociation or reversal (Pythia-1.4B's layer 16, Pythia-1.0B's layer 4) that a two-point (best-vs-best) comparison alone would miss.*

## Constraints honored

No layer was added or removed after seeing results; layer 0's steering substitution (→ layer 1) was decided and documented before the k-sweep/multiseed stages ran, from the injection hook's `assert layer_idx >= 1`, not from any observed result; every layer received its own independent k-pilot (no reuse of layer 20's k for other layers); the layer-16 negative result and the four null-CI shallow/mid layers were kept and reported, not discarded; layer selection was never touched by test-set performance (k was chosen on the pilot/val pool only, layers were fixed in advance by the spec); the layer-20 reference was reproduced before drawing any conclusions (§0).


## 10. Layer-indexing correction (post-hoc audit)

The Pythia-1.0B numbers in §8 above (steering-best layer, and the negative-steering anomaly referenced in §4) were re-labeled after an indexing audit. The source project (`pythia-1b-layer-sweep`) internally scans **0-indexed transformer *block* numbers** (`BLOCKS = [0, 3, 5, 8, 10, 12, 15]`) and saves that block number directly in its `layer` column/field, whereas this project (P3) and every Qwen steering script use the **raw `hidden_states` index** convention throughout (`hidden_states[l]` for l≥1 = output of 0-indexed block `l-1`; `layer_idx` passed straight to the injection hook). The two conventions differ by exactly 1: block `b` = hidden_states index `b+1`.

Concretely: `pythia-1b-layer-sweep/artifacts/pythia_1b_layer_sweep/layer_summary.csv` reports `"layer": 5` as the best block, with `"layer": 3` showing the significant negative margin — under the raw hidden_states convention used everywhere else in this paper, these are **hidden_states index 6** and **hidden_states index 4** respectively. The underlying margin values, CIs, and k/alpha are unaffected by this correction — only the layer *number* used to refer to them changes (5→6, 3→4). No retraining or recomputation was needed; the hidden_states-index-6 and index-4 results were already present in the source project's saved `layer_summary.csv`, just filed under their block-number label instead.

This does **not** affect any number reported elsewhere in this report for Pythia-1.4B, Qwen2.5-1.5B, Qwen3-1.7B, or Qwen3-4B — all four of those pipelines pass raw hidden_states indices directly to their injection hooks and self-document this convention in their own saved configs (`steering_layer_candidates.json` for the Qwen models; direct `layer_idx` pass-through for P3/Pythia-1.4B), confirmed by direct code inspection.

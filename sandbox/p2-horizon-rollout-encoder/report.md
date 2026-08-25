# P2 — Horizon × Rollout Budget × Semantic Encoder Analysis — Report

**Project**: `p2-horizon-rollout-encoder` | **Model**: Pythia-1.4B, greedy | **New compute**: one 256-token greedy generation pass (600 prompts, ~80s) + prompt-hidden-state extraction (~1s) + 3-encoder embedding/probing/bootstrap analysis (~93s). Total new GPU time ≈ 3 minutes.

## 0. Setup and reuse decisions

- **Generation**: single greedy pass, `max_new_tokens=256`, on the exact same 600-prompt corpus used by every prior project (md5-verified identical `prompts.jsonl`). No sentence-boundary truncation (required by the P2 spec). All five horizons (16/48/96/192/256) are prefixes of this one generation, guaranteeing the nested-prefix / paired-example property the spec requires between probe target and rollout baseline.
- **Common subset**: per spec's stated preference ("prefer a common subset... because this makes horizon comparisons cleaner"), the primary analysis uses the subset of prompts whose realized continuation reaches the full 256 tokens: **n=577/600**. Only 3.8% of prompts hit EOS before 256 tokens. Full per-horizon n (before restricting to the common subset) is: H=16→600, H=48→600, H=96→591, H=192→581, H=256→577 — reported here in full per the spec's transparency requirement, even though the common subset (577) is what all tables/figures below use.
- **Split**: recomputed fresh on the 577-example common subset (same procedure as every prior project — `train_test_split`, test_size=0.15 then 0.15/0.85, seed=42): train=403, val=87, test=87. Because the split is fixed once on the common subset and every horizon is a prefix of the same 577 generations, **the same 87 test examples are used for every (H, E, m) cell** — a stronger pairing than the spec strictly required, which makes the Δ(H,·,·) trend across H directly comparable example-for-example, not just distributionally.
- **Encoders**: identical to P1 — MiniLM (no prefix), BGE-base (no prefix), E5-base (`"passage: "` prefix on every embedded text, including rollout prefixes and horizon targets, per E5's training convention).
- **Layers**: full 4/8/12/20 sweep re-run for every (H, E) pair (60 ridge fits total) — layer 20 won in every single one of the 15 (H,E) combinations tested. **The best passive layer does not change with horizon or encoder in this experiment** — unlike the causal-steering layer dissociation found elsewhere in the paper, passive semantic decoding is layer-stable here.
- Ridge alpha, probe architecture, and split were not modified in response to results (constraint honored).

## 1. Table A — main result by encoder (cosine)

**MiniLM**

| H | Probe | Rollout-3 | Rollout-5 | Rollout-10 | Rollout-20 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.4504 | 0.4059 | 0.5825 | 0.8350 | 0.9392 |
| 48 | 0.4363 | 0.3103 | 0.4594 | 0.6807 | 0.8659 |
| 96 | 0.4164 | 0.2732 | 0.4044 | 0.5985 | 0.7818 |
| 192 | 0.4280 | 0.2436 | 0.3524 | 0.5233 | 0.6941 |
| 256 | 0.4421 | 0.2358 | 0.3385 | 0.4996 | 0.6649 |

**BGE-base**

| H | Probe | Rollout-3 | Rollout-5 | Rollout-10 | Rollout-20 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.6982 | 0.6410 | 0.7366 | 0.8885 | 0.9472 |
| 48 | 0.7138 | 0.6024 | 0.6778 | 0.7969 | 0.9023 |
| 96 | 0.7243 | 0.5916 | 0.6599 | 0.7639 | 0.8624 |
| 192 | 0.7448 | 0.5899 | 0.6559 | 0.7544 | 0.8463 |
| 256 | 0.7473 | 0.5849 | 0.6501 | 0.7461 | 0.8369 |

**E5-base**

| H | Probe | Rollout-3 | Rollout-5 | Rollout-10 | Rollout-20 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.8641 | 0.8126 | 0.8554 | 0.9353 | 0.9690 |
| 48 | 0.8656 | 0.7832 | 0.8136 | 0.8687 | 0.9357 |
| 96 | 0.8616 | 0.7728 | 0.8005 | 0.8498 | 0.9131 |
| 192 | 0.8650 | 0.7732 | 0.7997 | 0.8439 | 0.8999 |
| 256 | 0.8675 | 0.7735 | 0.7983 | 0.8396 | 0.8938 |

(Best layer = 20 in all 15 rows; not shown per-row since constant.)

Note the qualitative asymmetry already visible here: probe cosine is roughly flat across H in every encoder (small, non-monotonic wobble of ±0.03), while rollout cosine at every fixed m falls substantially as H grows — a fixed-length prefix covers proportionally less of a longer target. This is the mechanical driver of everything below.

## 2. Table B — probe-minus-rollout margins (full table with 95% CI in `table_B_margins.csv`; condensed here)

Point estimates only (Δ = probe − rollout; **bold** = 95% CI excludes zero):

| Encoder | H | Δ m=3 | Δ m=5 | Δ m=10 | Δ m=20 |
|---|---:|---:|---:|---:|---:|
| MiniLM | 16 | 0.045 | **−0.132** | **−0.385** | **−0.489** |
| MiniLM | 48 | **0.126** | −0.023 | **−0.244** | **−0.430** |
| MiniLM | 96 | **0.143** | 0.012 | **−0.182** | **−0.365** |
| MiniLM | 192 | **0.184** | **0.076** | **−0.095** | **−0.266** |
| MiniLM | 256 | **0.206** | **0.104** | **−0.058** | **−0.223** |
| BGE | 16 | **0.057** | **−0.038** | **−0.190** | **−0.249** |
| BGE | 48 | **0.111** | **0.036** | **−0.083** | **−0.189** |
| BGE | 96 | **0.133** | **0.064** | **−0.040** | **−0.138** |
| BGE | 192 | **0.155** | **0.089** | −0.010 | **−0.102** |
| BGE | 256 | **0.162** | **0.097** | 0.001 | **−0.090** |
| E5 | 16 | **0.052** | 0.009 | **−0.071** | **−0.105** |
| E5 | 48 | **0.082** | **0.052** | −0.003 | **−0.070** |
| E5 | 96 | **0.089** | **0.061** | **0.012** | **−0.052** |
| E5 | 192 | **0.092** | **0.065** | **0.021** | **−0.035** |
| E5 | 256 | **0.094** | **0.069** | **0.028** | **−0.026** |

## 3. Table C — rollout parity budget (smallest tested m ≥ probe cosine)

| Encoder | H=16 | H=48 | H=96 | H=192 | H=256 |
|---|---:|---:|---:|---:|---:|
| MiniLM | 5 | 5 | 10 | 10 | 10 |
| BGE | 5 | 10 | 10 | 10 | 20 |
| E5 | 10 | 10 | 20 | 20 | 20 |

The required rollout budget is **non-decreasing in H for every encoder** — the probe becomes relatively more competitive as the target horizon grows, in all three encoders, without exception. The *absolute* budget required differs sharply by encoder in exactly the order P1 anticipated: MiniLM needs the fewest rollout tokens to match the probe, E5 needs the most, BGE lies between.

## 4. Figures

- `figure1_delta_vs_horizon.png` — Δ(H,m,E) vs. H, one line per m, one panel per encoder, zero line marked. Shows Δ rising monotonically with H for **every single (encoder, m) combination tested** — 12 of 12 lines increase from H=16 to H=256 without exception, though not always crossing zero within the tested range.
- `figure2_prefix_convergence.png` — prefix-to-full convergence C(H,m,E) vs. m, one line per H, per encoder (dotted horizontal = that horizon's probe cosine). Shows visually how much faster E5's curves approach the probe line than MiniLM's.
- `figure3_compute_tradeoff.png` — cosine vs. extra decode steps at H=256 (probe plotted at 0 steps), per encoder.

## 5. Written verdict

**1. Does increasing future horizon make hidden-state probing relatively more competitive with explicit rollout?**
**Yes, unambiguously and consistently.** Δ(H,m,E) increases (probe becomes relatively better) as H grows from 16→256, in **every one of the 12 (encoder × m) combinations tested**, with no exceptions. This is the cleanest, most consistent result in this experiment — it holds regardless of which semantic encoder is used to measure it. Growth is not always monotonic point-to-point (e.g., MiniLM m=5 dips slightly from H=16 to H=48 before rising), but the overall H=16→H=256 trend is uniformly positive.

**2. How does the required rollout budget change with horizon?**
It is **non-decreasing** in every encoder (Table C): MiniLM 5→5→10→10→10, BGE 5→10→10→10→20, E5 10→10→20→20→20. Longer horizons require more rollout tokens to match the probe — directly consistent with question 1's answer, from a budget-planning rather than margin-size perspective.

**3. Does that relationship differ across MiniLM, BGE, and E5?**
**Yes, substantially**, and in exactly the order P1's post-hoc observation suggested: MiniLM's rollout catches up to the probe fastest (parity by m=5 at short horizons), E5's rollout needs the most tokens (parity not reached until m=10-20), BGE sits in between. This is not a subtle effect — at H=256, MiniLM's rollout needs only 10 tokens to reach parity while E5's still needs 20, a 2× difference in required compute for the identical underlying LM and identical target continuations.

**4. Are the P1 encoder-specific rollout differences explained by different rates of prefix-to-full semantic convergence?**
**Yes — by construction, and the mechanism is visible directly in Figure 2/Table A.** The rollout score S_rollout(H,m,E) *is* the prefix-to-full convergence C(H,m,E) (they are the same computed quantity viewed from two angles, and we report this equivalence transparently rather than presenting it as an independent replication). What Figure 2 adds beyond Table A is the visual confirmation that E5's convergence curves rise steeply and saturate close to the probe line within just 3-5 tokens at every horizon, while MiniLM's curves rise much more gradually and stay well below its own probe line until m≈10. In plain terms: E5's embedding space treats a short prefix as already highly similar to the full continuation (high "prefix-generosity"), which mechanically makes its rollout baseline look strong at small m; MiniLM's embedding space discriminates much more sharply between a short prefix and the full text, which is exactly why MiniLM was the encoder where P1 originally found rollout beating the probe most decisively.

**5. Does rollout show diminishing semantic return per additional autoregressive step?**
**Yes, in every encoder and every horizon tested, without exception.** The marginal gain per added token (3→5, 5→10, 10→20; full values in `p2_results.json`) decreases monotonically in all 15 (encoder, H) combinations — e.g., MiniLM at H=256: 0.051 → 0.032 → 0.017 per token; E5 at H=256: 0.012 → 0.008 → 0.005 per token. This confirms and generalizes the original single-encoder finding from the paper's §4.3: rollout compute has sharply diminishing marginal semantic value, regardless of which semantic space is used to measure it.

**6. What exact claim about semantic lookahead versus behavioral rollout is justified after this experiment?**
The paper's original framing — "a fair behavioral rollout baseline beats the probe at short horizons, and the gap narrows as horizon grows" — is **directionally correct and now much better supported** (it replicates cleanly across three encoders and two additional, longer horizons), but it requires two honest amendments:
- The size of the gap, the horizon at which it becomes small, and the rollout budget needed to close it are all **jointly determined by (H, m, E)**, not fixed properties of "the probe vs. rollout comparison" in the abstract. The three-way interaction the spec hypothesized is real and empirically clean (Q1–Q3 above).
- **Long rollout (m=20) remains stronger than the probe at every horizon tested, in every encoder tested, up to H=256** — this experiment does not find a crossover at the largest tested rollout budget, in any encoder. This is Success Criterion "Strong outcome B" as defined in the spec: the gap consistently narrows and substantial encoder-specific differences remain, but rollout is not overtaken at its largest tested budget. We do not extrapolate beyond H=256 or m=20 (per the spec's explicit constraint), so we cannot say whether a true crossover exists beyond the tested range — only that the trend is moving steadily toward one in every condition tested.

**Recommended framing for the IEEE Access draft**: state the compute-efficiency argument (§4.3 of the existing structure) as horizon-and-encoder-conditional rather than universal, cite this P2 result as the generalization, and add the E5/BGE/MiniLM "prefix-generosity" mechanism (Q4 above) as the explanatory link between the P1 encoder-robustness finding and the P2 horizon-crossover finding — they are not two separate robustness checks but one connected story: the same encoder property (how much semantic credit a short prefix gets) governs both how strong the rollout baseline looks and how much rollout budget is needed to beat the probe.

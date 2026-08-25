# Pythia-1.0b causal steering layer sweep

## Hypothesis

The accepted multi-seed run (`steering-variance-and-1b-recheck/`) found pythia-1.0b's causal steering margin to be effectively zero (mean -0.00002, 95% CI [-0.00307, 0.00303]) at layer=13 (hidden_states index), the layer chosen because it was BEST for PASSIVE semantic probing, not necessarily for causal steering. This run tests whether that null result is a layer-selection artifact (some other layer IS steerable) or a real model-level property (no layer works).

## Model config

- `model_name`: EleutherAI/pythia-1b
- `n_layers` (transformer blocks, programmatically read from config): 16
- `hidden_size`: 2048

Candidate layers (0-indexed transformer block number, per the protocol formula with n_layers=16): **[0, 3, 5, 8, 10, 12, 15]** (7 distinct layers spanning early/mid/late; block *b* is steered internally at hidden_states index *b+1*, i.e. the output of `gpt_neox.layers[b]`).

Pipeline reused unchanged from the accepted norm-relative steering run: probe-derived sentiment direction `d_sem = normalize(W.T @ v_sem)` (a ridge probe trained per-layer here, identical procedure/alpha-grid to the original passive-probing probe trainer), per-prompt random-direction control (`RandomState(200000+prompt_id)`), injection at the last real prompt token on the prefill step only, greedy decoding (`MAX_NEW_TOKENS=24`), the sentiment-axis scoring metric (`score(text)=cos(emb,pos)-cos(emb,neg)`), the seed-42 train/val/test split (val=83 as pilot, test=83 as final eval), the dense k-grid `DENSE_K_GRID_1B` reused verbatim from `steering-variance-and-1b-recheck` (`[0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6, 2.4, 3.2, 4.8, 6.4, 8.0]`), the norm-relative alpha rule `alpha = k * mean_hidden_norm(layer, val split)`, 5 seeds per layer (`[101, 202, 303, 404, 505]`), and the t-interval 95% CI method.

## Per-layer k-sweep winner

| layer (block) | selected_k | final_alpha |
|---:|---:|---:|
| 0 | 1.0 | 31.0858 |
| 3 | 2.4 | 87.4741 |
| 5 | 8.0 | 361.1448 |
| 8 | 8.0 | 401.5651 |
| 10 | 1.6 | 92.1519 |
| 12 | 0.2 | 18.3111 |
| 15 | 3.2 | 322.0446 |

## Per-layer steering margin (mean +/- 95% CI, n seeds)

| layer (block) | n_seeds | mean_margin | std_margin | 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 5 | +0.00416 | 0.00479 | [-0.00178, 0.01010] |
| 3 | 5 | -0.01378 | 0.00671 | [-0.02211, -0.00545] |
| 5 | 5 | +0.03167 | 0.00700 | [0.02298, 0.04037] |
| 8 | 5 | +0.01268 | 0.00692 | [0.00408, 0.02128] |
| 10 | 5 | +0.00724 | 0.00440 | [0.00177, 0.01271] |
| 12 | 5 | +0.00019 | 0.00262 | [-0.00307, 0.00345] |
| 15 | 5 | +0.00169 | 0.00623 | [-0.00604, 0.00943] |

**Best layer by mean margin: block 5**, mean_margin=+0.03167, 95% CI=[0.02298, 0.04037] (CI excludes zero).

## Comparison to prior model-size results and the old 1.0b layer-13 result

| model / layer | mean margin | 95% CI |
|---|---:|---:|
| pythia-410m | 0.0565 | [0.043, 0.07] |
| pythia-1.4b | 0.0404 | [0.033, 0.047] |
| pythia-2.8b | 0.0087 | [0.005, 0.0124] |
| pythia-1.0b (OLD, layer=13, passive-probing-optimal) | -2e-05 | [-0.00307, 0.00303] |
| pythia-1.0b (THIS RUN, best layer=5) | +0.03167 | [0.02298, 0.04037] |

The best pythia-1.0b layer's mean margin (+0.03167) is 0.56x the 410m margin (0.0565), 0.78x the 1.4b margin (0.0404), and 3.64x the 2.8b margin (0.0087). This is comparable in order of magnitude to the other three model sizes.

## Final verdict

**layer-selection artifact: at least one pythia-1.0b layer has CI excluding zero**

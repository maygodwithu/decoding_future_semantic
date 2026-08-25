# Norm-relative alpha scaling report
## Setup (exact prior config reused)
Reused unchanged from the accepted run in `pythia-scaling-and-controls/`: probe-derived sentiment direction construction (`d_sem = normalize(W.T @ v_sem)`), per-prompt random-direction control (`RandomState(200000+prompt_id)`), injection site (output of `gpt_neox.layers[layer-1]`, last prompt token, prefill step only), greedy decoding with `MAX_NEW_TOKENS=24`, the sentiment-axis scoring metric, the seed-42 train/val/test split, and each model's own accepted `best_layer` (410m=20, 1.0b=13, 1.4b=20, 2.8b=27). The `val` split (83 prompts) is used as the pilot split for hidden-norm measurement and k-selection (n_pilot=40 sampled prompts); the `test` split (83 prompts) is the same held-out test split scored in the accepted run (n_test=60 sampled prompts, identical eligibility filter + `RandomState(42)` shuffle + cap). Pilot and test prompt pools are disjoint by construction (see `pilot_prompts.json` / `test_prompts.json`).
k grid swept on pilot only: [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.4, 0.8, 1.6, 3.2, 6.4]
## Do mean hidden norms increase with model size?
| model | mean_hidden_norm |
|---|---|
| pythia-410m | 48.077 |
| pythia-1.0b | 91.555 |
| pythia-1.4b | 173.205 |
| pythia-2.8b | 272.339 |

**Yes** -- mean hidden-state norm at the intervention site is monotonically non-decreasing across 410m -> 1.0b -> 1.4b -> 2.8b.

## Did the old fixed raw alpha imply a much smaller relative k at 2.8b?
| model | old_raw_alpha | mean_hidden_norm | old_implied_k |
|---|---|---|---|
| pythia-410m | 99.12 | 48.08 | 2.06163 |
| pythia-1.0b | 200.23 | 91.55 | 2.18702 |
| pythia-1.4b | 271.26 | 173.20 | 1.56615 |
| pythia-2.8b | 576.74 | 272.34 | 2.11773 |

Mean old_implied_k across the three smaller models = 1.93827; 2.8b old_implied_k = 2.11773 (NOT much smaller, ratio to small-model mean = 1.093).

## Is 2.8b's new margin rescued to the smaller models' ballpark?
| model | old_raw_alpha | mean_hidden_norm | old_implied_k | best_k_pilot | new_raw_alpha | old_margin | new_margin | delta_margin |
|---|---|---|---|---|---|---|---|---|
| pythia-410m | 99.12 | 48.08 | 2.06163 | 6.4000 | 307.69 | 0.003432 | 0.053391 | 0.049959 |
| pythia-1.0b | 200.23 | 91.55 | 2.18702 | 0.4000 | 36.62 | 0.011275 | 0.004318 | -0.006956 |
| pythia-1.4b | 271.26 | 173.20 | 1.56615 | 6.4000 | 1108.51 | 0.008364 | 0.035302 | 0.026938 |
| pythia-2.8b | 576.74 | 272.34 | 2.11773 | 3.2000 | 871.48 | 0.000042 | 0.001017 | 0.000975 |

Mean smaller-model old_margin = 0.007690, mean smaller-model new_margin = 0.031004. 2.8b old_margin = 0.000042, 2.8b new_margin = 0.001017 (delta = +0.000975).

**2.8b new margin is NOT rescued -- it remains far below / near zero relative to the smaller models.**

## Paper-facing conclusion
**COLLAPSE PERSISTS AFTER NORMALIZATION**

Even after selecting alpha per model as a pilot-tuned multiple of that model's own hidden-state norm at the intervention site, the 2.8b semantic-vs-random steering margin remains near zero / far below the smaller models. This indicates the near-zero margin at 2.8b is not fully explained by the fixed-raw-alpha methodology artifact, and is at least partly a genuine scale-dependent reduction in causal steerability of this probe-derived direction under this protocol.
